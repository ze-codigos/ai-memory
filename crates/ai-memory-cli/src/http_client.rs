//! Shared HTTP-client glue for thin-client CLI subcommands.
//!
//! Every state-touching subcommand (status, search, bootstrap, …) goes
//! through these helpers so URL resolution + bearer-auth handling stays
//! consistent in one place.
//!
//! ## Configuration
//!
//! [`crate::config::Config`] captures `AI_MEMORY_SERVER_URL` and
//! `AI_MEMORY_AUTH_TOKEN` exactly once; this module consumes those values
//! and can fall back to the stored OIDC device-flow token used by native hooks.

use std::fmt;
use std::io::{BufWriter, Write as _};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::header::{HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::config::{Config, DEFAULT_SERVER_URL};
use ai_memory_web::normalize_prefix;

/// Env var carrying extra static headers for hook/MCP HTTP delivery.
///
/// Format: one `Name: value` pair per `\n`-separated line. Designed for edge
/// authenticating proxies (Cloudflare Access sends its user JWT in
/// `cf-access-token`, which the server's own bearer slot cannot carry): a
/// wrapping tool resolves the short-lived credential and exports it right
/// before invoking the CLI. Values are read at request time and are therefore
/// deliberately never serialized into the hook spool — a token valid at
/// capture time is routinely expired by drain time.
pub(crate) const EXTRA_HEADERS_ENV: &str = "AI_MEMORY_HTTP_EXTRA_HEADERS";

/// Header names the extra-header channel must not override: `Authorization`
/// belongs to the configured bearer path and `Content-Type` to the JSON
/// payload serialization. Conflicting lines are skipped with a warning rather
/// than producing duplicate headers on the wire.
const EXTRA_HEADERS_RESERVED: [&str; 2] = ["authorization", "content-type"];

/// Parse one `Name: value` line into a validated header pair. Returns `None`
/// (after warning) for malformed lines, reserved names, or values containing
/// control bytes — the request path must never fail over bad configuration.
pub(crate) fn parse_header_pair(line: &str) -> Option<(HeaderName, HeaderValue)> {
    let line = line.trim();
    let (name, value) = line.split_once(':')?;
    let name = name.trim();
    let value = value.trim();
    if name.is_empty() || value.is_empty() {
        eprintln!(
            "ai-memory warning: ignoring malformed extra header line (expected 'Name: value'): {line:?}"
        );
        return None;
    }
    if EXTRA_HEADERS_RESERVED.contains(&name.to_ascii_lowercase().as_str()) {
        eprintln!(
            "ai-memory warning: extra header {name:?} is reserved and was skipped (configure auth via the token settings instead)"
        );
        return None;
    }
    let header_name = HeaderName::from_bytes(name.as_bytes()).ok()?;
    let header_value = HeaderValue::from_str(value).ok()?;
    Some((header_name, header_value))
}

/// Env var naming a command that resolves extra headers AT REQUEST TIME.
///
/// [`EXTRA_HEADERS_ENV`] is a value frozen into the process environment at
/// launch, which is fine for a hook that lives milliseconds and wrong for
/// `ai-memory run`, which lives as long as the agent session: a Cloudflare
/// Access token exported at 10:46 is expired by the `finish` at 14:29, the
/// proxy answers 302 and the whole transcript import fails. The command named
/// here (POSIX shell-style quoting so a path may contain spaces, but executed
/// directly, never through a shell) prints the same `Name: value` lines to
/// stdout and is re-run before a request whenever the cached output is older
/// than [`HEADER_COMMAND_CACHE_TTL`]. Its headers override same-named static
/// ones. Only [`ServerEndpoint::authenticate`] honours it — the long-lived
/// launcher's path; hook and bridge requests read the static lines only.
///
/// Empty output, a failing command, or a command that does not finish within
/// [`HEADER_COMMAND_TIMEOUT`] leave the static headers untouched — the request
/// path must never fail over configuration.
pub(crate) const EXTRA_HEADERS_CMD_ENV: &str = "AI_MEMORY_HTTP_EXTRA_HEADERS_CMD";

/// How long one command output is reused. The managed heartbeat fires every
/// 30s; forking a credential helper for each of those would be pure waste,
/// while a minute-old token is as good as a fresh one against a 24h session.
const HEADER_COMMAND_CACHE_TTL: Duration = Duration::from_secs(60);

/// Upper bound on one command execution. A helper stuck on the network must
/// not hang the `finish` forever; past this the request goes out with the
/// static headers and the server's answer says what happened.
const HEADER_COMMAND_TIMEOUT: Duration = Duration::from_secs(15);

/// Process-wide memo of the last header-command execution. One entry is
/// enough: a process configures a single command (the env var is read from
/// the process environment, which does not change).
#[derive(Debug)]
pub(crate) struct HeaderCommandCache {
    entry: Option<HeaderCommandEntry>,
}

#[derive(Debug)]
struct HeaderCommandEntry {
    resolved_at: Instant,
    output: String,
}

impl HeaderCommandCache {
    pub(crate) const fn new() -> Self {
        Self { entry: None }
    }

    /// Return the memoised output when it is younger than `ttl`, else run
    /// `run`. Only a SUCCESS is memoised: `finish_with_retry` retries a failed
    /// request within a second, and a memoised failure would hand every retry
    /// the same stale static header. A broken helper therefore runs once per
    /// request — in the launcher that is one heartbeat every 30s.
    pub(crate) fn resolve(
        &mut self,
        now: Instant,
        ttl: Duration,
        run: impl FnOnce() -> Option<String>,
    ) -> Option<String> {
        if let Some(entry) = &self.entry
            && now.saturating_duration_since(entry.resolved_at) < ttl
        {
            return Some(entry.output.clone());
        }
        let output = run()?;
        self.entry = Some(HeaderCommandEntry {
            resolved_at: now,
            output: output.clone(),
        });
        Some(output)
    }
}

/// How often a running header command is polled for completion.
const HEADER_COMMAND_POLL: Duration = Duration::from_millis(50);

/// Execute `command` (whitespace-split argv, no shell) and return its stdout.
/// Warns and returns `None` on spawn failure, non-zero exit, or `timeout`. A
/// child that outruns the timeout is killed and reaped — nothing outlives
/// this call, so a hung helper costs one warning per cache TTL, not a leaked
/// process per heartbeat.
///
/// Output is read only after the child exits, so a helper printing more than
/// the pipe buffer (64 KiB on Linux) would block on write until the timeout
/// kills it. A header line is a few hundred bytes; that is by design.
pub(crate) fn execute_header_command(command: &str, timeout: Duration) -> Option<String> {
    use std::io::Read as _;
    use std::process::{Command, Stdio};

    let Some(mut argv) = shlex::split(command).filter(|argv| !argv.is_empty()) else {
        eprintln!(
            "ai-memory warning: extra-header command {command:?} is empty or has unbalanced quotes; using the static headers"
        );
        return None;
    };
    let program = argv.remove(0);
    let spawned = Command::new(&program)
        .args(&argv)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            eprintln!("ai-memory warning: extra-header command {command:?} could not run: {error}");
            return None;
        }
    };
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(HEADER_COMMAND_POLL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                eprintln!(
                    "ai-memory warning: extra-header command {command:?} did not finish within {timeout:?} and was killed; using the static headers"
                );
                return None;
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                eprintln!(
                    "ai-memory warning: extra-header command {command:?} could not be waited on: {error}"
                );
                return None;
            }
        }
    };
    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_string(&mut stdout);
    }
    if status.success() {
        return Some(stdout);
    }
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    eprintln!(
        "ai-memory warning: extra-header command {command:?} exited with {status}: {}",
        stderr.trim()
    );
    None
}

fn parse_header_lines(raw: &str) -> Vec<(HeaderName, HeaderValue)> {
    raw.split('\n').filter_map(parse_header_pair).collect()
}

/// The static extra headers only: [`EXTRA_HEADERS_ENV`] via `lookup`, the
/// command channel ignored. What hooks and the MCP bridge use.
pub(crate) fn static_extra_headers_from(
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<(HeaderName, HeaderValue)> {
    extra_headers_from_with(lookup, |_| None)
}

/// Static lines from [`EXTRA_HEADERS_ENV`], then the output of
/// [`EXTRA_HEADERS_CMD_ENV`] resolved through `runner`, the latter overriding
/// the former name by name.
pub(crate) fn extra_headers_from_with(
    lookup: impl Fn(&str) -> Option<String>,
    mut runner: impl FnMut(&str) -> Option<String>,
) -> Vec<(HeaderName, HeaderValue)> {
    let mut headers = lookup(EXTRA_HEADERS_ENV)
        .map(|raw| parse_header_lines(&raw))
        .unwrap_or_default();
    let command = lookup(EXTRA_HEADERS_CMD_ENV)
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty());
    if let Some(command) = command
        && let Some(output) = runner(&command)
    {
        for (name, value) in parse_header_lines(&output) {
            headers.retain(|(existing, _)| *existing != name);
            headers.push((name, value));
        }
    }
    headers
}

/// Stamp the STATIC extra headers onto a hook request from the process
/// environment. [`EXTRA_HEADERS_CMD_ENV`] is deliberately ignored here: a
/// hook process lives milliseconds and its wrapper hands it a fresh static
/// header, so forking a credential helper per hook request would be waste.
/// The command channel belongs to the long-lived launcher, which reaches it
/// through [`ServerEndpoint::authenticate`].
pub(crate) fn apply_extra_headers(req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    apply_static_extra_headers_with(|k| std::env::var(k).ok(), req)
}

/// Testable core of [`apply_extra_headers`]: static lines only.
pub(crate) fn apply_static_extra_headers_with(
    lookup: impl Fn(&str) -> Option<String>,
    req: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    apply_extra_headers_with_runner(lookup, |_| None, req)
}

/// Resolve every extra header (static lines plus the command channel through
/// `runner`) via `lookup` and stamp them onto `req`.
pub(crate) fn apply_extra_headers_with_runner(
    lookup: impl Fn(&str) -> Option<String>,
    runner: impl FnMut(&str) -> Option<String>,
    req: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    let mut req = req;
    for (name, value) in extra_headers_from_with(lookup, runner) {
        req = req.header(name, value);
    }
    req
}

/// Non-success response returned by the configured ai-memory server.
#[derive(Debug)]
pub(crate) struct ServerResponseError {
    method: reqwest::Method,
    path: String,
    status: reqwest::StatusCode,
    body: String,
}

impl ServerResponseError {
    #[cfg(test)]
    pub(crate) fn for_tests(status: reqwest::StatusCode, body: String) -> Self {
        Self {
            method: reqwest::Method::POST,
            path: "/workstream/runs/x/finish".to_owned(),
            status,
            body,
        }
    }

    #[must_use]
    pub(crate) const fn status(&self) -> reqwest::StatusCode {
        self.status
    }

    #[must_use]
    pub(crate) fn body(&self) -> &str {
        &self.body
    }
}

impl fmt::Display for ServerResponseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {}: server returned {}: {}",
            self.method, self.path, self.status, self.body
        )?;
        if self.status.is_redirection() {
            // The API itself never redirects and the client never follows:
            // a 3xx is either an edge auth proxy (Cloudflare Access) bouncing
            // an expired credential to its login page, or a scheme/host
            // redirect that AI_MEMORY_SERVER_URL should already point past.
            write!(
                formatter,
                " (a redirect the client will not follow: an auth-proxy login wall — extra-header credential missing or expired — or AI_MEMORY_SERVER_URL pointing at a redirecting scheme/host)"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ServerResponseError {}

fn server_response_error(
    method: reqwest::Method,
    url: &reqwest::Url,
    status: reqwest::StatusCode,
    body: String,
) -> anyhow::Error {
    ServerResponseError {
        method,
        path: url.path().to_owned(),
        status,
        body,
    }
    .into()
}

/// Pass a 2xx response through, else consume the body into a
/// [`ServerResponseError`].
///
/// Only the request path reaches the error, so userinfo and query
/// credentials in the URL never land in a message. `reqwest` does not carry
/// the request method on the response, so callers supply it.
async fn require_success(
    method: reqwest::Method,
    resp: reqwest::Response,
) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let url = resp.url().clone();
    let body = resp.text().await.unwrap_or_default();
    Err(server_response_error(method, &url, status, body))
}

/// Client for the thin-client helpers below.
///
/// Redirects are NOT followed, mirroring `hook_capture::build_client`: the
/// server API never redirects, so a 3xx can only be an auth wall. Following it
/// turned an expired Cloudflare Access token into a `200 text/html` login page
/// that `.json()` then rejected with "expected value at line 1 column 1" —
/// the `finish` of every long `ai-memory run` died that way. With redirects
/// off the 3xx reaches [`server_response_error`] and says what it is.
pub(crate) fn build_client() -> reqwest::Client {
    // `Client::new()` panics on the same TLS-backend failure this would
    // report, so expecting here changes nothing about failure and keeps the
    // redirect policy unconditional (a fallback to `new()` would follow them).
    client_builder()
        .build()
        .expect("reqwest client with the default TLS backend")
}

/// The one place the "never follow a redirect" policy is spelled out. The
/// hook client layers `no_proxy` on top; everything else builds from here.
pub(crate) fn client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder().redirect(reqwest::redirect::Policy::none())
}

/// Resolved server target — origin URL + base-path prefix + optional bearer token.
#[derive(Debug, Clone)]
pub struct ServerEndpoint {
    /// Origin only (scheme + authority), any trailing slash and path
    /// stripped, e.g. `http://127.0.0.1:49374` or `http://192.168.0.90:49374`.
    pub url: String,
    /// Normalised base-path prefix the server is mounted under, e.g.
    /// `/wiki`, or empty when serving at the root. Always either empty or
    /// `/`-prefixed with no trailing slash (normalised by `normalize_prefix`).
    pub base_path: String,
    /// Bearer token when present, else `None`.
    pub auth_token: Option<String>,
    url_configured: bool,
    /// Memo of the [`EXTRA_HEADERS_CMD_ENV`] output, owned by the endpoint
    /// (no process-wide state): the launcher's one endpoint serves every
    /// request of the session; clones share it, a fresh endpoint starts cold.
    header_command: Arc<Mutex<HeaderCommandCache>>,
}

impl ServerEndpoint {
    /// Build the endpoint from config, resolving bearer auth and base path.
    ///
    /// Bearer precedence matches hooks: static config/env token first, stored
    /// OIDC device token second, no token last.
    ///
    /// The base-path prefix the server is mounted under, so client routes
    /// resolve as `<origin><base><path>` instead of 404ing) is resolved
    /// from, in order of precedence:
    /// 1. the **path component of `AI_MEMORY_SERVER_URL`** (the remote-client
    ///    case — `http://host:49374/wiki` → origin `http://host:49374`,
    ///    base `/wiki`), then
    /// 2. **`Config::base_path`** (the in-pod case — figment populates the
    ///    field from `AI_MEMORY_BASE_PATH` inside `Config::load`, keeping the
    ///    "one config-read path" invariant; the CLI runs in the same
    ///    container as `serve`, which already reads the same env var via
    ///    clap to nest its router).
    pub async fn from_config_resolving_auth(config: &Config) -> Self {
        let client = reqwest::Client::new();
        let token = crate::auth_bearer::resolve_bearer(
            &client,
            &config.oidc_device_token_path(),
            config.auth.bearer_token.as_deref(),
        )
        .await;
        Self::build(
            Some(config.server_url.clone()),
            token,
            config.server_url_configured(),
            Some(config.base_path.clone()).filter(|s| !s.is_empty()),
        )
    }

    /// Build from an explicit URL + token pair, for callers that have both
    /// already and no `Config` to resolve them from.
    ///
    /// The lifecycle hook is the real case: `main` dispatches it before
    /// `Config::load` runs, so it carries its own `--server-url` and resolves
    /// its own bearer. The detached drainer inherits the same situation.
    /// Tests use it too, since it reads no environment.
    ///
    /// `url` defaults to `http://127.0.0.1:49374` when `None` or empty;
    /// trailing slashes are stripped. `token` is treated as absent when
    /// `None` or empty. The env base-path fallback is `None`; use
    /// [`from_pair_with_base`] to exercise that path.
    #[must_use]
    pub(crate) fn from_pair(url: Option<String>, token: Option<String>) -> Self {
        let url_configured = url.as_deref().is_some_and(|s| !s.is_empty());
        Self::build(url, token, url_configured, None)
    }

    /// Like [`from_pair`] but with an explicit `AI_MEMORY_BASE_PATH` env
    /// fallback value, so the env-fallback branch is testable hermetically.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn from_pair_with_base(
        url: Option<String>,
        token: Option<String>,
        env_base: Option<String>,
    ) -> Self {
        let url_configured = url.as_deref().is_some_and(|s| !s.is_empty());
        Self::build(url, token, url_configured, env_base)
    }

    fn build(
        url: Option<String>,
        token: Option<String>,
        url_configured: bool,
        env_base: Option<String>,
    ) -> Self {
        let raw = url
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_SERVER_URL.to_string());
        let (origin, url_path) = split_origin_and_path(&raw);
        // The URL path component wins over the env var: it is the more
        // explicit, request-target-level statement of where the server is.
        let base_path = if url_path.is_empty() {
            env_base.map_or_else(String::new, |s| normalize_prefix(&s))
        } else {
            normalize_prefix(&url_path)
        };
        Self {
            url: origin,
            base_path,
            auth_token: token.filter(|s| !s.is_empty()),
            url_configured,
            header_command: Arc::new(Mutex::new(HeaderCommandCache::new())),
        }
    }

    /// Join the resolved origin + base-path prefix + a root-absolute route
    /// `path` (e.g. `/admin/status`) into the full request URL. When no base
    /// path is configured this is byte-identical to the old
    /// `format!("{origin}{path}")`.
    pub(crate) fn build_url(&self, path: &str) -> String {
        format!("{}{}{path}", self.url, self.base_path)
    }

    /// Stable, credential-free identity for client-local state tied to this
    /// server. The normalized mount path is part of the identity; bearer
    /// material deliberately is not.
    pub(crate) fn identity(&self) -> String {
        let raw = format!("{}{}", self.url.trim_end_matches('/'), self.base_path);
        let Ok(mut parsed) = reqwest::Url::parse(&raw) else {
            return "<invalid-server-url>".to_owned();
        };
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
        parsed.set_query(None);
        parsed.set_fragment(None);
        parsed.as_str().trim_end_matches('/').to_owned()
    }

    /// Apply auth header to a `reqwest::RequestBuilder` if a token is set.
    ///
    /// ze-codigos: also stamps `AI_MEMORY_HTTP_EXTRA_HEADERS` here, so EVERY
    /// helper that funnels through `authenticate` (`get_json`, `post_json`,
    /// `patch_json`, `post_empty`, ... and therefore `run`, `bootstrap`,
    /// `status`, `embed`) reaches a server behind an edge-auth proxy.
    /// Previously only `hook_capture` applied them, which left `ai-memory run`
    /// unable to open a managed workstream through Cloudflare Access: the
    /// proxy answered 302 + text/html and the CLI died decoding JSON.
    /// `hook_capture` builds its request without `authenticate`, so headers
    /// are never stamped twice.
    pub(crate) fn authenticate(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        self.authenticate_with(|k| std::env::var(k).ok(), req)
    }

    /// Testable core of [`Self::authenticate`]: resolve the extra-header env
    /// via `lookup` instead of the process environment.
    pub(crate) fn authenticate_with(
        &self,
        lookup: impl Fn(&str) -> Option<String>,
        req: reqwest::RequestBuilder,
    ) -> reqwest::RequestBuilder {
        let req = apply_extra_headers_with_runner(
            lookup,
            |command| self.run_header_command(command),
            req,
        );
        match &self.auth_token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    /// Run the header command through this endpoint's memo. The lock is held
    /// while the command runs; requests through one endpoint are sequential
    /// (one heartbeat at a time, then the finish), so nothing waits on it.
    ///
    /// The command is a blocking wait of up to [`HEADER_COMMAND_TIMEOUT`]
    /// issued from inside async request helpers. On a multi-thread runtime it
    /// is wrapped in `block_in_place` so the worker keeps driving other
    /// tasks; a current-thread runtime (tests) or no runtime just blocks.
    fn run_header_command(&self, command: &str) -> Option<String> {
        let resolve = || {
            let mut cache = self
                .header_command
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.resolve(Instant::now(), HEADER_COMMAND_CACHE_TTL, || {
                execute_header_command(command, HEADER_COMMAND_TIMEOUT)
            })
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(resolve)
            }
            _ => resolve(),
        }
    }
}

/// Split a server URL into `(origin, path)` where `origin` is
/// scheme+authority (`http://host:port`) and `path` is everything from the
/// first `/` after the authority (`/wiki`, possibly empty). Trailing slashes
/// are trimmed from both. A URL with no scheme separator is treated as an
/// opaque origin with no path.
pub(crate) fn split_origin_and_path(raw: &str) -> (String, String) {
    let trimmed = raw.trim_end_matches('/');
    if let Some(scheme_end) = trimmed.find("://") {
        let after_scheme = scheme_end + 3;
        if let Some(rel_slash) = trimmed[after_scheme..].find('/') {
            let split = after_scheme + rel_slash;
            return (
                trimmed[..split].to_string(),
                trimmed[split..].trim_end_matches('/').to_string(),
            );
        }
    }
    (trimmed.to_string(), String::new())
}

/// GET `<endpoint>{path}` with optional query params, deserialise JSON.
///
/// # Errors
/// Returns an error when the connection fails, the response is non-2xx,
/// or the body can't be deserialised into `T`.
pub async fn get_json<T: DeserializeOwned>(
    endpoint: &ServerEndpoint,
    path: &str,
    query: &[(&str, &str)],
) -> Result<T> {
    let client = build_client();
    let url = endpoint.build_url(path);
    let mut req = client.get(&url);
    if !query.is_empty() {
        req = req.query(query);
    }
    req = endpoint.authenticate(req);
    let resp = req
        .send()
        .await
        .map_err(|e| augment_connect_error(e, endpoint, &url))?;
    let resp = require_success(reqwest::Method::GET, resp).await?;
    resp.json::<T>()
        .await
        .with_context(|| format!("parsing JSON body from GET {url}"))
}

/// PATCH JSON body to `<endpoint>{path}`, deserialise JSON response.
///
/// # Errors
/// Same as [`get_json`].
pub async fn patch_json<B: Serialize, T: DeserializeOwned>(
    endpoint: &ServerEndpoint,
    path: &str,
    body: &B,
) -> Result<T> {
    let client = build_client();
    let url = endpoint.build_url(path);
    let req = endpoint.authenticate(client.patch(&url).json(body));
    let resp = req
        .send()
        .await
        .map_err(|e| augment_connect_error(e, endpoint, &url))?;
    let resp = require_success(reqwest::Method::PATCH, resp).await?;
    resp.json::<T>()
        .await
        .with_context(|| format!("parsing JSON body from PATCH {url}"))
}

/// POST JSON body to `<endpoint>{path}`, deserialise JSON response.
///
/// # Errors
/// Same as [`get_json`].
pub async fn post_json<B: Serialize, T: DeserializeOwned>(
    endpoint: &ServerEndpoint,
    path: &str,
    body: &B,
) -> Result<T> {
    post_json_with_query(endpoint, path, &[], body).await
}

/// POST JSON and require a successful response without decoding its body.
pub async fn post_json_no_content<B: Serialize>(
    endpoint: &ServerEndpoint,
    path: &str,
    body: &B,
) -> Result<()> {
    let client = build_client();
    let url = endpoint.build_url(path);
    let req = endpoint.authenticate(client.post(&url).json(body));
    let resp = req
        .send()
        .await
        .map_err(|e| augment_connect_error(e, endpoint, &url))?;
    require_success(reqwest::Method::POST, resp).await?;
    Ok(())
}

/// POST an empty body and require a successful response.
pub async fn post_empty(endpoint: &ServerEndpoint, path: &str) -> Result<()> {
    let client = build_client();
    let url = endpoint.build_url(path);
    let req = endpoint.authenticate(client.post(&url));
    let resp = req
        .send()
        .await
        .map_err(|e| augment_connect_error(e, endpoint, &url))?;
    require_success(reqwest::Method::POST, resp).await?;
    Ok(())
}

/// POST JSON body to `<endpoint>{path}` with URL-encoded query params.
///
/// # Errors
/// Same as [`post_json`].
pub async fn post_json_with_query<B: Serialize, T: DeserializeOwned>(
    endpoint: &ServerEndpoint,
    path: &str,
    query: &[(&str, &str)],
    body: &B,
) -> Result<T> {
    let client = build_client();
    let url = build_url_with_query(endpoint, path, query)?;
    let req = client.post(&url);
    let req = endpoint.authenticate(req.json(body));
    let resp = req
        .send()
        .await
        .map_err(|e| augment_connect_error(e, endpoint, &url))?;
    let resp = require_success(reqwest::Method::POST, resp).await?;
    resp.json::<T>()
        .await
        .with_context(|| format!("parsing JSON body from POST {url}"))
}

fn build_url_with_query(
    endpoint: &ServerEndpoint,
    path: &str,
    query: &[(&str, &str)],
) -> Result<String> {
    let mut url = reqwest::Url::parse(&endpoint.build_url(path))?;
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(query.iter().copied());
    }
    Ok(url.to_string())
}

/// Turn a low-level reqwest connect/timeout error into a friendlier
/// message that surfaces the resolved server URL. The common case is
/// "Connection refused" — typically because the CLI defaulted to
/// loopback on a host that has no local server running.
fn augment_connect_error(
    err: reqwest::Error,
    endpoint: &ServerEndpoint,
    url: &str,
) -> anyhow::Error {
    // Walk the source chain to see if there's a Connection-refused
    // io::Error buried somewhere. reqwest wraps its errors deeply.
    let chain_contains_refused = {
        let mut src: Option<&dyn std::error::Error> = Some(&err);
        let mut found = false;
        while let Some(e) = src {
            if e.to_string().contains("Connection refused")
                || e.to_string().contains("connection refused")
            {
                found = true;
                break;
            }
            src = e.source();
        }
        found
    };

    if chain_contains_refused {
        let hint = if endpoint.url_configured {
            format!(
                "\nAI_MEMORY_SERVER_URL is set to {} but nothing answered. \
                 Check the server is running, the port is reachable from \
                 this host, and (if remote) any firewall + bearer-token \
                 config matches.",
                endpoint.url
            )
        } else {
            format!(
                "\nAI_MEMORY_SERVER_URL is NOT set; the CLI defaulted to \
                 {} and nothing answered. If your server lives on another \
                 machine (e.g. a homelab), `export AI_MEMORY_SERVER_URL=\
                 http://<server>:49374` and (if auth is on) \
                 `export AI_MEMORY_AUTH_TOKEN=<token>` before re-running.",
                endpoint.url
            )
        };
        anyhow::Error::new(err).context(format!("could not reach {url}.{hint}"))
    } else {
        anyhow::Error::new(err).context(format!("HTTP request to {url} failed"))
    }
}

/// POST an empty body to `<endpoint>{path}`, streaming the response to `dest`.
///
/// Intended for routes whose response is binary (e.g. `POST /admin/backup`
/// returns an `application/gzip` tarball). On non-2xx the response body is
/// consumed and returned as an error string. Returns bytes written.
///
/// # Errors
/// Returns an error when the connection fails, the response is non-2xx,
/// or the body cannot be read or written.
pub async fn post_to_file(endpoint: &ServerEndpoint, path: &str, dest: &Path) -> Result<u64> {
    let client = build_client();
    let url = endpoint.build_url(path);
    let req = endpoint.authenticate(client.post(&url));
    let resp = req
        .send()
        .await
        .map_err(|e| augment_connect_error(e, endpoint, &url))?;
    let mut resp = require_success(reqwest::Method::POST, resp).await?;
    let file = private_output_file(dest)
        .with_context(|| format!("creating output file {}", dest.display()))?;
    let mut writer = BufWriter::new(file);
    let mut written = 0_u64;
    while let Some(chunk) = resp
        .chunk()
        .await
        .with_context(|| format!("reading response chunk from POST {url}"))?
    {
        writer
            .write_all(&chunk)
            .with_context(|| format!("writing response chunk to {}", dest.display()))?;
        written += chunk.len() as u64;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", dest.display()))?;
    Ok(written)
}

/// Open a downloaded backup output file with private permissions before the
/// first response bytes are written. Existing user-selected output files keep
/// their existing permissions when overwritten.
fn private_output_file(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Extra headers (edge-proxy credentials) — parse + apply
    // ----------------------------------------------------------------

    #[test]
    fn parse_header_pair_accepts_a_trimmed_pair() {
        let (name, value) = parse_header_pair("  cf-access-token:  eyJabc.def  ").unwrap();
        assert_eq!(name.as_str(), "cf-access-token");
        assert_eq!(value.to_str().unwrap(), "eyJabc.def");
    }

    #[test]
    fn parse_header_pair_rejects_malformed_lines() {
        assert!(parse_header_pair("no-colon-here").is_none());
        assert!(parse_header_pair(": value").is_none());
        assert!(parse_header_pair("name: ").is_none());
        assert!(parse_header_pair("").is_none());
        assert!(parse_header_pair("bad name: value").is_none());
    }

    #[test]
    fn parse_header_pair_rejects_reserved_names() {
        assert!(parse_header_pair("Authorization:Bearer x").is_none());
        assert!(parse_header_pair("authorization: Bearer x").is_none());
        assert!(parse_header_pair("Content-Type: text/plain").is_none());
        assert!(parse_header_pair("content-type: text/plain").is_none());
    }

    #[test]
    fn extra_headers_from_parses_each_line_and_skips_garbage() {
        let lookup = |_k: &str| {
            Some(
                "cf-access-token: jwt-1\n\
                 \n\
                 broken-line\n\
                 Authorization: Bearer nope\n\
                 x-custom: yes"
                    .to_string(),
            )
        };
        let headers = static_extra_headers_from(lookup);
        assert_eq!(headers.len(), 2, "{headers:?}");
        assert_eq!(headers[0].0.as_str(), "cf-access-token");
        assert_eq!(headers[0].1.to_str().unwrap(), "jwt-1");
        assert_eq!(headers[1].0.as_str(), "x-custom");
    }

    #[test]
    fn extra_headers_from_returns_empty_when_env_absent() {
        assert!(static_extra_headers_from(|_| None).is_empty());
    }

    // ----------------------------------------------------------------
    // Extra headers resolved by a command at request time (MEM-3)
    // ----------------------------------------------------------------

    fn env(
        static_lines: Option<&'static str>,
        command: Option<&'static str>,
    ) -> impl Fn(&str) -> Option<String> {
        move |key| match key {
            EXTRA_HEADERS_ENV => static_lines.map(str::to_string),
            EXTRA_HEADERS_CMD_ENV => command.map(str::to_string),
            _ => None,
        }
    }

    #[test]
    fn command_headers_override_static_ones_by_name() {
        // O `run` vive horas: o token estatico exportado no lancamento vence,
        // o comando devolve o token vivo e tem de vencer o estatico.
        let headers = extra_headers_from_with(
            env(
                Some("cf-access-token: stale\nx-custom: keep"),
                Some("resolve"),
            ),
            |command| {
                assert_eq!(command, "resolve");
                Some("cf-access-token: fresh\n".to_string())
            },
        );
        let as_pairs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
            .collect();
        assert_eq!(
            as_pairs,
            vec![("x-custom", "keep"), ("cf-access-token", "fresh")]
        );
    }

    #[test]
    fn command_with_empty_output_keeps_the_static_headers() {
        let headers =
            extra_headers_from_with(env(Some("cf-access-token: stale"), Some("resolve")), |_| {
                Some(String::new())
            });
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].1.to_str().unwrap(), "stale");
    }

    #[test]
    fn command_failure_keeps_the_static_headers() {
        let headers =
            extra_headers_from_with(env(Some("cf-access-token: stale"), Some("resolve")), |_| {
                None
            });
        assert_eq!(headers.len(), 1);
    }

    #[test]
    fn blank_command_is_ignored() {
        let calls = std::cell::Cell::new(0);
        let _ = extra_headers_from_with(env(None, Some("   ")), |_| {
            calls.set(calls.get() + 1);
            None
        });
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn header_command_cache_reuses_the_output_within_the_ttl() {
        let mut cache = HeaderCommandCache::new();
        let ttl = Duration::from_secs(60);
        let t0 = Instant::now();
        let runs = std::cell::Cell::new(0);
        let run = || {
            runs.set(runs.get() + 1);
            Some("cf-access-token: a".to_string())
        };
        assert_eq!(
            cache.resolve(t0, ttl, run).as_deref(),
            Some("cf-access-token: a")
        );
        assert_eq!(
            cache
                .resolve(t0 + Duration::from_secs(30), ttl, run)
                .as_deref(),
            Some("cf-access-token: a")
        );
        assert_eq!(
            runs.get(),
            1,
            "a heartbeat every 30s must not fork the resolver every time"
        );
        let _ = cache.resolve(t0 + Duration::from_secs(61), ttl, run);
        assert_eq!(runs.get(), 2, "past the TTL the command runs again");
    }

    #[test]
    fn header_command_cache_does_not_memoise_a_failure() {
        // O finish faz 3 tentativas em menos de 1s; uma falha transitoria do
        // helper na 1a nao pode condenar as outras duas ao header estatico.
        let mut cache = HeaderCommandCache::new();
        let ttl = Duration::from_secs(60);
        let t0 = Instant::now();
        let runs = std::cell::Cell::new(0);
        let flaky = || {
            runs.set(runs.get() + 1);
            (runs.get() > 1).then(|| "cf-access-token: b".to_string())
        };
        assert!(cache.resolve(t0, ttl, flaky).is_none());
        assert_eq!(
            cache
                .resolve(t0 + Duration::from_millis(250), ttl, flaky)
                .as_deref(),
            Some("cf-access-token: b")
        );
        assert_eq!(runs.get(), 2, "the retry re-runs the helper");
        let _ = cache.resolve(t0 + Duration::from_millis(500), ttl, flaky);
        assert_eq!(runs.get(), 2, "the success is memoised as usual");
    }

    #[test]
    fn authenticate_honours_the_header_command_and_memoises_per_endpoint() {
        // O launcher usa UM endpoint pra sessao inteira: o comando roda no
        // 1o request e o memo serve os seguintes; outro endpoint comeca frio.
        let client = reqwest::Client::new();
        let endpoint = ServerEndpoint::from_pair(None, None);
        let lookup = env(
            Some("cf-access-token: stale"),
            Some("/bin/echo cf-access-token: live"),
        );
        let stamped = |endpoint: &ServerEndpoint| {
            endpoint
                .authenticate_with(&lookup, client.get("http://localhost/x"))
                .build()
                .unwrap()
                .headers()
                .get("cf-access-token")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        };
        assert_eq!(stamped(&endpoint).as_deref(), Some("live"));
        assert_eq!(stamped(&endpoint.clone()).as_deref(), Some("live"));
        assert_eq!(
            stamped(&ServerEndpoint::from_pair(None, None)).as_deref(),
            Some("live")
        );
    }

    #[cfg(unix)]
    #[test]
    fn execute_header_command_accepts_shell_style_quoting() {
        // Caminho de plugin com espaco: o wrapper exporta com `printf %q`.
        let dir = tempfile::tempdir().unwrap();
        let helper = dir.path().join("My Plugins").join("token-header");
        std::fs::create_dir_all(helper.parent().unwrap()).unwrap();
        std::fs::write(
            &helper,
            "#!/bin/sh\nprintf 'cf-access-token: %s\\n' \"$1\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&helper, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        let quoted = shlex::try_quote(helper.to_str().unwrap()).unwrap();
        let out = execute_header_command(&format!("{quoted} spaced"), Duration::from_secs(5));
        assert_eq!(
            out.as_deref().map(str::trim),
            Some("cf-access-token: spaced")
        );
        assert!(execute_header_command("'unbalanced", Duration::from_secs(5)).is_none());
    }

    #[test]
    fn hook_requests_ignore_the_header_command() {
        // Hook vive milissegundos e recebe header estatico do wrapper; forkar
        // o helper a cada POST de hook seria puro desperdicio.
        let client = reqwest::Client::new();
        let req = apply_static_extra_headers_with(
            env(
                Some("cf-access-token: static"),
                Some("/bin/echo cf-access-token: cmd"),
            ),
            client.post("http://localhost/hook"),
        )
        .build()
        .unwrap();
        assert_eq!(
            req.headers()
                .get("cf-access-token")
                .and_then(|v| v.to_str().ok()),
            Some("static")
        );
    }

    #[cfg(unix)]
    #[test]
    fn execute_header_command_runs_without_a_shell_and_captures_stdout() {
        let out = execute_header_command("/bin/echo cf-access-token: live", Duration::from_secs(5));
        assert_eq!(out.as_deref().map(str::trim), Some("cf-access-token: live"));
    }

    #[cfg(unix)]
    #[test]
    fn execute_header_command_reports_a_failing_command_as_none() {
        assert!(execute_header_command("/bin/false", Duration::from_secs(5)).is_none());
        assert!(execute_header_command("/nonexistent/resolver", Duration::from_secs(5)).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn execute_header_command_kills_the_child_at_the_timeout() {
        // Um helper pendurado (cloudflared num buraco negro de rede) nao pode
        // deixar um processo orfao por heartbeat: o filho e' morto e colhido.
        let marker = format!("ai-memory-mem3-{}", std::process::id());
        let started = Instant::now();
        assert!(
            execute_header_command(
                &format!("/bin/sleep 30 {marker}"),
                Duration::from_millis(200)
            )
            .is_none()
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        let survivors = std::process::Command::new("/bin/sh")
            .args([
                "-c",
                &format!("ps -eo args | grep -c '[s]leep 30 {marker}'"),
            ])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&survivors.stdout).trim(),
            "0",
            "the timed-out child must not survive the call"
        );
    }

    // ----------------------------------------------------------------
    // Auth-wall redirects surface as errors, never as a login page (MEM-3)
    // ----------------------------------------------------------------

    async fn serve_once(status: &'static str, extra: String, body: &'static str) -> String {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 2048];
            let _ = stream.read(&mut buf).await;
            let response = format!(
                "HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn post_json_does_not_follow_an_auth_wall_redirect() {
        // Regressao: o Cloudflare Access responde 302 pra um token vencido;
        // seguindo o redirect o cliente recebia a pagina de login em 200 e
        // morria com "expected value at line 1 column 1" ao decodificar JSON.
        let login = serve_once(
            "200 OK",
            "Content-Type: text/html\r\n".to_string(),
            "<html>Sign in</html>",
        )
        .await;
        // A live login page behind the Location: following it would turn the
        // 302 into this 200 text/html, which is the regression.
        let origin = serve_once("302 Found", format!("Location: {login}/login\r\n"), "").await;
        let endpoint = ServerEndpoint::from_pair(Some(origin), None);
        let error = post_json::<_, serde_json::Value>(
            &endpoint,
            "/workstream/runs/x/finish",
            &serde_json::json!({}),
        )
        .await
        .unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains("302"), "{text}");
        assert!(!text.contains("parsing JSON"), "{text}");
    }

    #[tokio::test]
    async fn get_json_does_not_follow_an_auth_wall_redirect() {
        let origin = serve_once(
            "302 Found",
            "Location: http://127.0.0.1:9/login\r\n".to_string(),
            "",
        )
        .await;
        let endpoint = ServerEndpoint::from_pair(Some(origin), None);
        let error = get_json::<serde_json::Value>(&endpoint, "/handoff", &[])
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("302"));
    }

    #[test]
    fn a_redirect_error_names_both_causes() {
        // Um 302 do Access e um 301 http->https recebem a mesma dica: o
        // texto tem de apontar a credencial E a URL, nao so a credencial.
        let error: anyhow::Error =
            ServerResponseError::for_tests(reqwest::StatusCode::FOUND, String::new()).into();
        let text = format!("{error:#}");
        assert!(text.contains("302"), "{text}");
        assert!(text.contains("expired"), "{text}");
        assert!(text.contains("AI_MEMORY_SERVER_URL"), "{text}");
    }

    #[test]
    fn a_plain_server_error_carries_no_credential_hint() {
        let error: anyhow::Error =
            ServerResponseError::for_tests(reqwest::StatusCode::BAD_GATEWAY, "boom".into()).into();
        let text = format!("{error:#}");
        assert!(text.contains("502"));
        assert!(!text.contains("expired"), "{text}");
    }

    #[test]
    fn apply_extra_headers_with_stamps_the_request() {
        let client = reqwest::Client::new();
        let req = apply_static_extra_headers_with(
            |_| Some("cf-access-token: jwt-2".to_string()),
            client.get("http://localhost"),
        )
        .build()
        .unwrap();
        assert_eq!(
            req.headers()
                .get("cf-access-token")
                .and_then(|v| v.to_str().ok()),
            Some("jwt-2")
        );
        // Reserved names never reach the wire through this channel.
        assert!(req.headers().get("authorization").is_none());
    }

    #[test]
    fn authenticate_stamps_extra_headers_so_every_helper_reaches_an_edge_proxy() {
        // Regressao: so o hook_capture aplicava os extra headers, entao
        // `ai-memory run` batia no Cloudflare Access (302 + text/html) ao abrir
        // o workstream e morria decodificando JSON. Todos os helpers HTTP
        // passam por `authenticate`, entao e aqui que o header tem de entrar.
        let client = reqwest::Client::new();
        let endpoint = ServerEndpoint::from_pair(None, Some("tok".to_string()));
        let req = endpoint
            .authenticate_with(
                |_| Some("cf-access-token: jwt-3".to_string()),
                client.post("http://localhost/workstream/runs"),
            )
            .build()
            .unwrap();
        assert_eq!(
            req.headers()
                .get("cf-access-token")
                .and_then(|v| v.to_str().ok()),
            Some("jwt-3")
        );
        // o bearer continua sendo aplicado
        assert!(req.headers().get("authorization").is_some());
    }

    #[test]
    fn apply_extra_headers_with_leaves_request_unchanged_when_unset() {
        let client = reqwest::Client::new();
        let req = apply_static_extra_headers_with(|_| None, client.get("http://localhost"))
            .build()
            .unwrap();
        assert!(req.headers().get("cf-access-token").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn downloaded_backup_output_is_private_when_created() {
        use std::os::unix::fs::PermissionsExt as _;

        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("backup.tar.gz");
        private_output_file(&output).unwrap();
        assert_eq!(
            std::fs::metadata(output).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn response_errors_include_method_and_credential_free_path() {
        let url = reqwest::Url::parse(
            "https://username:password@example.test/workstream/runs?access_token=secret",
        )
        .expect("test URL parses");
        let error = server_response_error(
            reqwest::Method::POST,
            &url,
            reqwest::StatusCode::NOT_FOUND,
            "missing".to_owned(),
        );

        assert_eq!(
            error.to_string(),
            "POST /workstream/runs: server returned 404 Not Found: missing"
        );
        let structured = error
            .downcast_ref::<ServerResponseError>()
            .expect("response errors retain their structured type");
        assert_eq!(structured.status(), reqwest::StatusCode::NOT_FOUND);
        assert_eq!(structured.body(), "missing");
        assert!(!error.to_string().contains("username"));
        assert!(!error.to_string().contains("password"));
        assert!(!error.to_string().contains("access_token"));
        assert!(!error.to_string().contains("secret"));
    }

    // ----------------------------------------------------------------
    // ServerEndpoint::from_pair
    // ----------------------------------------------------------------

    #[test]
    fn from_pair_defaults_to_loopback_when_none() {
        let ep = ServerEndpoint::from_pair(None, None);
        assert_eq!(ep.url, "http://127.0.0.1:49374");
        assert!(ep.auth_token.is_none());
    }

    #[test]
    fn from_pair_defaults_to_loopback_when_empty() {
        let ep = ServerEndpoint::from_pair(Some(String::new()), None);
        assert_eq!(ep.url, "http://127.0.0.1:49374");
    }

    #[test]
    fn from_pair_strips_trailing_slash() {
        let ep = ServerEndpoint::from_pair(Some("http://10.0.0.1:8080/".to_string()), None);
        assert_eq!(ep.url, "http://10.0.0.1:8080");
    }

    #[test]
    fn from_pair_strips_multiple_trailing_slashes() {
        let ep = ServerEndpoint::from_pair(Some("http://10.0.0.1:8080///".to_string()), None);
        assert_eq!(ep.url, "http://10.0.0.1:8080");
    }

    #[test]
    fn from_pair_empty_token_treated_as_none() {
        let ep = ServerEndpoint::from_pair(None, Some(String::new()));
        assert!(ep.auth_token.is_none());
    }

    #[test]
    fn from_pair_non_empty_token_preserved() {
        let ep = ServerEndpoint::from_pair(None, Some("secret".to_string()));
        assert_eq!(ep.auth_token.as_deref(), Some("secret"));
    }

    #[test]
    fn client_state_identity_normalizes_mounts_and_never_contains_credentials() {
        let first = ServerEndpoint::from_pair(
            Some("http://alice:secret@MEMORY.example:49374/wiki/".to_owned()),
            None,
        );
        let second = ServerEndpoint::from_pair(
            Some("http://memory.example:49374/wiki".to_owned()),
            Some("bearer-secret".to_owned()),
        );

        assert_eq!(first.identity(), second.identity());
        assert!(!first.identity().contains("alice"));
        assert!(!first.identity().contains("secret"));
    }

    // ----------------------------------------------------------------
    // Base-path awareness — graduated from the Docker live exploration of
    // the CLI-client-under-AI_MEMORY_BASE_PATH bug. The server nests every
    // route under `--base-path` (`/wiki/admin/status`), but the thin client
    // baked root-absolute paths (`/admin/status`) → 404. Each test below
    // pins one hypothesis that the live run validated end-to-end.
    // ----------------------------------------------------------------

    /// H1 — no base path configured: the joined URL is byte-identical to the
    /// old `format!("{origin}{path}")`. The regression guard for the OFF-by-
    /// default promise.
    #[test]
    fn build_url_without_base_is_byte_identical() {
        let ep = ServerEndpoint::from_pair(Some("http://h:49374".to_string()), None);
        assert_eq!(ep.base_path, "");
        assert_eq!(ep.build_url("/admin/status"), "http://h:49374/admin/status");
    }

    /// H4/H9 — remote client: the base path comes from the **path component
    /// of the server URL**, which is split off the origin and normalised
    /// (trailing slash trimmed).
    #[test]
    fn server_url_path_becomes_base_path() {
        let ep = ServerEndpoint::from_pair(Some("http://h:49374/wiki".to_string()), None);
        assert_eq!(ep.url, "http://h:49374");
        assert_eq!(ep.base_path, "/wiki");
        assert_eq!(
            ep.build_url("/admin/status"),
            "http://h:49374/wiki/admin/status"
        );

        // Trailing slash on the URL path normalises away (H9).
        let ep = ServerEndpoint::from_pair(Some("http://h:49374/wiki/".to_string()), None);
        assert_eq!(ep.url, "http://h:49374");
        assert_eq!(ep.base_path, "/wiki");
    }

    /// H3 — in-pod client: no path on the server URL, base path falls back to
    /// the `AI_MEMORY_BASE_PATH` env value (passed explicitly here to stay
    /// hermetic).
    #[test]
    fn env_base_path_used_when_url_has_no_path() {
        let ep = ServerEndpoint::from_pair_with_base(None, None, Some("/wiki".to_string()));
        assert_eq!(ep.url, "http://127.0.0.1:49374");
        assert_eq!(ep.base_path, "/wiki");
        assert_eq!(
            ep.build_url("/admin/write-page"),
            "http://127.0.0.1:49374/wiki/admin/write-page"
        );
    }

    /// H8 — precedence: when BOTH a URL path and an env base are present, the
    /// URL path wins (it is the more explicit, request-target-level value).
    #[test]
    fn url_path_wins_over_env_base() {
        let ep = ServerEndpoint::from_pair_with_base(
            Some("http://h:49374/url-base".to_string()),
            None,
            Some("/env-base".to_string()),
        );
        assert_eq!(ep.base_path, "/url-base");
    }

    /// H11 — multi-segment base path is preserved verbatim.
    #[test]
    fn multi_segment_base_path_is_preserved() {
        let ep = ServerEndpoint::from_pair_with_base(None, None, Some("/a/b".to_string()));
        assert_eq!(ep.base_path, "/a/b");
        assert_eq!(
            ep.build_url("/admin/status"),
            "http://127.0.0.1:49374/a/b/admin/status"
        );
    }

    /// H10 — a traversal-y base (`/wiki/../etc`) is neutralised by
    /// `normalize_prefix` to empty (it rejects dot-segments), so the client
    /// falls back to the root rather than emitting `/wiki/../etc/...`. The
    /// client then 404s consistently instead of walking out of the prefix.
    #[test]
    fn traversal_base_path_is_neutralised_to_root() {
        let ep = ServerEndpoint::from_pair_with_base(None, None, Some("/wiki/../etc".to_string()));
        assert_eq!(ep.base_path, "");
        let joined = ep.build_url("/admin/status");
        assert!(
            !joined.contains("/../") && !joined.contains("/etc/admin"),
            "traversal must not leak into the request URL; got {joined}"
        );
        assert_eq!(joined, "http://127.0.0.1:49374/admin/status");
    }

    /// A bare base value without a leading slash is normalised to `/<core>`,
    /// matching the server's own `normalize_prefix` so the two agree.
    #[test]
    fn bare_env_base_gets_leading_slash() {
        let ep = ServerEndpoint::from_pair_with_base(None, None, Some("wiki".to_string()));
        assert_eq!(ep.base_path, "/wiki");
    }

    /// `/` and empty both mean "root" — no prefix added.
    #[test]
    fn root_like_base_values_mean_no_prefix() {
        for raw in ["", "/", "//", "  /  "] {
            let ep = ServerEndpoint::from_pair_with_base(None, None, Some(raw.to_string()));
            assert_eq!(ep.base_path, "", "{raw:?} should normalise to no prefix");
            assert_eq!(
                ep.build_url("/admin/status"),
                "http://127.0.0.1:49374/admin/status"
            );
        }
    }

    #[test]
    fn build_url_with_query_url_encodes_values() {
        let ep = ServerEndpoint::from_pair(Some("http://h:49374/wiki".to_string()), None);
        let url = super::build_url_with_query(
            &ep,
            "/admin/pending-writes/id/approve",
            &[("workspace", "default workspace"), ("project", "a/b & c")],
        )
        .unwrap();
        assert_eq!(
            url,
            "http://h:49374/wiki/admin/pending-writes/id/approve?workspace=default+workspace&project=a%2Fb+%26+c"
        );
    }

    // ----------------------------------------------------------------
    // split_origin_and_path
    // ----------------------------------------------------------------

    #[test]
    fn split_origin_and_path_cases() {
        assert_eq!(
            split_origin_and_path("http://h:49374"),
            ("http://h:49374".to_string(), String::new())
        );
        assert_eq!(
            split_origin_and_path("http://h:49374/"),
            ("http://h:49374".to_string(), String::new())
        );
        assert_eq!(
            split_origin_and_path("http://h:49374/wiki"),
            ("http://h:49374".to_string(), "/wiki".to_string())
        );
        assert_eq!(
            split_origin_and_path("http://h:49374/wiki/"),
            ("http://h:49374".to_string(), "/wiki".to_string())
        );
        assert_eq!(
            split_origin_and_path("https://h:49374/a/b"),
            ("https://h:49374".to_string(), "/a/b".to_string())
        );
        // No scheme separator → opaque origin, no path split.
        assert_eq!(
            split_origin_and_path("127.0.0.1:49374"),
            ("127.0.0.1:49374".to_string(), String::new())
        );
    }

    // ----------------------------------------------------------------
    // ServerEndpoint::authenticate
    // ----------------------------------------------------------------

    #[test]
    fn authenticate_no_token_leaves_request_unchanged() {
        let ep = ServerEndpoint::from_pair(None, None);
        let client = reqwest::Client::new();
        // Build a request, authenticate it, then build to inspect.
        let req = ep
            .authenticate(client.get("http://localhost"))
            .build()
            .unwrap();
        // No Authorization header should be present.
        assert!(
            req.headers().get("authorization").is_none(),
            "no Authorization header expected"
        );
    }

    #[test]
    fn authenticate_with_token_sets_bearer_header() {
        let ep = ServerEndpoint::from_pair(None, Some("tok123".to_string()));
        let client = reqwest::Client::new();
        let req = ep
            .authenticate(client.get("http://localhost"))
            .build()
            .unwrap();
        let auth = req
            .headers()
            .get("authorization")
            .expect("Authorization header must be set")
            .to_str()
            .unwrap();
        assert_eq!(auth, "Bearer tok123");
    }

    #[tokio::test]
    async fn from_config_resolving_auth_uses_stored_oidc_for_authorization_header() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config {
            data_dir: tmp.path().to_path_buf(),
            ..Config::default()
        };
        config.auth.bearer_token = None;
        ai_memory_llm::OidcToken {
            access: secrecy::SecretString::from("oidc-access".to_string()),
            refresh: secrecy::SecretString::from("refresh-token".to_string()),
            expires_at_ms: u64::MAX,
            extra: ai_memory_llm::OidcExtras {
                issuer: "https://issuer.example.com/realms/team".to_string(),
                client_id: "ai-memory-cli".to_string(),
                token_endpoint: "https://issuer.example.com/token".to_string(),
            },
        }
        .save(&config.oidc_device_token_path())
        .expect("save test OIDC token");

        let ep = ServerEndpoint::from_config_resolving_auth(&config).await;
        let client = reqwest::Client::new();
        let req = ep
            .authenticate(client.get("http://localhost"))
            .build()
            .unwrap();
        let auth = req
            .headers()
            .get("authorization")
            .expect("Authorization header must be set")
            .to_str()
            .unwrap();

        assert_eq!(auth, "Bearer oidc-access");
    }
}
