//! Rename of a launcher placeholder workstream from inside the first-prompt hook.
//!
//! `passabot-memory run` opens every interactive session in a fresh workstream
//! named `novo-<n>`: nobody has said what the session is about yet. The desktop
//! app and the prompt slug only exist once the user has spoken, so the earliest
//! moment a real name exists is the first `UserPromptSubmit`. This module does
//! that rename in-process — the model is never asked to run a command for it.
//!
//! Never blocks the prompt. A failure is retried on every prompt, but the
//! developer hears about it once: the model gets the reason and the manual
//! command a single time per session.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ai_memory_core::{RenameManagedWorkstreamRequest, RenamedManagedWorkstream};
use ai_memory_workstream::inspect_repository;
use anyhow::{Context as _, Result};
use sha2::{Digest as _, Sha256};

use crate::commands::session_name;
use crate::config::DEFAULT_WORKSPACE;
use crate::http_client::{ServerEndpoint, ServerResponseError, post_json};
use crate::marker::{find_marker, parse_toml_key, repo_root_project};

/// What `run_ws_auto` in the shell launcher emits before anything better is
/// known: the prefix followed by a checksum, digits only.
const PLACEHOLDER_PREFIX: &str = "novo-";
/// How long the rename may hold up the first prompt. The launcher can block
/// forever waiting on the server; a hook cannot: it sits between the user
/// pressing enter and the model answering, and one POST against a healthy
/// server is milliseconds.
const RENAME_TIMEOUT: Duration = Duration::from_secs(5);

/// Whether a workstream name is the launcher's placeholder. Digits only after
/// the prefix: `novo-recurso` is a name somebody typed for `run --new`, and
/// renaming it would replace a choice, not a placeholder.
fn is_placeholder(name: &str) -> bool {
    name.strip_prefix(PLACEHOLDER_PREFIX)
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Whether this prompt should rename the workstream: a launcher-managed
/// session whose exported name is still the placeholder and which this
/// session has not renamed yet. The environment carries the placeholder for
/// the whole session, so the marker is the only way to tell the first prompt
/// from the rest.
pub(crate) fn wants_placeholder_rename(
    workstream_env: Option<&str>,
    data_dir: &Path,
    native_session_id: &str,
) -> bool {
    workstream_env
        .is_some_and(|name| is_placeholder(name) && !already_renamed(data_dir, native_session_id))
}

/// Disambiguate a name the server already has. The desktop app titles sessions
/// from their content, so two sessions on the same task collide easily.
pub(crate) fn with_suffix(name: &str, native_session_id: &str) -> String {
    let short: String = native_session_id.chars().take(8).collect();
    session_name::with_suffix_within_limit(name, &format!("-{short}"))
}

/// Concrete workspace/project for the rename body, mirroring the fallbacks the
/// server applies to hook events: marker first, then the repo-root strategy,
/// then the directory name. The hook path normally lets the server decide, but
/// the rename needs both named outright.
pub(crate) fn rename_scope(cwd: &Path) -> (String, String) {
    let cwd_str = cwd.to_string_lossy();
    let marker = find_marker(&cwd_str);
    let workspace = marker
        .as_ref()
        .and_then(|path| parse_toml_key(path, "workspace"))
        .unwrap_or_else(|| DEFAULT_WORKSPACE.to_string());
    let declared = marker
        .as_ref()
        .and_then(|path| parse_toml_key(path, "project"));
    let strategy = marker
        .as_ref()
        .and_then(|path| parse_toml_key(path, "project_strategy"));
    let project = declared
        .or_else(|| {
            matches!(strategy.as_deref(), Some("repo-root" | "repo_root"))
                .then(|| repo_root_project(&cwd_str))
                .flatten()
        })
        .or_else(|| {
            cwd.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "default".to_string());
    (workspace, project)
}

fn state_marker(data_dir: &Path, kind: &str, native_session_id: &str) -> PathBuf {
    data_dir.join("hook-state").join(format!(
        "{kind}-{:x}",
        Sha256::digest(native_session_id.as_bytes())
    ))
}

fn write_marker(marker: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(parent) = marker.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(marker, contents)
}

fn already_renamed(data_dir: &Path, native_session_id: &str) -> bool {
    state_marker(data_dir, "renamed", native_session_id).exists()
}

/// The marker holds the new name: useful when reading the state directory by
/// hand, and it costs nothing.
pub(crate) fn mark_renamed(
    data_dir: &Path,
    native_session_id: &str,
    new_name: &str,
) -> std::io::Result<()> {
    write_marker(
        &state_marker(data_dir, "renamed", native_session_id),
        new_name,
    )
}

/// Bookkeeping after the server said yes. Its failure is not the rename's:
/// the name is set, and the marker only suppresses a retry that would now
/// find the placeholder gone and stop by itself.
fn record_renamed(data_dir: &Path, native_session_id: &str, new_name: &str) {
    if let Err(error) = mark_renamed(data_dir, native_session_id, new_name) {
        eprintln!(
            "ai-memory hook warning: workstream renamed to '{new_name}' but the marker could not be written: {error}"
        );
    }
    let _ = std::fs::remove_file(state_marker(data_dir, "rename-pending", native_session_id));
}

/// Whether this session still owes the developer a rename warning. Marks it
/// as spent in the same call — the rename itself keeps being retried, only the
/// paragraph in the turn is rate-limited.
fn claim_rename_warning(data_dir: &Path, native_session_id: &str) -> bool {
    let marker = state_marker(data_dir, "rename-warned", native_session_id);
    if marker.exists() {
        return false;
    }
    write_marker(&marker, "").is_ok()
}

/// The slug the first prompt produced, kept for retries: a rename that fails
/// on prompt one and succeeds on prompt three must not be named after "sim".
fn pending_name(data_dir: &Path, native_session_id: &str) -> Option<String> {
    let raw = std::fs::read_to_string(state_marker(data_dir, "rename-pending", native_session_id))
        .ok()?;
    let name = raw.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// The fallback when the hook could not rename by itself. Selects by id
/// rather than by the placeholder name: by the time the model runs this, the
/// placeholder may already have been renamed by a later retry.
pub(crate) fn rename_nudge(current_name: &str, reason: &str) -> String {
    format!(
        "[nomear workstream] Este workstream ainda tem nome provisório \
         ('{current_name}') e a memória não conseguiu renomeá-lo sozinha \
         ({reason}). Avise o usuário na primeira resposta. Depois escolha um \
         nome curto do trabalho atual e rode, silenciosamente, como parte do \
         seu trabalho:\n\n    \
         ai-memory rename-workstream --workstream-id \"$AI_MEMORY_WORKSTREAM_ID\" --to '<nome>'\n\n\
         Convenção: YYYY-MM-DD-<scope>:<slug> (kebab-case, ~12-18 chars): toca 1 serviço -> \
         <servico>:<slug>; vários serviços de 1 produto -> <produto>:<slug>; \
         vários produtos -> <slug>."
    )
}

pub(crate) struct RenameInput<'a> {
    pub data_dir: &'a Path,
    pub server_url: &'a str,
    pub bearer: Option<&'a str>,
    pub cwd: &'a Path,
    /// Platform config root for the desktop title lookup; `None` skips it
    /// rather than searching the checkout.
    pub config_dir: Option<&'a Path>,
    pub native_session_id: &'a str,
    pub host_session_id: Option<&'a str>,
    pub first_prompt: &'a str,
    /// The name the launcher exported, i.e. what the rename selects by.
    pub placeholder: &'a str,
    pub today: jiff::civil::Date,
}

/// What the first prompt of a launcher-managed session injects, if anything.
///
/// A successful rename is silent: the name is the hook's job, not the
/// model's. Failure — network, a server that is down, a timeout — is logged,
/// retried on the next prompt (the marker is only written on success), and
/// reported to the model once with the reason and the manual command.
pub(crate) async fn managed_prompt_context(input: RenameInput<'_>) -> Option<String> {
    let data_dir = input.data_dir;
    let native = input.native_session_id;
    let placeholder = input.placeholder;
    // The hook gates on this too; kept here so the function is safe to call
    // on every prompt regardless of who asks.
    if already_renamed(data_dir, native) {
        return None;
    }
    let outcome = tokio::time::timeout(RENAME_TIMEOUT, rename_placeholder(input)).await;
    let reason = match outcome {
        Ok(Ok(_)) => return None,
        Ok(Err(error)) => format!("{error:#}"),
        Err(_) => "o servidor da memória não respondeu a tempo".to_string(),
    };
    eprintln!("ai-memory hook warning: placeholder rename failed: {reason}");
    claim_rename_warning(data_dir, native).then(|| rename_nudge(placeholder, &reason))
}

fn status_of(error: &anyhow::Error) -> Option<reqwest::StatusCode> {
    error
        .downcast_ref::<ServerResponseError>()
        .map(ServerResponseError::status)
}

/// Retitle the placeholder workstream after the session's first prompt.
/// Returns the stored name; the session is marked renamed only on success —
/// which includes finding the placeholder already gone, because someone
/// (the model after a nudge, the developer by hand) renamed it first.
pub(crate) async fn rename_placeholder(input: RenameInput<'_>) -> Result<String> {
    let repository = inspect_repository(input.cwd)?;
    let resolved = session_name::resolve_name(
        input.host_session_id,
        input.config_dir,
        input.first_prompt,
        input.today,
    );
    let name = if resolved.provisional {
        match pending_name(input.data_dir, input.native_session_id) {
            Some(first) => first,
            None => {
                // Best effort: without it a retry is named after a later
                // prompt, which is worse than a missing file.
                let _ = write_marker(
                    &state_marker(input.data_dir, "rename-pending", input.native_session_id),
                    &resolved.name,
                );
                resolved.name
            }
        }
    } else {
        resolved.name
    };
    let (workspace, project) = rename_scope(&repository.cwd);
    let endpoint = ServerEndpoint::from_pair(
        Some(input.server_url.to_string()),
        input.bearer.map(str::to_string),
    );
    let request = |to: String| RenameManagedWorkstreamRequest {
        workspace: workspace.clone(),
        project: project.clone(),
        repo_fingerprint: repository.repo_fingerprint.clone(),
        worktree_fingerprint: repository.worktree_fingerprint.clone(),
        // By the placeholder name, on purpose: the decision to rename comes
        // from the name the launcher exported, so the selector must be that
        // same name. A rename that happened in the meantime then answers 404
        // and nothing is overwritten; selecting by id would clobber it.
        from: Some(input.placeholder.to_string()),
        workstream_id: None,
        to,
    };
    let renamed: RenamedManagedWorkstream =
        match post_json(&endpoint, "/workstream/rename", &request(name.clone())).await {
            Ok(renamed) => renamed,
            Err(error) if status_of(&error) == Some(reqwest::StatusCode::NOT_FOUND) => {
                record_renamed(input.data_dir, input.native_session_id, input.placeholder);
                return Ok(input.placeholder.to_string());
            }
            // The only 409 the rename route produces is a taken name
            // (`StoreError::WorkstreamNameTaken`): two sessions on the same
            // task, same day. One retry with a session suffix, then give up.
            Err(error) if status_of(&error) == Some(reqwest::StatusCode::CONFLICT) => post_json(
                &endpoint,
                "/workstream/rename",
                &request(with_suffix(&name, input.native_session_id)),
            )
            .await
            .context("renaming the placeholder workstream (suffixed retry)")?,
            Err(error) => return Err(error.context("renaming the placeholder workstream")),
        };
    record_renamed(input.data_dir, input.native_session_id, &renamed.to);
    Ok(renamed.to)
}

#[cfg(test)]
mod tests {
    use ai_memory_core::WorkstreamId;

    use super::*;

    const TODAY: jiff::civil::Date = jiff::civil::date(2026, 9, 17);

    #[test]
    fn the_launcher_placeholder_is_recognised() {
        assert!(is_placeholder("novo-491181"));
        assert!(!is_placeholder("nexus:bus-cancel"));
        assert!(!is_placeholder("2026-09-17-novo-recurso"));
        // A name a developer typed for `run --new` — Portuguese is the
        // working language, so `novo-…` words are plausible.
        assert!(!is_placeholder("novo-recurso"));
        assert!(!is_placeholder("novo-"));
    }

    #[test]
    fn a_session_is_renamed_once() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!already_renamed(tmp.path(), "nat-1"));
        mark_renamed(tmp.path(), "nat-1", "2026-09-17-ajuste").unwrap();
        assert!(already_renamed(tmp.path(), "nat-1"));
        assert!(
            !already_renamed(tmp.path(), "nat-2"),
            "the marker is per session, not per machine"
        );
    }

    #[test]
    fn only_an_unrenamed_launcher_placeholder_wants_a_rename() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(wants_placeholder_rename(
            Some("novo-1"),
            tmp.path(),
            "nat-1"
        ));
        assert!(!wants_placeholder_rename(
            Some("nexus:x"),
            tmp.path(),
            "nat-1"
        ));
        assert!(!wants_placeholder_rename(None, tmp.path(), "nat-1"));
        mark_renamed(tmp.path(), "nat-1", "2026-09-17-x").unwrap();
        assert!(!wants_placeholder_rename(
            Some("novo-1"),
            tmp.path(),
            "nat-1"
        ));
    }

    /// Recording HTTP stub: canned replies served in order (the last one
    /// repeats), request heads streamed back. Reads until the announced body
    /// has arrived so a split write cannot make a test flaky.
    async fn serve_sequence(
        replies: Vec<(&'static str, String)>,
    ) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut replies = replies.into_iter().peekable();
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut raw = Vec::new();
                let mut buf = [0_u8; 8192];
                loop {
                    let read = stream.read(&mut buf).await.unwrap_or(0);
                    if read == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..read]);
                    let text = String::from_utf8_lossy(&raw);
                    if let Some((head, body)) = text.split_once("\r\n\r\n") {
                        let wanted = head
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|v| v.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if body.len() >= wanted {
                            break;
                        }
                    }
                }
                let _ = tx.send(String::from_utf8_lossy(&raw).into_owned());
                let (status, body) = if replies.len() > 1 {
                    replies.next().unwrap()
                } else {
                    replies.peek().cloned().unwrap()
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{addr}"), rx)
    }

    async fn serve_requests(
        status: &'static str,
        body: String,
    ) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        serve_sequence(vec![(status, body)]).await
    }

    fn request_body(request: &str) -> serde_json::Value {
        serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
    }

    async fn no_more_requests(requests: &mut tokio::sync::mpsc::UnboundedReceiver<String>) {
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), requests.recv())
                .await
                .is_err(),
            "no further request expected"
        );
    }

    fn renamed_reply(from: &str, to: &str) -> String {
        serde_json::json!({"workstream_id": WorkstreamId::new(), "from": from, "to": to})
            .to_string()
    }

    /// What the server says on the rename route for a taken name
    /// (`StoreError::WorkstreamNameTaken`), verbatim.
    fn taken_reply(name: &str) -> String {
        format!(r#"{{"error":"workstream name '{name}' is already taken in this checkout"}}"#)
    }

    fn input<'a>(
        tmp: &'a Path,
        base: &'a str,
        config_dir: Option<&'a Path>,
        prompt: &'a str,
    ) -> RenameInput<'a> {
        RenameInput {
            data_dir: tmp,
            server_url: base,
            bearer: None,
            cwd: tmp,
            config_dir,
            native_session_id: "nat-1",
            host_session_id: None,
            first_prompt: prompt,
            placeholder: "novo-1",
            today: TODAY,
        }
    }

    const PROMPT: &str = "Corrigir o timeout do nexus";
    const DATED_SLUG: &str = "2026-09-17-corrigir-o-timeout";

    #[tokio::test]
    async fn renames_by_placeholder_name_to_the_dated_slug_and_marks_the_session() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) =
            serve_requests("200 OK", renamed_reply("novo-1", DATED_SLUG)).await;

        let renamed = rename_placeholder(input(tmp.path(), &base, None, PROMPT))
            .await
            .unwrap();

        assert_eq!(renamed, DATED_SLUG);
        let request = requests.recv().await.unwrap();
        assert!(request.starts_with("POST /workstream/rename"), "{request}");
        let body = request_body(&request);
        assert_eq!(body["from"], "novo-1");
        assert_eq!(body["to"], DATED_SLUG);
        assert!(
            body.get("workstream_id").is_none(),
            "by name, so a rename done by hand in the meantime is never clobbered: {body}"
        );
        assert!(already_renamed(tmp.path(), "nat-1"));
    }

    #[tokio::test]
    async fn the_desktop_title_wins_over_the_prompt_slug() {
        let tmp = tempfile::tempdir().unwrap();
        let config = tmp.path().join("config");
        let sessions = config.join("Claude/claude-code-sessions/conta/ws");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(
            sessions.join("local_abc.json"),
            r#"{"title":"Ajuste no checkout"}"#,
        )
        .unwrap();
        let (base, mut requests) = serve_requests(
            "200 OK",
            renamed_reply("novo-1", "2026-09-17-Ajuste no checkout"),
        )
        .await;
        let mut input = input(tmp.path(), &base, Some(&config), PROMPT);
        input.host_session_id = Some("local_abc");

        rename_placeholder(input).await.unwrap();

        let body = request_body(&requests.recv().await.unwrap());
        assert_eq!(body["to"], "2026-09-17-Ajuste no checkout");
    }

    #[tokio::test]
    async fn a_failed_rename_leaves_the_session_unmarked() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, _requests) =
            serve_requests("500 Internal Server Error", r#"{"error":"boom"}"#.into()).await;

        let result = rename_placeholder(input(tmp.path(), &base, None, PROMPT)).await;

        assert!(result.is_err());
        assert!(
            !already_renamed(tmp.path(), "nat-1"),
            "the next prompt must try again"
        );
    }

    #[tokio::test]
    async fn a_taken_name_is_retried_once_with_a_session_suffix() {
        let tmp = tempfile::tempdir().unwrap();
        let suffixed = format!("{DATED_SLUG}-nat-1");
        let (base, mut requests) = serve_sequence(vec![
            ("409 Conflict", taken_reply(DATED_SLUG)),
            ("200 OK", renamed_reply("novo-1", &suffixed)),
        ])
        .await;

        let renamed = rename_placeholder(input(tmp.path(), &base, None, PROMPT))
            .await
            .unwrap();

        assert_eq!(renamed, suffixed);
        let first = request_body(&requests.recv().await.unwrap());
        let second = request_body(&requests.recv().await.unwrap());
        assert_eq!(first["to"], DATED_SLUG);
        assert_eq!(second["to"], suffixed);
        assert!(already_renamed(tmp.path(), "nat-1"));
    }

    #[tokio::test]
    async fn the_suffixed_retry_stays_within_the_server_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let long_prompt = "x".repeat(200);
        let (base, mut requests) = serve_sequence(vec![
            ("409 Conflict", taken_reply("whatever")),
            ("200 OK", renamed_reply("novo-1", "any")),
        ])
        .await;

        rename_placeholder(input(tmp.path(), &base, None, &long_prompt))
            .await
            .unwrap();

        let _first = requests.recv().await.unwrap();
        let second = request_body(&requests.recv().await.unwrap());
        let to = second["to"].as_str().unwrap();
        assert!(to.chars().count() <= 128, "{}", to.len());
        assert!(to.ends_with("-nat-1"), "{to}");
    }

    #[tokio::test]
    async fn a_bad_request_is_not_retried() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) = serve_requests(
            "400 Bad Request",
            r#"{"error":"invalid workstream name"}"#.into(),
        )
        .await;

        let result = rename_placeholder(input(tmp.path(), &base, None, PROMPT)).await;

        assert!(result.is_err());
        let _first = requests.recv().await.unwrap();
        no_more_requests(&mut requests).await;
    }

    #[tokio::test]
    async fn a_placeholder_already_renamed_elsewhere_is_left_alone_and_marked() {
        // The model (told by an earlier nudge) or the developer renamed it by
        // hand: the server no longer knows the placeholder.
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) = serve_requests(
            "404 Not Found",
            r#"{"error":"workstream 'novo-1' not found"}"#.into(),
        )
        .await;

        let renamed = rename_placeholder(input(tmp.path(), &base, None, PROMPT))
            .await
            .unwrap();

        assert_eq!(
            renamed, "novo-1",
            "reports the name it knows; nothing was written"
        );
        let _first = requests.recv().await.unwrap();
        no_more_requests(&mut requests).await;
        assert!(already_renamed(tmp.path(), "nat-1"));
    }

    #[tokio::test]
    async fn a_retry_keeps_the_first_prompts_slug() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) = serve_sequence(vec![
            ("500 Internal Server Error", r#"{"error":"boom"}"#.into()),
            ("200 OK", renamed_reply("novo-1", DATED_SLUG)),
        ])
        .await;

        assert!(
            rename_placeholder(input(tmp.path(), &base, None, PROMPT))
                .await
                .is_err()
        );
        rename_placeholder(input(tmp.path(), &base, None, "sim"))
            .await
            .unwrap();

        let _first = requests.recv().await.unwrap();
        let second = request_body(&requests.recv().await.unwrap());
        assert_eq!(second["to"], DATED_SLUG, "not `2026-09-17-sim`");
    }

    #[tokio::test]
    async fn first_prompt_renames_a_placeholder_and_stays_silent() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) =
            serve_requests("200 OK", renamed_reply("novo-1", DATED_SLUG)).await;

        let context = managed_prompt_context(input(tmp.path(), &base, None, PROMPT)).await;

        assert_eq!(context, None, "the model is not asked to do anything");
        assert!(
            requests
                .recv()
                .await
                .unwrap()
                .starts_with("POST /workstream/rename")
        );
        assert!(already_renamed(tmp.path(), "nat-1"));
    }

    #[tokio::test]
    async fn later_prompts_do_not_rename_again() {
        let tmp = tempfile::tempdir().unwrap();
        mark_renamed(tmp.path(), "nat-1", DATED_SLUG).unwrap();
        let (base, mut requests) = serve_requests("200 OK", "{}".into()).await;

        let context = managed_prompt_context(input(tmp.path(), &base, None, PROMPT)).await;

        assert_eq!(context, None);
        no_more_requests(&mut requests).await;
    }

    #[tokio::test]
    async fn a_failed_rename_nudges_once_with_the_reason_and_keeps_retrying() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) =
            serve_requests("500 Internal Server Error", r#"{"error":"boom"}"#.into()).await;

        let first = managed_prompt_context(input(tmp.path(), &base, None, PROMPT))
            .await
            .expect("the model gets the manual instructions");
        let second = managed_prompt_context(input(tmp.path(), &base, None, "sim")).await;

        assert!(first.contains("rename-workstream"));
        assert!(
            first.contains("--workstream-id"),
            "the placeholder name may be stale: {first}"
        );
        assert!(
            first.contains("boom"),
            "the reason reaches the developer: {first}"
        );
        assert!(first.contains("Avise o usuário"), "{first}");
        assert_eq!(second, None, "a warning on every prompt cries wolf");
        let _ = requests.recv().await.unwrap();
        assert!(
            requests.recv().await.is_some(),
            "the rename itself is still retried on the next prompt"
        );
        assert!(!already_renamed(tmp.path(), "nat-1"));
    }
}
