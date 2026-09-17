//! Adoption of a session that started outside the `ai-memory run` launcher.
//!
//! Called from inside the native hook, never as a CLI subcommand. That is what
//! makes it work on native Windows, where the shell plugin does not run, and
//! it keeps the user's prompt out of any process argument list.

use std::path::Path;

use ai_memory_core::{
    AgentKind, LinkManagedRunRequest, PrepareManagedRunRequest, PrepareManagedRunResponse,
};
use ai_memory_workstream::inspect_repository;
use anyhow::{Context as _, Result};

use crate::commands::adopted_state::{self, AdoptedRun, SessionLink};
use crate::commands::session_name;
use crate::config::DEFAULT_WORKSPACE;
use crate::http_client::{ServerEndpoint, ServerResponseError, post_json, post_json_no_content};
use crate::marker::{find_marker, parse_toml_key, repo_root_project};

pub(crate) struct AdoptInput<'a> {
    pub data_dir: &'a Path,
    pub server_url: &'a str,
    pub bearer: Option<&'a str>,
    pub cwd: &'a Path,
    pub native_session_id: &'a str,
    pub host_session_id: Option<&'a str>,
    pub first_prompt: &'a str,
}

/// Whether this event should open a run.
///
/// The hook fires on EVERY prompt, so without this the second prompt would
/// open a second run and split the ledger in two. A session the launcher
/// started already has `AI_MEMORY_RUN_ID` and is never adopted.
pub(crate) fn should_adopt(
    data_dir: &Path,
    native_session_id: &str,
    managed_run_env: Option<&str>,
) -> bool {
    if managed_run_env.is_some_and(|value| !value.trim().is_empty()) {
        return false;
    }
    adopted_state::load(data_dir, native_session_id).is_none()
}

/// Disambiguate a name the server already has. The desktop app titles sessions
/// from their content, so two sessions on the same task collide easily.
pub(crate) fn with_suffix(name: &str, native_session_id: &str) -> String {
    let short: String = native_session_id.chars().take(8).collect();
    session_name::with_suffix_within_limit(name, &format!("-{short}"))
}

/// How the prepare selects a workstream, in the order they are tried. Both
/// are `new_workstream` requests carrying the native session id: identity
/// lives on the server, in its session map. The local link is deliberately
/// NOT a selection — selecting by a cached name would land in whatever
/// workstream holds that name today (a rename frees it for the next
/// same-title conversation), and a server without the map cannot tell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Attempt {
    /// A new workstream under the resolved name. A server with the map
    /// answers with the session's own workstream instead when it has one.
    Fresh(String),
    /// The resolved name is taken by another conversation.
    Suffixed(String),
}

/// The selections to try. The suffix goes last on purpose: a same-title
/// session that is NOT this one must end up under its own name, and only
/// the server can tell those apart (by the native session id, never the
/// title).
pub(crate) fn attempts(resolved_name: &str, native_session_id: &str) -> Vec<Attempt> {
    vec![
        Attempt::Fresh(resolved_name.to_string()),
        Attempt::Suffixed(with_suffix(resolved_name, native_session_id)),
    ]
}

/// Whether the next attempt could answer this failure. Only a taken name
/// (409) can: anything else — 401, 5xx, a connection that never opened —
/// would repeat, and its message is the one worth showing, not the
/// suffixed attempt's "already exists; select it with --workstream".
pub(crate) fn next_attempt_may_help(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<ServerResponseError>()
        .is_some_and(|response| response.status() == reqwest::StatusCode::CONFLICT)
}

/// Fragmentation detected on the client: this session was adopted before,
/// into another workstream of this checkout, and the server did not hand
/// that one back. Either it could not (no session map: a server older than
/// this client) or would not (another live session holds it). The hook
/// puts this in the turn once, at adoption, so the user learns why their
/// conversation now spans two workstreams instead of finding out later.
pub(crate) fn fragmentation_notice(previous: &str, current: &str, busy: bool) -> String {
    let why = if busy {
        "aquele workstream está ocupado por outra sessão viva"
    } else {
        "o servidor da memória não reconhece a sessão nativa (versão anterior ao mapa de sessões)"
    };
    format!(
        "⚠️ Esta conversa já tinha o workstream '{previous}' e foi adotada agora em          '{current}': {why}. O ledger desta sessão fica dividido entre os dois.          Avise o usuário na primeira resposta; nada a fazer no código."
    )
}

/// What adoption hands back to the hook: the state is on disk already (that
/// is how the next prompt knows not to adopt again), so the only thing the
/// caller needs is a paragraph for the model, when something the user
/// should know happened without failing the adoption.
pub(crate) type AdoptionNotice = Option<String>;

/// Concrete workspace/project for the prepare body, mirroring the fallbacks the
/// server applies to hook events: marker first, then the repo-root strategy,
/// then the directory name. The hook path normally lets the server decide, but
/// `prepare` needs both named outright.
pub(crate) fn adopt_scope(cwd: &Path) -> (String, String) {
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

pub(crate) async fn adopt(input: AdoptInput<'_>) -> Result<AdoptionNotice> {
    let repository = inspect_repository(input.cwd)?;
    let config_dir = dirs::config_dir();
    let resolved = session_name::resolve_name(
        input.host_session_id,
        config_dir.as_deref(),
        input.first_prompt,
        session_name::today(),
    );
    let (workspace, project) = adopt_scope(&repository.cwd);
    let endpoint = ServerEndpoint::from_pair(
        Some(input.server_url.to_string()),
        input.bearer.map(str::to_string),
    );

    // Never an unnamed selection: reusing "the current workstream" would
    // interleave two conversations in one ledger (with no heartbeat, the
    // first session's run lapses after 90s and the second would find it
    // free). Identity comes from the native session id — sent on every
    // attempt, so a server with the map reopens the session's own
    // workstream and links the run in one transaction.
    let request = |attempt: &Attempt| PrepareManagedRunRequest {
        workspace: workspace.clone(),
        project: project.clone(),
        cwd: repository.cwd.to_string_lossy().into_owned(),
        repo_fingerprint: repository.repo_fingerprint.clone(),
        worktree_fingerprint: repository.worktree_fingerprint.clone(),
        agent: AgentKind::ClaudeCode,
        automatic_harness: false,
        available_agents: Vec::new(),
        workstream: None,
        new_workstream: match attempt {
            Attempt::Fresh(name) | Attempt::Suffixed(name) => Some(name.clone()),
        },
        lease_owner: crate::commands::run::lease_owner(),
        native_session_id: Some(input.native_session_id.to_string()),
    };

    let mut prepared: Option<PrepareManagedRunResponse> = None;
    let mut last_error = None;
    for attempt in attempts(&resolved.name, input.native_session_id) {
        match post_json::<_, PrepareManagedRunResponse>(
            &endpoint,
            "/workstream/runs",
            &request(&attempt),
        )
        .await
        {
            Ok(response) => {
                prepared = Some(response);
                break;
            }
            Err(error) => {
                let retry = next_attempt_may_help(&error);
                last_error = Some(error);
                if !retry {
                    break;
                }
            }
        }
    }
    let Some(prepared) = prepared else {
        return Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("no selection to try"))
            .context("opening a workstream for the adopted session"));
    };
    let reattached = prepared.session_reattached;
    let run_path = format!("/workstream/runs/{}", prepared.run_id);

    // Without this the run keeps a NULL native_session_id, and the lapsed-lease
    // path in `finish_run` — which requires the claimed id to equal the linked
    // one — would refuse this session's transcript forever.
    post_json_no_content(
        &endpoint,
        &format!("{run_path}/link"),
        &LinkManagedRunRequest {
            native_session_id: input.native_session_id.to_string(),
        },
    )
    .await
    .context("linking the native session to the adopted run")?;

    let now = jiff::Timestamp::now().as_second();
    // The link is the client's memory of where this session went, kept past
    // the run's close (see `SessionLink`). Its only job is to notice when a
    // later adoption of the same session lands somewhere else — the server
    // decides where, by its own map. Losing it loses that notice, nothing
    // more, so its failure is not the adoption's.
    let notice = adopted_state::load_link(
        input.data_dir,
        input.native_session_id,
        &repository.cwd,
        input.server_url,
    )
    .filter(|link| link.workstream_id != prepared.workstream_id && !reattached)
    .map(|link| {
        fragmentation_notice(
            &link.workstream_name,
            &prepared.workstream_name,
            prepared.session_link_busy,
        )
    });
    if let Err(error) = adopted_state::save_link(
        input.data_dir,
        &SessionLink {
            native_session_id: input.native_session_id.to_string(),
            workstream_id: prepared.workstream_id,
            workstream_name: prepared.workstream_name.clone(),
            cwd: repository.cwd.clone(),
            server_url: input.server_url.to_string(),
            linked_at: now,
        },
    ) {
        eprintln!(
            "ai-memory hook warning: adopted session linked on the server but the local link could not be written: {error:#}"
        );
    }
    let state = AdoptedRun {
        run_id: prepared.run_id,
        run_path,
        native_session_id: input.native_session_id.to_string(),
        workstream_name: prepared.workstream_name,
        // A reopened workstream already has its name; only a workstream this
        // adoption created can still be waiting for a better one.
        provisional: resolved.provisional && !reattached,
        server_url: input.server_url.to_string(),
        cwd: repository.cwd,
        adopted_at: now,
        ended: false,
        home: None,
        session_dir: None,
        kept_by_launcher: false,
        agent: None,
        exit_code: None,
        checkpoint: None,
    };
    adopted_state::save(input.data_dir, &state)?;
    Ok(notice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::adopted_state::sample_run;

    #[test]
    fn does_not_adopt_a_session_that_already_has_state() {
        let tmp = tempfile::tempdir().unwrap();
        adopted_state::save(tmp.path(), &sample_run("nat-1")).unwrap();
        assert!(!should_adopt(tmp.path(), "nat-1", None));
    }

    #[test]
    fn does_not_adopt_a_launcher_managed_session() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!should_adopt(tmp.path(), "nat-1", Some("run-id-existente")));
    }

    #[test]
    fn an_empty_run_id_is_not_managed() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(should_adopt(tmp.path(), "nat-1", Some("   ")));
    }

    #[test]
    fn adopts_a_fresh_session() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(should_adopt(tmp.path(), "nat-1", None));
    }

    #[test]
    fn adoption_is_per_session() {
        let tmp = tempfile::tempdir().unwrap();
        adopted_state::save(tmp.path(), &sample_run("nat-1")).unwrap();
        assert!(should_adopt(tmp.path(), "nat-2", None));
    }

    #[test]
    fn attempts_are_fresh_then_suffixed_never_a_cached_name() {
        assert_eq!(
            attempts("ajuste", "0c892539-b62d-4475"),
            vec![
                Attempt::Fresh("ajuste".into()),
                Attempt::Suffixed("ajuste-0c892539".into()),
            ]
        );
    }

    #[test]
    fn only_a_taken_name_earns_the_suffixed_attempt() {
        let taken: anyhow::Error = ServerResponseError::for_tests(
            reqwest::StatusCode::CONFLICT,
            "workstream 'x' already exists".into(),
        )
        .into();
        assert!(next_attempt_may_help(&taken));
        let unauthorized: anyhow::Error =
            ServerResponseError::for_tests(reqwest::StatusCode::UNAUTHORIZED, String::new()).into();
        assert!(!next_attempt_may_help(&unauthorized));
        assert!(!next_attempt_may_help(&anyhow::anyhow!(
            "error sending request: connection refused"
        )));
    }

    #[test]
    fn fragmentation_notice_names_both_workstreams_and_the_cause() {
        let busy = fragmentation_notice("a", "a-0c892539", true);
        assert!(busy.contains("'a'") && busy.contains("'a-0c892539'"));
        assert!(busy.contains("ocupado"));
        let old_server = fragmentation_notice("a", "a-0c892539", false);
        assert!(old_server.contains("não reconhece"));
    }

    #[test]
    fn collision_suffix_uses_the_session_id_head() {
        assert_eq!(
            with_suffix("ajuste-checkout", "0c892539-b62d-4475"),
            "ajuste-checkout-0c892539"
        );
    }

    #[test]
    fn scope_falls_back_to_the_directory_name() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("meu-repo");
        std::fs::create_dir_all(&repo).unwrap();
        let (workspace, project) = adopt_scope(&repo);
        assert_eq!(workspace, DEFAULT_WORKSPACE);
        assert_eq!(project, "meu-repo");
    }

    #[test]
    fn scope_honours_a_marker_declaration() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("meu-repo");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(
            repo.join(".ai-memory.toml"),
            "workspace = \"passabot\"\nproject = \"nexus\"\n",
        )
        .unwrap();
        assert_eq!(
            adopt_scope(&repo),
            ("passabot".to_string(), "nexus".to_string())
        );
    }
}
