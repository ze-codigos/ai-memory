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
use crate::http_client::{ServerEndpoint, post_json, post_json_no_content};
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

/// How the prepare selects a workstream, in the order they are tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Attempt {
    /// This session was adopted before, into this workstream: select it by
    /// name. Works against a server without the native-session map; fails
    /// (and falls through) when the workstream was renamed or is busy.
    Cached(String),
    /// A new workstream under the resolved name. A server with the map
    /// answers with the linked workstream instead when it has one.
    Fresh(String),
    /// The resolved name is taken by another conversation.
    Suffixed(String),
}

/// The selections to try. The suffix goes last on purpose: a same-title
/// session that is NOT this one must end up under its own name, and only
/// the server can tell those apart (by the native session id, never the
/// title).
pub(crate) fn attempts(
    cached: Option<&SessionLink>,
    resolved_name: &str,
    native_session_id: &str,
) -> Vec<Attempt> {
    let mut attempts = Vec::with_capacity(3);
    if let Some(link) = cached {
        attempts.push(Attempt::Cached(link.workstream_name.clone()));
    }
    attempts.push(Attempt::Fresh(resolved_name.to_string()));
    attempts.push(Attempt::Suffixed(with_suffix(
        resolved_name,
        native_session_id,
    )));
    attempts
}

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

pub(crate) async fn adopt(input: AdoptInput<'_>) -> Result<AdoptedRun> {
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
        workstream: match attempt {
            Attempt::Cached(name) => Some(name.clone()),
            Attempt::Fresh(_) | Attempt::Suffixed(_) => None,
        },
        new_workstream: match attempt {
            Attempt::Cached(_) => None,
            Attempt::Fresh(name) | Attempt::Suffixed(name) => Some(name.clone()),
        },
        lease_owner: crate::commands::run::lease_owner(),
        native_session_id: Some(input.native_session_id.to_string()),
    };

    let cached = adopted_state::load_link(
        input.data_dir,
        input.native_session_id,
        &repository.cwd,
        input.server_url,
    );
    let mut prepared: Option<(PrepareManagedRunResponse, Attempt)> = None;
    let mut last_error = None;
    for attempt in attempts(cached.as_ref(), &resolved.name, input.native_session_id) {
        match post_json::<_, PrepareManagedRunResponse>(
            &endpoint,
            "/workstream/runs",
            &request(&attempt),
        )
        .await
        {
            Ok(response) => {
                prepared = Some((response, attempt));
                break;
            }
            // A taken or renamed name is the failure the next attempt
            // exists for; any other error would only repeat, but telling
            // them apart costs a round trip the next attempt makes anyway.
            Err(error) => last_error = Some(error),
        }
    }
    let Some((prepared, attempt)) = prepared else {
        return Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("no selection to try"))
            .context("opening a workstream for the adopted session"));
    };
    let reattached = prepared.session_reattached || matches!(attempt, Attempt::Cached(_));
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
    // The link outlives the run record on purpose: see `SessionLink`. Its
    // failure is not the adoption's — the server keeps the same map.
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
    Ok(state)
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
    fn attempts_try_the_cached_workstream_then_fresh_then_suffixed() {
        let link = SessionLink {
            native_session_id: "0c892539-b62d-4475".into(),
            workstream_id: ai_memory_core::WorkstreamId::new(),
            workstream_name: "2026-09-17-teste".into(),
            cwd: std::path::PathBuf::from("/repo"),
            server_url: "https://memory-test.example".into(),
            linked_at: 0,
        };
        assert_eq!(
            attempts(Some(&link), "2026-09-17-outro", "0c892539-b62d-4475"),
            vec![
                Attempt::Cached("2026-09-17-teste".into()),
                Attempt::Fresh("2026-09-17-outro".into()),
                Attempt::Suffixed("2026-09-17-outro-0c892539".into()),
            ]
        );
    }

    #[test]
    fn attempts_without_a_link_start_fresh() {
        assert_eq!(
            attempts(None, "ajuste", "0c892539-b62d-4475"),
            vec![
                Attempt::Fresh("ajuste".into()),
                Attempt::Suffixed("ajuste-0c892539".into()),
            ]
        );
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
