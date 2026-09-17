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

use crate::commands::adopted_state::{self, AdoptedRun};
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

    let request = |name: String| PrepareManagedRunRequest {
        workspace: workspace.clone(),
        project: project.clone(),
        cwd: repository.cwd.to_string_lossy().into_owned(),
        repo_fingerprint: repository.repo_fingerprint.clone(),
        worktree_fingerprint: repository.worktree_fingerprint.clone(),
        agent: AgentKind::ClaudeCode,
        automatic_harness: false,
        available_agents: Vec::new(),
        workstream: None,
        // Always its own workstream. Reusing an existing one would interleave
        // two conversations in one ledger: with no heartbeat, the first
        // session's run lapses after 90s and the second would find the name
        // free.
        new_workstream: Some(name),
        lease_owner: crate::commands::run::lease_owner(),
    };

    let prepared: PrepareManagedRunResponse = match post_json(
        &endpoint,
        "/workstream/runs",
        &request(resolved.name.clone()),
    )
    .await
    {
        Ok(prepared) => prepared,
        // A duplicate name is the one failure worth a second attempt; any
        // other error would only repeat. One retry, then give up.
        Err(_) => post_json(
            &endpoint,
            "/workstream/runs",
            &request(with_suffix(&resolved.name, input.native_session_id)),
        )
        .await
        .context("opening a workstream for the adopted session")?,
    };
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

    let state = AdoptedRun {
        run_id: prepared.run_id,
        run_path,
        native_session_id: input.native_session_id.to_string(),
        workstream_name: prepared.workstream_name,
        provisional: resolved.provisional,
        server_url: input.server_url.to_string(),
        cwd: repository.cwd,
        adopted_at: jiff::Timestamp::now().as_second(),
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
