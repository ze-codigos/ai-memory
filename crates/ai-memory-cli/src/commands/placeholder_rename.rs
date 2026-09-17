//! Rename of a launcher placeholder workstream from inside the first-prompt hook.
//!
//! `passabot-memory run` opens every interactive session in a fresh workstream
//! named `novo-<n>`: nobody has said what the session is about yet. The desktop
//! app and the prompt slug only exist once the user has spoken, so the earliest
//! moment a real name exists is the first `UserPromptSubmit`. This module does
//! that rename in-process — the model is never asked to run a command for it.
//!
//! Never blocks the prompt: a failure is reported once and the model gets the
//! old nudge as a fallback.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ai_memory_core::{RenameManagedWorkstreamRequest, RenamedManagedWorkstream, WorkstreamId};
use ai_memory_workstream::inspect_repository;
use anyhow::{Context as _, Result};
use sha2::{Digest as _, Sha256};

use crate::commands::adopt_session::{adopt_scope, with_suffix};
use crate::commands::session_name;
use crate::http_client::{ServerEndpoint, ServerResponseError, post_json};

/// What `run_ws_auto` in the shell launcher emits before anything better is known.
pub(crate) const PLACEHOLDER_PREFIX: &str = "novo-";
/// Stable id of the managed workstream, exported by `ai-memory run`.
pub(crate) const WORKSTREAM_ID_ENV: &str = "AI_MEMORY_WORKSTREAM_ID";
/// How long the rename may hold up the first prompt. Same reasoning as
/// adoption: a hook sits between the user pressing enter and the model
/// answering, and one POST against a healthy server is milliseconds.
const RENAME_TIMEOUT: Duration = Duration::from_secs(5);

/// Whether a workstream name is the launcher's placeholder.
pub(crate) fn is_placeholder(name: &str) -> bool {
    name.starts_with(PLACEHOLDER_PREFIX)
}

/// Whether this prompt should rename the workstream: a placeholder name that
/// this session has not renamed yet.
pub(crate) fn needs_rename(data_dir: &Path, native_session_id: &str, name: &str) -> bool {
    is_placeholder(name) && !already_renamed(data_dir, native_session_id)
}

/// The fallback when the hook could not rename by itself: the model is asked
/// to run the command. Only reachable after a failure, so it is rare.
pub(crate) fn rename_nudge(current_name: &str) -> String {
    format!(
        "[nomear workstream] Este workstream ainda tem nome provisório \
         ('{current_name}') e a memória não conseguiu renomeá-lo sozinha. \
         Escolha um nome curto do trabalho atual e rode agora, silenciosamente, \
         como parte do seu trabalho:\n\n    \
         ai-memory rename-workstream --from '{current_name}' --to '<nome>'\n\n\
         Convenção: YYYY-MM-DD-<scope>:<slug> (kebab-case, ~12-18 chars): toca 1 serviço -> \
         <servico>:<slug>; vários serviços de 1 produto -> <produto>:<slug>; \
         vários produtos -> <slug>."
    )
}

/// What the first prompt of a launcher-managed session injects, if anything.
///
/// A successful rename is silent: the name is the hook's job, not the
/// model's. Failure — network, a server that is down, a timeout — is logged
/// and the model gets the manual instructions instead; the next prompt tries
/// the rename again, since the marker is only written on success.
pub(crate) async fn managed_prompt_context(
    workstream_name: &str,
    input: RenameInput<'_>,
) -> Option<String> {
    if !needs_rename(input.data_dir, input.native_session_id, workstream_name) {
        return None;
    }
    let outcome = tokio::time::timeout(RENAME_TIMEOUT, rename_placeholder(input)).await;
    let reason = match outcome {
        Ok(Ok(_)) => return None,
        Ok(Err(error)) => format!("{error:#}"),
        Err(_) => "o servidor da memória não respondeu a tempo".to_string(),
    };
    eprintln!("ai-memory hook warning: placeholder rename failed: {reason}");
    Some(rename_nudge(workstream_name))
}

/// The one rename failure worth a second attempt: another workstream of this
/// checkout already has the name (two sessions on the same task, same day).
fn is_duplicate_name(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<ServerResponseError>()
        .is_some_and(|response| {
            response.status() == reqwest::StatusCode::CONFLICT
                && response.body().contains("already exists")
        })
}

fn renamed_marker(data_dir: &Path, native_session_id: &str) -> PathBuf {
    data_dir.join("hook-state").join(format!(
        "renamed-{:x}",
        Sha256::digest(native_session_id.as_bytes())
    ))
}

/// Whether this session's placeholder was already renamed. The environment
/// still carries the placeholder for the whole session, so the hook cannot
/// tell from it alone.
pub(crate) fn already_renamed(data_dir: &Path, native_session_id: &str) -> bool {
    renamed_marker(data_dir, native_session_id).exists()
}

/// The marker holds the new name: useful when reading the state directory by
/// hand, and it costs nothing.
fn mark_renamed(data_dir: &Path, native_session_id: &str, new_name: &str) -> std::io::Result<()> {
    let marker = renamed_marker(data_dir, native_session_id);
    if let Some(parent) = marker.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(marker, new_name)
}

pub(crate) struct RenameInput<'a> {
    pub data_dir: &'a Path,
    pub server_url: &'a str,
    pub bearer: Option<&'a str>,
    pub cwd: &'a Path,
    pub native_session_id: &'a str,
    pub host_session_id: Option<&'a str>,
    pub first_prompt: &'a str,
    pub workstream_id: WorkstreamId,
    pub today: jiff::civil::Date,
}

/// Retitle the placeholder workstream after the session's first prompt.
/// Returns the stored name; the session is marked renamed only on success.
pub(crate) async fn rename_placeholder(input: RenameInput<'_>) -> Result<String> {
    let repository = inspect_repository(input.cwd)?;
    let config_dir = dirs::config_dir().unwrap_or_else(|| input.cwd.to_path_buf());
    let resolved = session_name::resolve_name(
        input.host_session_id,
        &config_dir,
        input.first_prompt,
        input.today,
    );
    let (workspace, project) = adopt_scope(&repository.cwd);
    let endpoint = ServerEndpoint::from_pair(
        Some(input.server_url.to_string()),
        input.bearer.map(str::to_string),
    );
    let request = |to: String| RenameManagedWorkstreamRequest {
        workspace: workspace.clone(),
        project: project.clone(),
        repo_fingerprint: repository.repo_fingerprint.clone(),
        worktree_fingerprint: repository.worktree_fingerprint.clone(),
        from: None,
        // By id, not by the placeholder name: the id is what `run` exported
        // and it cannot go stale, whereas a name lookup would race a rename
        // someone did by hand.
        workstream_id: Some(input.workstream_id),
        to,
    };
    let renamed: RenamedManagedWorkstream = match post_json(
        &endpoint,
        "/workstream/rename",
        &request(resolved.name.clone()),
    )
    .await
    {
        Ok(renamed) => renamed,
        Err(error) if is_duplicate_name(&error) => post_json(
            &endpoint,
            "/workstream/rename",
            &request(with_suffix(&resolved.name, input.native_session_id)),
        )
        .await
        .context("renaming the placeholder workstream (suffixed retry)")?,
        Err(error) => return Err(error.context("renaming the placeholder workstream")),
    };
    mark_renamed(input.data_dir, input.native_session_id, &renamed.to)
        .context("recording the workstream rename")?;
    Ok(renamed.to)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_launcher_placeholder_is_recognised() {
        assert!(is_placeholder("novo-491181"));
        assert!(!is_placeholder("nexus:bus-cancel"));
        assert!(!is_placeholder("2026-09-17-novo-recurso"));
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

    /// Recording HTTP stub: canned replies served in order (the last one
    /// repeats), request heads streamed back.
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
                let mut buf = [0_u8; 8192];
                let read = stream.read(&mut buf).await.unwrap_or(0);
                let _ = tx.send(String::from_utf8_lossy(&buf[..read]).into_owned());
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

    #[tokio::test]
    async fn a_duplicate_name_is_retried_once_with_a_session_suffix() {
        let tmp = tempfile::tempdir().unwrap();
        let id = WorkstreamId::new();
        let ok = serde_json::json!({
            "workstream_id": id,
            "from": "novo-1",
            "to": "2026-09-17-corrigir-o-timeout-nat-1"
        })
        .to_string();
        let (base, mut requests) = serve_sequence(vec![
            (
                "409 Conflict",
                r#"{"error":"workstream '2026-09-17-corrigir-o-timeout' already exists; select it with --workstream"}"#.into(),
            ),
            ("200 OK", ok),
        ])
        .await;

        let renamed = rename_placeholder(input(tmp.path(), &base, id, tmp.path()))
            .await
            .unwrap();

        assert_eq!(renamed, "2026-09-17-corrigir-o-timeout-nat-1");
        let first = request_body(&requests.recv().await.unwrap());
        let second = request_body(&requests.recv().await.unwrap());
        assert_eq!(first["to"], "2026-09-17-corrigir-o-timeout");
        assert_eq!(second["to"], "2026-09-17-corrigir-o-timeout-nat-1");
        assert!(already_renamed(tmp.path(), "nat-1"));
    }

    #[tokio::test]
    async fn a_conflict_that_is_not_a_duplicate_name_is_not_retried() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) = serve_requests(
            "409 Conflict",
            r#"{"error":"workstream is already active: owned by host:1"}"#.into(),
        )
        .await;

        let result =
            rename_placeholder(input(tmp.path(), &base, WorkstreamId::new(), tmp.path())).await;

        assert!(result.is_err());
        let _first = requests.recv().await.unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), requests.recv())
                .await
                .is_err(),
            "no second request"
        );
    }

    fn prompt_input<'a>(tmp: &'a Path, base: &'a str, id: WorkstreamId) -> RenameInput<'a> {
        input(tmp, base, id, tmp)
    }

    #[tokio::test]
    async fn first_prompt_renames_a_placeholder_and_stays_silent() {
        let tmp = tempfile::tempdir().unwrap();
        let id = WorkstreamId::new();
        let reply =
            serde_json::json!({"workstream_id": id, "from": "novo-1", "to": "2026-09-17-x"})
                .to_string();
        let (base, mut requests) = serve_requests("200 OK", reply).await;

        let context = managed_prompt_context("novo-1", prompt_input(tmp.path(), &base, id)).await;

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
        mark_renamed(tmp.path(), "nat-1", "2026-09-17-x").unwrap();
        let (base, mut requests) = serve_requests("200 OK", "{}".into()).await;

        let context = managed_prompt_context(
            "novo-1",
            prompt_input(tmp.path(), &base, WorkstreamId::new()),
        )
        .await;

        assert_eq!(context, None);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), requests.recv())
                .await
                .is_err(),
            "no request at all"
        );
    }

    #[tokio::test]
    async fn a_chosen_name_is_left_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, mut requests) = serve_requests("200 OK", "{}".into()).await;

        let context = managed_prompt_context(
            "nexus:bus-cancel",
            prompt_input(tmp.path(), &base, WorkstreamId::new()),
        )
        .await;

        assert_eq!(context, None);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), requests.recv())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn a_failed_rename_falls_back_to_the_nudge() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, _requests) =
            serve_requests("500 Internal Server Error", r#"{"error":"boom"}"#.into()).await;

        let context = managed_prompt_context(
            "novo-491181",
            prompt_input(tmp.path(), &base, WorkstreamId::new()),
        )
        .await
        .expect("the model gets the manual instructions");

        assert!(context.contains("rename-workstream"));
        assert!(context.contains("novo-491181"));
        assert!(
            !already_renamed(tmp.path(), "nat-1"),
            "the next prompt tries again"
        );
    }

    fn input<'a>(
        tmp: &'a Path,
        base: &'a str,
        id: WorkstreamId,
        config_dir: &'a Path,
    ) -> RenameInput<'a> {
        let _ = config_dir;
        RenameInput {
            data_dir: tmp,
            server_url: base,
            bearer: None,
            cwd: tmp,
            native_session_id: "nat-1",
            host_session_id: None,
            first_prompt: "Corrigir o timeout do nexus",
            workstream_id: id,
            today: jiff::civil::date(2026, 9, 17),
        }
    }

    #[tokio::test]
    async fn renames_by_id_to_the_dated_slug_and_marks_the_session() {
        let tmp = tempfile::tempdir().unwrap();
        let id = WorkstreamId::new();
        let reply = serde_json::json!({
            "workstream_id": id,
            "from": "novo-1",
            "to": "2026-09-17-corrigir-o-timeout"
        })
        .to_string();
        let (base, mut requests) = serve_requests("200 OK", reply).await;

        let renamed = rename_placeholder(input(tmp.path(), &base, id, tmp.path()))
            .await
            .unwrap();

        assert_eq!(renamed, "2026-09-17-corrigir-o-timeout");
        let request = requests.recv().await.unwrap();
        assert!(request.starts_with("POST /workstream/rename"), "{request}");
        let body = request.split("\r\n\r\n").nth(1).unwrap();
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["workstream_id"], serde_json::json!(id));
        assert_eq!(body["to"], "2026-09-17-corrigir-o-timeout");
        assert!(
            body.get("from").is_none(),
            "id and name selectors are exclusive: {body}"
        );
        assert!(already_renamed(tmp.path(), "nat-1"));
    }

    #[tokio::test]
    async fn a_failed_rename_leaves_the_session_unmarked() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, _requests) =
            serve_requests("500 Internal Server Error", r#"{"error":"boom"}"#.into()).await;

        let result =
            rename_placeholder(input(tmp.path(), &base, WorkstreamId::new(), tmp.path())).await;

        assert!(result.is_err());
        assert!(
            !already_renamed(tmp.path(), "nat-1"),
            "the next prompt must try again"
        );
    }
}
