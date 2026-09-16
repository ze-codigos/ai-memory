//! Local state for an adopted session: the run a hook opened on its behalf.
//!
//! On disk rather than in the environment because a hook is a child process —
//! it cannot stamp `AI_MEMORY_RUN_ID` onto the harness that is already
//! running. Keyed by the native session id, which every hook payload carries.

use std::path::{Path, PathBuf};

use ai_memory_core::{AgentKind, ManagedRunId, WorkstreamCheckpoint};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AdoptedRun {
    pub run_id: ManagedRunId,
    /// `/workstream/runs/<id>`, so the drainer does not have to rebuild it.
    pub run_path: String,
    pub native_session_id: String,
    pub workstream_name: String,
    /// Named from a prompt slug; the hook asks the model to rename it.
    pub provisional: bool,
    /// The drainer finalises this run and has no `HookArgs` to read a URL from.
    pub server_url: String,
    pub cwd: PathBuf,
    pub adopted_at: i64,
    /// SessionEnd sets this; the drainer closes only what is marked ended.
    #[serde(default)]
    pub ended: bool,
    /// Where the launcher looked for the native transcript, when it knew
    /// better than the defaults (`AI_MEMORY_HOME`, `CLAUDE_CONFIG_DIR`). A
    /// hook-adopted session leaves both unset and the drainer uses the
    /// process defaults, as it always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_dir: Option<PathBuf>,
    /// Set by the launcher when its own final import failed. Such a record
    /// closes the run with the launcher's exit code and checkpoint, for the
    /// harness it launched, and is abandoned past the age cap: the server
    /// refuses a lapsed run it never linked, and retrying that at every
    /// boundary forever helps nobody.
    #[serde(default)]
    pub kept_by_launcher: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<WorkstreamCheckpoint>,
}

pub(crate) fn state_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("adopted-runs")
}

/// The session id arrives from the environment and becomes a file name, so it
/// is not trusted verbatim: anything outside `[A-Za-z0-9_-]` collapses to `_`.
/// That also keeps a caller from walking out of the state directory.
fn file_name(native_session_id: &str) -> String {
    let safe: String = native_session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{safe}.json")
}

fn path_for(data_dir: &Path, native_session_id: &str) -> PathBuf {
    state_dir(data_dir).join(file_name(native_session_id))
}

/// Write through a temporary file and rename, so `list` can never observe a
/// half-written state: the drainer reads this directory while hooks write it.
pub(crate) fn save(data_dir: &Path, run: &AdoptedRun) -> anyhow::Result<()> {
    let dir = state_dir(data_dir);
    std::fs::create_dir_all(&dir)?;
    let target = path_for(data_dir, &run.native_session_id);
    let temp = target.with_extension("json.tmp");
    std::fs::write(&temp, serde_json::to_vec_pretty(run)?)?;
    std::fs::rename(&temp, &target)?;
    Ok(())
}

pub(crate) fn load(data_dir: &Path, native_session_id: &str) -> Option<AdoptedRun> {
    let raw = std::fs::read(path_for(data_dir, native_session_id)).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// Best-effort: a state that cannot be deleted is retried on the next
/// boundary, and failing the hook over it would be worse than the retry.
pub(crate) fn remove(data_dir: &Path, native_session_id: &str) {
    let _ = std::fs::remove_file(path_for(data_dir, native_session_id));
}

/// Skips anything that does not deserialise. One corrupt state must not stop
/// every other session from being finalised.
pub(crate) fn list(data_dir: &Path) -> Vec<AdoptedRun> {
    let Ok(entries) = std::fs::read_dir(state_dir(data_dir)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| std::fs::read(entry.path()).ok())
        .filter_map(|raw| serde_json::from_slice(&raw).ok())
        .collect()
}

/// Shared with the adoption and finalisation tests, which both need a state to
/// act on. Lives outside `mod tests` so those modules can import it.
#[cfg(test)]
pub(crate) fn sample_run(session: &str) -> AdoptedRun {
    AdoptedRun {
        run_id: ManagedRunId::new(),
        run_path: "/workstream/runs/abc".into(),
        native_session_id: session.into(),
        workstream_name: "ajuste-checkout".into(),
        provisional: false,
        server_url: "https://memory-test.example".into(),
        cwd: PathBuf::from("/repo"),
        adopted_at: 1_700_000_000,
        ended: false,
        home: None,
        session_dir: None,
        kept_by_launcher: false,
        agent: None,
        exit_code: None,
        checkpoint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_then_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let run = sample_run("nat-1");
        save(tmp.path(), &run).unwrap();
        assert_eq!(load(tmp.path(), "nat-1"), Some(run));
    }

    #[test]
    fn load_absent_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(load(tmp.path(), "nao-existe"), None);
    }

    #[test]
    fn save_overwrites_the_same_session() {
        let tmp = tempfile::tempdir().unwrap();
        save(tmp.path(), &sample_run("nat-1")).unwrap();
        let mut ended = sample_run("nat-1");
        ended.ended = true;
        save(tmp.path(), &ended).unwrap();
        assert!(load(tmp.path(), "nat-1").unwrap().ended);
        assert_eq!(list(tmp.path()).len(), 1);
    }

    #[test]
    fn remove_deletes() {
        let tmp = tempfile::tempdir().unwrap();
        save(tmp.path(), &sample_run("nat-1")).unwrap();
        remove(tmp.path(), "nat-1");
        assert_eq!(load(tmp.path(), "nat-1"), None);
    }

    #[test]
    fn remove_is_quiet_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        remove(tmp.path(), "nunca-existiu");
    }

    #[test]
    fn list_is_empty_without_a_state_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(list(tmp.path()).is_empty());
    }

    #[test]
    fn list_returns_every_state() {
        let tmp = tempfile::tempdir().unwrap();
        save(tmp.path(), &sample_run("nat-1")).unwrap();
        save(tmp.path(), &sample_run("nat-2")).unwrap();
        let mut ids: Vec<_> = list(tmp.path())
            .into_iter()
            .map(|r| r.native_session_id)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["nat-1", "nat-2"]);
    }

    #[test]
    fn corrupt_file_is_skipped_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(state_dir(tmp.path())).unwrap();
        std::fs::write(state_dir(tmp.path()).join("quebrado.json"), "{ nao json").unwrap();
        save(tmp.path(), &sample_run("nat-1")).unwrap();
        assert_eq!(list(tmp.path()).len(), 1);
    }

    #[test]
    fn session_id_cannot_escape_the_state_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mut run = sample_run("x");
        run.native_session_id = "../../fuga".into();
        save(tmp.path(), &run).unwrap();

        assert!(!tmp.path().join("fuga.json").exists());
        assert!(load(tmp.path(), "../../fuga").is_some());
    }
}
