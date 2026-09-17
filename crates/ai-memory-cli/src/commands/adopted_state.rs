//! Local state for an adopted session: the run a hook opened on its behalf.
//!
//! On disk rather than in the environment because a hook is a child process —
//! it cannot stamp `AI_MEMORY_RUN_ID` onto the harness that is already
//! running. Keyed by the native session id, which every hook payload carries.

use std::path::{Path, PathBuf};

use ai_memory_core::{AgentKind, ManagedRunId, WorkstreamCheckpoint, WorkstreamId};
use serde::{Deserialize, Serialize};

/// How long a session link is kept. A desktop conversation reopened weeks
/// later is rare, and the server holds the same map without any cap; the
/// link only detects fragmentation, it never decides where a session goes.
const LINK_MAX_AGE_SECS: i64 = 30 * 24 * 3_600;

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

/// Which workstream a native session was adopted into. Outlives the run:
/// `remove` deletes the run record when the drainer closes it, and the next
/// adoption of the same session (the desktop app reopening a conversation)
/// compares where the server put it with where it was — the server's map
/// decides, this only notices a split. Never used to select by name: a
/// name can be freed by a rename and taken by another conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SessionLink {
    pub native_session_id: String,
    pub workstream_id: WorkstreamId,
    pub workstream_name: String,
    /// The link is checkout-local, like the server's map.
    pub cwd: PathBuf,
    pub server_url: String,
    pub linked_at: i64,
}

/// A subdirectory, so `list` (which reads the run records at the top level)
/// never tries to parse a link as a run.
fn links_dir(data_dir: &Path) -> PathBuf {
    state_dir(data_dir).join("sessions")
}

fn link_path(data_dir: &Path, native_session_id: &str) -> PathBuf {
    links_dir(data_dir).join(file_name(native_session_id))
}

pub(crate) fn save_link(data_dir: &Path, link: &SessionLink) -> anyhow::Result<()> {
    let dir = links_dir(data_dir);
    std::fs::create_dir_all(&dir)?;
    prune_links(&dir, link.linked_at);
    let target = link_path(data_dir, &link.native_session_id);
    let temp = target.with_extension("json.tmp");
    std::fs::write(&temp, serde_json::to_vec_pretty(link)?)?;
    std::fs::rename(&temp, &target)?;
    Ok(())
}

/// The link for this session in this checkout against this server, if any.
/// A link that does not match either is not "wrong", it is another
/// adoption's: ignore it rather than reopen the wrong workstream.
pub(crate) fn load_link(
    data_dir: &Path,
    native_session_id: &str,
    cwd: &Path,
    server_url: &str,
) -> Option<SessionLink> {
    let raw = std::fs::read(link_path(data_dir, native_session_id)).ok()?;
    let link: SessionLink = serde_json::from_slice(&raw).ok()?;
    (link.native_session_id == native_session_id
        && link.cwd == cwd
        && link.server_url == server_url)
        .then_some(link)
}

/// Best effort, on every save, and by age only: a link this binary cannot
/// parse may belong to a newer one on the same machine (the wrapper probes
/// two install paths), and deleting it would erase the newer binary's
/// memory. A stale link is a missed notice, not a failure, so this never
/// blocks the adoption that triggered it.
fn prune_links(dir: &Path, now: i64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let expired = std::fs::read(&path)
            .ok()
            .and_then(|raw| serde_json::from_slice::<SessionLink>(&raw).ok())
            .is_some_and(|link| now.saturating_sub(link.linked_at) > LINK_MAX_AGE_SECS);
        if expired {
            let _ = std::fs::remove_file(path);
        }
    }
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

    fn sample_link(session: &str, linked_at: i64) -> SessionLink {
        SessionLink {
            native_session_id: session.into(),
            workstream_id: WorkstreamId::new(),
            workstream_name: "ajuste-checkout".into(),
            cwd: PathBuf::from("/repo"),
            server_url: "https://memory-test.example".into(),
            linked_at,
        }
    }

    #[test]
    fn a_link_outlives_the_run_record() {
        let tmp = tempfile::tempdir().unwrap();
        let link = sample_link("nat-1", 1_700_000_000);
        save(tmp.path(), &sample_run("nat-1")).unwrap();
        save_link(tmp.path(), &link).unwrap();
        remove(tmp.path(), "nat-1");
        assert_eq!(load(tmp.path(), "nat-1"), None);
        assert_eq!(
            load_link(
                tmp.path(),
                "nat-1",
                Path::new("/repo"),
                "https://memory-test.example"
            ),
            Some(link)
        );
        // The links directory is not mistaken for run records.
        assert!(list(tmp.path()).is_empty());
    }

    #[test]
    fn a_link_is_checkout_and_server_local() {
        let tmp = tempfile::tempdir().unwrap();
        save_link(tmp.path(), &sample_link("nat-1", 1_700_000_000)).unwrap();
        assert!(
            load_link(
                tmp.path(),
                "nat-1",
                Path::new("/outro-repo"),
                "https://memory-test.example"
            )
            .is_none()
        );
        assert!(
            load_link(
                tmp.path(),
                "nat-1",
                Path::new("/repo"),
                "https://memory-prod.example"
            )
            .is_none()
        );
    }

    #[test]
    fn pruning_leaves_a_link_it_cannot_parse_alone() {
        // A newer binary on the same machine may have written it.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(links_dir(tmp.path())).unwrap();
        let foreign = links_dir(tmp.path()).join("de-outro-binario.json");
        std::fs::write(&foreign, "{\"campo_novo\": 1}").unwrap();
        save_link(tmp.path(), &sample_link("nova", 1_700_000_000)).unwrap();
        assert!(foreign.exists());
    }

    #[test]
    fn saving_a_link_prunes_the_stale_ones() {
        let tmp = tempfile::tempdir().unwrap();
        save_link(tmp.path(), &sample_link("velha", 1_700_000_000)).unwrap();
        save_link(
            tmp.path(),
            &sample_link("nova", 1_700_000_000 + LINK_MAX_AGE_SECS + 1),
        )
        .unwrap();
        assert!(
            load_link(
                tmp.path(),
                "velha",
                Path::new("/repo"),
                "https://memory-test.example"
            )
            .is_none()
        );
        assert!(
            load_link(
                tmp.path(),
                "nova",
                Path::new("/repo"),
                "https://memory-test.example"
            )
            .is_some()
        );
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
