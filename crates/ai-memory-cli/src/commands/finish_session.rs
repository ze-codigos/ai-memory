//! Finalisation of kept ledgers (`adopted_state`), run by the detached spool
//! drainer: a managed run whose final import failed when the launcher's child
//! exited is closed here instead of being lost.
//!
//! The drainer already fires on every boundary the harness reaches
//! (`session-end`, `stop`, `pre-compact`) and on `session-start` when a record
//! is pending. Putting the import here keeps transcript export and the batched
//! POSTs out of the hook's latency, and gets retries for free.
//!
//! Every record the launcher writes is `ended` by construction, so the only
//! outcomes are "close it" and "give up". A record that is NOT ended was
//! written by a binary that still adopted sessions from the hook (0.12–0.17);
//! nothing marks such a record ended any more, so it is abandoned at once,
//! with a warning, rather than re-exported at every boundary forever.

use std::path::Path;

use crate::commands::adopted_state::AdoptedRun;

/// A record older than this is abandoned rather than imported. Long enough to
/// survive a weekend, short enough that a ledger the server keeps refusing
/// does not sit in the directory forever.
const MAX_AGE_SECS: i64 = 48 * 3_600;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Action {
    /// Import and close.
    Close,
    /// Not worth importing; drop the record.
    Drop,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Closed,
    Dropped,
    /// Transient failure — state kept so the next boundary retries.
    Kept,
}

/// Why a record is dropped, for the warning: the two reasons call for
/// different reactions from whoever reads the log.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DropReason {
    /// Past the age cap: the server kept refusing (or was never reachable).
    TooOld,
    /// Written by a hook-adoption binary; nothing can end it now.
    Legacy,
}

/// The server refuses, by design, a lapsed run it never linked, and that
/// verdict never changes — so past the cap the record is abandoned (with a
/// warning naming the transcript) instead of being retried forever.
pub(crate) fn plan_for(run: &AdoptedRun, now: i64, max_age_secs: i64) -> Action {
    if !run.ended {
        return Action::Drop;
    }
    if now.saturating_sub(run.adopted_at) > max_age_secs {
        return Action::Drop;
    }
    Action::Close
}

pub(crate) fn drop_reason(run: &AdoptedRun) -> DropReason {
    if run.ended {
        DropReason::TooOld
    } else {
        DropReason::Legacy
    }
}

/// Whether retrying could ever succeed.
///
/// Deliberately narrow. Only two verdicts are permanent: a run the server
/// cannot find at all, and one whose native session is not the one we claim.
/// Neither can change by asking again.
///
/// A 409 `managed run is not active` is NOT on that list, and the reason is
/// concrete: a server that predates the lapsed-lease import answers exactly
/// that to every adopted session older than its 90-second lease. Treating it
/// as permanent threw the ledger away instead of waiting for the server to
/// catch up — measured against the live dev server, not hypothesised. Keeping
/// the state costs a retry per boundary until the 48-hour cap collects it;
/// dropping it costs the transcript, permanently.
pub(crate) fn is_permanent(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}").to_lowercase();
    text.contains("404") || text.contains("does not match")
}

pub(crate) async fn finalize_adopted_runs(data_dir: &Path) -> Vec<Outcome> {
    let pending = crate::commands::adopted_state::list(data_dir);
    if pending.is_empty() {
        return Vec::new();
    }
    let now = jiff::Timestamp::now().as_second();
    let client = crate::commands::hook_capture::build_client();
    let token = std::env::var(crate::commands::hook_spool::LIVE_TOKEN_ENV).ok();
    let bearer =
        crate::commands::hook_spool::resolve_bearer(&client, data_dir, token.as_deref()).await;
    let home = dirs::home_dir().unwrap_or_default();

    let mut outcomes = Vec::with_capacity(pending.len());
    for run in pending {
        if plan_for(&run, now, MAX_AGE_SECS) == Action::Drop {
            match drop_reason(&run) {
                DropReason::TooOld => eprintln!(
                    "ai-memory hook-drain warning: giving up on the transcript of workstream '{}' (native session {}, {}): it could not be imported within {}h",
                    run.workstream_name,
                    run.native_session_id,
                    run.cwd.display(),
                    MAX_AGE_SECS / 3_600
                ),
                DropReason::Legacy => eprintln!(
                    "ai-memory hook-drain warning: dropping the record of workstream '{}' (native session {}, {}): it was written by a version that adopted sessions from the hook, and nothing can close it now; the server run lapses by itself",
                    run.workstream_name,
                    run.native_session_id,
                    run.cwd.display(),
                ),
            }
            crate::commands::adopted_state::remove(data_dir, &run.native_session_id);
            outcomes.push(Outcome::Dropped);
            continue;
        }
        // One broken run must not stop the others from being finalised.
        let outcome = match import_one(&run, &home, bearer.as_deref()).await {
            Ok(()) => {
                crate::commands::adopted_state::remove(data_dir, &run.native_session_id);
                Outcome::Closed
            }
            Err(error) if is_permanent(&error) => {
                eprintln!(
                    "ai-memory hook-drain warning: adopted run {} dropped, the server will not accept it: {error:#}",
                    run.run_id
                );
                crate::commands::adopted_state::remove(data_dir, &run.native_session_id);
                Outcome::Dropped
            }
            Err(error) => {
                eprintln!(
                    "ai-memory hook-drain warning: adopted run {} kept for the next boundary: {error:#}",
                    run.run_id
                );
                Outcome::Kept
            }
        };
        outcomes.push(outcome);
    }
    outcomes
}

async fn import_one(run: &AdoptedRun, home: &Path, bearer: Option<&str>) -> anyhow::Result<()> {
    let endpoint = crate::http_client::ServerEndpoint::from_pair(
        Some(run.server_url.clone()),
        bearer.map(str::to_string),
    );
    // The launcher recorded which harness it launched, where it looked for
    // the transcript, and the exit code and checkpoint of the moment; a record
    // without them gets the process defaults.
    let harness = run
        .agent
        .and_then(crate::commands::run::managed_harness_from_agent)
        .unwrap_or(ai_memory_workstream::ManagedHarness::Claude);
    let transcript = crate::commands::run::export_after_flush(
        harness,
        run.home.as_deref().unwrap_or(home),
        &run.cwd,
        run.session_dir.as_deref(),
        Some(&run.native_session_id),
        None,
    )
    .await;
    // Best effort: a checkpoint we cannot read is metadata, not a reason to
    // drop the transcript it would have annotated.
    let checkpoint = run.checkpoint.clone().unwrap_or_else(|| {
        ai_memory_workstream::inspect_repository(&run.cwd)
            .map(|identity| identity.checkpoint)
            .unwrap_or_default()
    });
    crate::commands::run::import_batches(
        &endpoint,
        &run.run_path,
        transcript,
        checkpoint,
        run.exit_code,
        true,
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::adopted_state::sample_run;

    #[test]
    fn a_fresh_kept_ledger_closes() {
        let mut run = sample_run("nat-1");
        run.ended = true;
        run.kept_by_launcher = true;
        assert_eq!(
            plan_for(&run, run.adopted_at + 3_600, MAX_AGE_SECS),
            Action::Close
        );
    }

    #[test]
    fn a_stale_kept_ledger_is_abandoned() {
        // O servidor recusa pra sempre um run vencido que nunca foi linkado;
        // insistir a cada boundary por semanas nao recupera nada.
        let mut run = sample_run("nat-1");
        run.ended = true;
        run.kept_by_launcher = true;
        assert_eq!(
            plan_for(&run, run.adopted_at + MAX_AGE_SECS + 1, MAX_AGE_SECS),
            Action::Drop
        );
        assert_eq!(drop_reason(&run), DropReason::TooOld);
    }

    #[test]
    fn a_legacy_unended_record_is_dropped_at_once() {
        // Written by a binary that adopted sessions from the hook; nothing
        // marks it ended any more, so importing it incrementally forever
        // would only re-export the transcript at every boundary.
        let run = sample_run("nat-1");
        assert_eq!(
            plan_for(&run, run.adopted_at + 60, MAX_AGE_SECS),
            Action::Drop
        );
        assert_eq!(drop_reason(&run), DropReason::Legacy);
    }

    #[test]
    fn a_legacy_ended_record_still_closes_within_the_cap() {
        // A 0.12-0.17 record that saw its SessionEnd has a complete
        // transcript on disk; it gets the same chance as a kept ledger, and
        // the same cap.
        let mut run = sample_run("nat-1");
        run.ended = true;
        assert_eq!(
            plan_for(&run, run.adopted_at + 3_600, MAX_AGE_SECS),
            Action::Close
        );
        assert_eq!(
            plan_for(&run, run.adopted_at + MAX_AGE_SECS + 1, MAX_AGE_SECS),
            Action::Drop
        );
    }

    #[test]
    fn a_missing_run_or_a_foreign_session_is_permanent() {
        assert!(is_permanent(&anyhow::anyhow!(
            "server returned 404 Not Found"
        )));
        assert!(is_permanent(&anyhow::anyhow!(
            "managed run X is expired and its native session id does not match the linked one"
        )));
    }

    #[test]
    fn a_run_that_is_not_active_is_kept_not_dropped() {
        // The exact body a server without the lapsed-lease import returns for
        // every adopted session older than its 90s lease. Observed against the
        // live dev server: classifying it as permanent discarded the ledger
        // instead of waiting for the server to be updated.
        assert!(!is_permanent(&anyhow::anyhow!(
            "server returned 409 Conflict: {{\"error\":\"managed run is not active\"}}"
        )));
    }

    #[test]
    fn a_network_failure_is_transient() {
        // The whole point of keeping state on a transient error: a server that
        // is down for thirty seconds at SessionEnd must not cost the ledger.
        assert!(!is_permanent(&anyhow::anyhow!(
            "error sending request: connection refused"
        )));
        assert!(!is_permanent(&anyhow::anyhow!("server returned 503")));
    }
}
