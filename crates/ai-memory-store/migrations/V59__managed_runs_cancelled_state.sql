-- Expand managed_runs.state CHECK for the `cancelled` state.
--
-- V31 enumerated 'active', 'finished', 'expired', which made a deliberate
-- cancel indistinguishable from a lapsed lease: `cancel_run` wrote 'expired'
-- too. That was harmless only while `finish_run` refused every non-active
-- run. It stops being harmless now that a lapsed lease is an accepted finish
-- path — sessions adopted by a lifecycle hook have no parent process to
-- heartbeat, so they always reach finish expired, and reconciliation after a
-- crash arrives later still.
--
-- Without a separate state, accepting a lapsed lease would also accept a run
-- that `run` cancelled after linking its native session (the launcher links
-- before spawning and cancels on any later failure), silently contradicting
-- cancel's contract: release the lease WITHOUT importing any events.
--
-- Rows already written as 'expired' by a past cancel stay 'expired'. They are
-- terminal either way and no adopted-session state references them, so there
-- is nothing to back-fill.

PRAGMA foreign_keys = OFF;

CREATE TABLE managed_runs_new (
    id                  BLOB PRIMARY KEY NOT NULL,
    workstream_id       BLOB NOT NULL REFERENCES workstreams(id) ON DELETE CASCADE,
    agent_kind          TEXT NOT NULL,
    lease_owner         TEXT NOT NULL,
    native_session_id   TEXT,
    state               TEXT NOT NULL CHECK (state IN ('active', 'finished', 'expired', 'cancelled')),
    sync_after          INTEGER NOT NULL DEFAULT 0,
    sync_through        INTEGER NOT NULL DEFAULT 0,
    context_delivered   INTEGER NOT NULL DEFAULT 0 CHECK (context_delivered IN (0, 1)),
    lease_expires_at    INTEGER NOT NULL,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    exit_code           INTEGER
);

INSERT INTO managed_runs_new SELECT * FROM managed_runs;

DROP TABLE managed_runs;

ALTER TABLE managed_runs_new RENAME TO managed_runs;

-- Both V31 indexes: dropping the old table dropped them with it. The partial
-- unique index is the one that enforces "at most one active run per
-- workstream" — losing it would let two live sessions share a workstream.
CREATE UNIQUE INDEX idx_managed_runs_one_active
    ON managed_runs(workstream_id)
    WHERE state = 'active';

CREATE INDEX idx_managed_runs_lease
    ON managed_runs(state, lease_expires_at);

PRAGMA foreign_keys = ON;
