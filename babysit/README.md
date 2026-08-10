# babysit — a Claude Code skill for watching a training run

Babysit a deep-learning training run on SLURM so it can run **unattended**. Each tick it **catches you
up** when healthy and **autonomously debugs, fixes, and resumes** when not — pinging you on **Slack**
only when something's worth telling. You leave an ablation running; you come back to either "still
healthy, +Δ steps" or "broke at X, fixed by Y, training again."

Built as a [Claude Code](https://claude.com/claude-code) skill. Two parts:

- **`train_watch.py`** — a stdlib-only, read-only **detector**. Auto-discovers active runs, reads
  per-run `max_steps`, and flags: non-finite loss / guard counters, divergence, plateau, throughput
  drop, stall (on-node only), chain-exhausted, and burned-segment waves (splitting preemption noise
  from real burns). Emits `--json` for the skill to branch on. 25 unit tests.
- **`SKILL.md`** — the **playbook**: measure → branch (catch-me-up digest **or** debug-fix-resume via
  `systematic-debugging` + a hypothesis Workflow) → resume → verify → Slack. `failure-catalog.md` is a
  14-mode field guide that warm-starts the debugger.

## What it detects

`train_watch.py` answers, per run: **is it alive / healthy / moving / will it finish**, plus
generative-model signals. Severity is guard-aware — a transient grad-NaN that a guard self-heals is a
WARN; a param-NaN or non-finite loss is CRITICAL. It won't cry wolf: on a chained SLURM job,
all-PENDING / zero-RUNNING is *healthy* (between segments), not dead.

```bash
python3 train_watch.py            # human health report over auto-discovered runs
python3 train_watch.py --json     # machine-readable (the skill consumes this)
python3 train_watch.py --runs a b # scope to specific runs
```

## Assumptions & configuration

Assumes a **PyTorch-Lightning-style layout**: each run is a directory with
`csv/version_*/metrics.csv` (one `version_*` per resume/segment) and optionally `resolved_config.yaml`
(read for `train.max_steps` and the LR schedule). Configure roots without editing code:

```bash
export BABYSIT_RUNS_ROOT=/path/to/runs      # default: ./runs
export BABYSIT_LOGS_DIR=/path/to/logs       # default: ./logs
export BABYSIT_SEGMENT_HOURS=4              # walltime of one SLURM segment
```

If your logger uses different column names, adapt `read_run()` (it keys on `train/loss` and optional
`train/nonfinite_*` guard counters). SLURM job→run mapping uses a `config=<run>` token on the first log
line (`<LOGS_DIR>/*_<jobid>.out`); adjust `job_to_run()` to match your launcher.

## Slack

```bash
cp notify.txt.example notify.txt         # then set it to a channel id or your user id (U… = DM)
export BABYSIT_SLACK_TOKEN=xoxb-...       # Slack bot token: chat:write (+ im:write to DM)
bash slack_notify.sh "hello from babysit"
```

`notify.txt` is gitignored. **Notify policy: only issues + big milestones; silence = healthy** — no
routine progress pings, and it notifies on state *change* (via a ledger), not every tick.

## Usage

In a dedicated tmux + Claude Code session:

```
babysit my-training-run     # continuous by default: ticks ~every 30 min, self-schedules, Slacks on issues
check once                  # a single digest, no scheduling
stop babysitting            # end the loop
```

## Run the tests

```bash
python3 tests/test_train_watch.py
```

## Safety notes

The debug-fix path can `scancel`/resubmit and edit code — grant that per-run. Guardrails in `SKILL.md`
protect *healthy* jobs (name-guarded scancel), preserve NaN evidence (never delete checkpoints;
rename-aside), require reproduce-before-edit, and log every action to a ledger.

## License

MIT.
