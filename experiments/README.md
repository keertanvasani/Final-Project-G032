# Experiments — Part 1

Everything needed to reproduce the Part 1 evaluation, in dependency order.
`PROTOCOL.md` is the binding specification; the rest is its implementation.

## Layout

```
PROTOCOL.md                 the binding evaluation spec (read first)
audit_group_leakage.py      quantifies dive-level leakage in supplied splits
make_splits.py              builds dive-disjoint Phase A splits
baseline/
  freeze_baseline.py        writes IMAGE_V1_BASELINE.json (params, hashes, costs)
  IMAGE_V1_BASELINE.json    the frozen baseline manifest
splits/
  phase_a_splits.json       J_cal / J_policy / J_test by dive (generated)
eval/
  harness.py                GT loading, matching, five-way taxonomy (§2)
  metrics.py                selective-prediction metrics + dive bootstrap (§3, §7.2)
run_phase_a.py              cache perception over a split, then report metrics
plot_phase_a.py             render the risk-coverage figure from a report
results/phase_a/
  RESULTS.md                Phase A findings (committed)
  *_risk_coverage.{pdf,png} motivating figure (committed)
  *_detections.jsonl        cached predictions (gitignored, regenerable)
  *_report.json             computed metrics (gitignored, regenerable)
```

## Reproduce from scratch

```bash
source venv/bin/activate

# 1. confirm the supplied splits leak (they do: 57.3% test contamination)
python experiments/audit_group_leakage.py

# 2. build dive-disjoint splits
python experiments/make_splits.py

# 3. freeze the baseline (params, weight hashes, measured per-action cost)
python experiments/baseline/freeze_baseline.py --n 30

# 4. Phase A — run the frozen baseline on deep-sea imagery
python experiments/run_phase_a.py cache  --split J_test   # ~7 min, resumable
python experiments/run_phase_a.py report --split J_test
python experiments/plot_phase_a.py        --split J_test
```

Steps 1–3 are deterministic. Step 4's `cache` is resumable — re-running skips
images already cached; `report` and `plot` read the cache and are cheap.

## Design decisions

All six open decisions are resolved in `PROTOCOL.md §9`; three confirmatory
hypotheses are pre-registered in `§7.4`. The protocol is ACTIVE — amend by
dated addition, not in-place edits.

## Test-set discipline (important)

`J_test` has been read once, for the frozen-baseline reference in
`results/phase_a/RESULTS.md`. It must not be read again until the final
agent-vs-baseline evaluation. **Every Phase 1 design step uses `J_cal` /
`J_policy`.** See `PROTOCOL.md §8`.

## Status

- **Phase 0 (complete):** protocol, leakage audit, dive-disjoint splits,
  baseline freeze, evaluation harness, Phase A baseline result + figure.
- **Phase 1 (next):** cache `J_cal` / `J_policy`; calibration; uncertainty
  features; OOD scorer; meta-confidence model; agent state, actions, budget;
  learned policy; ablations.
