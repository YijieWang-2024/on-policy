# Work Notes

## 2026-06-10 - MPE Gymnasium migration and rollout semantics

### Summary

Completed the first substantial update after the repository was reduced to the
MPE-focused version. The work modernizes the retained MPE stack for Gymnasium
and NumPy 2, and corrects how episode termination and time-limit truncation are
handled during rollout and return calculation.

### Key changes

- Migrated MPE environments, spaces, wrappers, training, evaluation, and
  rendering flows from the legacy Gym API to Gymnasium.
- Replaced global scenario randomness with seeded per-environment generators,
  making seeded resets reproducible.
- Rebuilt the retained vector environment wrappers with Gymnasium-style reset,
  step, autoreset, seed, and final-observation handling.
- Separated true task termination from time-limit truncation:
  - recurrent state and GAE continuation stop at either boundary;
  - critic bootstrapping remains enabled for truncation;
  - critic bootstrapping is disabled for true termination.
- Updated shared and separated MPE runners and replay buffers to preserve and
  evaluate the actual final observation before autoreset.
- Added PPO diagnostics for approximate KL divergence, clipping fraction, and
  explained variance, plus optional `--target_kl` early stopping.
- Updated dependencies for Gymnasium and NumPy 2 compatibility.
- Added focused migration tests and documented the new semantics in `README.md`.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
git diff --check
```

All 5 migration tests passed. The tests cover all retained MPE scenarios,
reproducible seeded resets, time-limit truncation, vector autoreset final
observations, and truncation-versus-termination bootstrapping.

### Notes

- `origin` is the private archive repository `YijieWang-2024/on-policy`.
- `upstream` remains the original `marlbenchmark/on-policy` repository.
- MPE environments should report external time limits as truncation and reserve
  termination for true task-ending conditions.
