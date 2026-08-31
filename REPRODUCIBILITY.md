# Reproducibility Protocol

This document separates the paper-compatible protocol from optional analysis
settings. It should be read before running or comparing experiments.

## Default data construction

- Poison sources are sampled deterministically with selection seed 42.
- `source_scope=all` is the default and reproduces the original all-class
  shuffle-and-split order exactly.
- `source_scope=non_target` is used only for the auxiliary source-composition
  sensitivity analysis. It excludes examples whose original label already
  equals the attack target, then takes the same nominal number of sources.
- The trigger is inserted as a prefix and the selected source label is replaced
  by the target label.
- ASR is evaluated only on originally non-target evaluation examples after
  inserting the same trigger.
- Data generation is independent of the distillation seed. Its seed and exact
  selected indices are recorded in `poison_metadata.json`.

For AG News with ratio 0.001, trigger `the`, target label 1, and selection seed
42, both source scopes construct 120 poison-source records. The all-class pool
contains 88 originally non-target and 32 originally target-class records. The
non-target pool contains 120 and 0, respectively. These counts describe the
generated subsets for this fixed configuration; they are not universal
properties of SAAB.

## Legacy-compatible training behavior

The public release preserves the paper-compatible training and evaluation
behavior. This is necessary because silent runtime corrections would produce a
different experimental protocol.

- Training uses 10 distillation epochs and a fixed distillation seed of 42 for
  the supplied scripts.
- SAAB uses one synthetic example per class, soft synthetic labels, CLS
  attention labels, one learner step, attention logits of +20 at token index 1
  and -20 elsewhere, and frozen synthetic attention labels.
- The clean and poison loaders use batch size 32 and `drop_last=True`. The
  poison loader is wrapped by `itertools.cycle`, so the first complete batches
  are cached and reused. With 120 generated AG News poison-source records, 96
  records enter the cached complete batches in a run. With 67 generated SST-2
  records, 64 enter them.
- Training-time `evaluate_fast` is a lightweight legacy validation routine used
  for checkpoint selection. The final report loads the checkpoint with the
  lowest recorded validation loss.
- AG News has no separate validation split in the source dataset. The legacy
  pipeline maps its official test split to the `validation` key.
- CTA and ASR are computed by two separate evaluator calls. Each call trains
  its own group of victim models from the distilled data.

These details should not be interpreted as a recommended general evaluation
protocol. A future protocol-corrected release should use a dedicated validation
split, explicit victim seeds, paired CTA/ASR evaluation, complete poison-stream
coverage, and independent distillation seeds. Results from different protocols
should not be compared as if only the attack method changed.

## Meaning of the reported uncertainty

The supplied scripts set `base.seed=42` and `evaluate.n_eval_model=5`. A value
formatted as `mean ± standard deviation` summarizes five victim-model
evaluations of one fixed distilled-data run. It does not summarize five
independent distillation runs.

## Result and artifact locations

Hydra writes each run below:

```text
save/<task>.<model>/<method>/<timestamp>/
```

Important files are:

- `results.json`: final CTA/ASR summary;
- `.hydra/config.yaml`: resolved experiment configuration;
- `checkpoints/best-ckpt/`: checkpoint selected by the legacy validation rule;
- `checkpoints/last-ckpt/`: final-epoch distilled data;
- `poison_metadata.json`: data-generation provenance in the input data folder.


## Reproduction checklist

1. Record the Git commit and environment file.
2. Generate data with an explicit source scope and selection seed.
3. Retain `poison_metadata.json` with the experiment artifacts.
4. Run the corresponding script without changing its seed or batching options.
5. Preserve `results.json`, Hydra configuration, and the selected checkpoint.
6. State whether uncertainty is over victim evaluations or independent
   distillation runs.
