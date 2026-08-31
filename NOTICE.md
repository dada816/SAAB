# Notice and provenance

This repository is a research derivative of
[`arumaekawa/dataset-distillation-with-attention-labels`](https://github.com/arumaekawa/dataset-distillation-with-attention-labels).
The preserved starting snapshot of this public repository is commit `b479c34`.
The upstream copyright notice remains in `LICENSE.txt`.

The MIT License covers source code in this repository. It does not relicense
external datasets, pretrained models, model weights, or third-party packages.
Users are responsible for following the terms of:

- GLUE/SST-2 and AG News;
- Hugging Face model and dataset distributions;
- `bert-base-uncased`, `google/bert_uncased_L-6_H-768_A-12`, and
  `prajjwal1/bert-tiny`;
- PyTorch, Transformers, Datasets, Hydra, MLflow, and other dependencies.

Generated poison datasets and distilled checkpoints are intentionally excluded
from the source release. They should be created only for authorized security
research and should be clearly labeled when shared.
