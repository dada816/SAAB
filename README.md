# <p align="center">Semantic-Adaptive Attention Backdoor Attacks Against Text Dataset Distillation</p>

<div align="center">

![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

</div>

---

> **Semantic-Adaptive Attention Backdoor Attacks Against Text Dataset Distillation**  
> Hang Ren, Xin Wang, Yudan Chen, Jin Yang, Tong Yue, Wen Chen, and Junqing Le

> **TL;DR:** We identify distilled attention labels as a structural attack surface and propose **Semantic-Adaptive Attention Backdoor (SAAB)**, which combines saturation-based attention injection with structural freezing to preserve a trigger-centered backdoor under highly compressed text dataset distillation.

> **Abstract:** Text dataset distillation compresses large corpora into compact synthetic sets for efficient training and data sharing. However, attention-guided distillation introduces structural supervision whose security risks remain insufficiently understood. We identify distilled attention labels as an attack surface and propose **Semantic-Adaptive Attention Backdoor (SAAB)**. SAAB combines saturation-based hard attention injection with structural freezing, converting the trigger-position attention target into a fixed constraint during bilevel optimization. This preserves the backdoor under extreme compression while retaining clean-task utility. We further formulate the *Semantic Anchoring Hypothesis* to explain how trigger semantics shape the interaction between malicious and benign objectives. Domain-related triggers couple with task-relevant evidence, whereas high-frequency functional triggers rely on a more separated structural signal. We evaluate SAAB on SST-2 and AG News across multiple triggers, synthetic-data budgets, poisoning ratios, and BERT capacities. SAAB achieves high attack success rates while maintaining competitive clean accuracy. For domain-related triggers, clean accuracy remains comparable to the unattacked attention-guided reference. For high-frequency functional triggers, SAAB better limits degradation in attack effectiveness than dynamic-injection baselines while keeping clean accuracy close to the reference. Analyses of convergence, parameter sensitivity, model-capacity sensitivity, and post-hoc defenses show that the structurally embedded backdoor remains persistent across the evaluated settings. These findings establish distilled attention supervision as an attack surface and motivate security-aware auditing and sanitization of synthetic-data supply chains.

This repository contains the implementation of SAAB and the SI, DI-Std, and DI-Attn baselines for attention-guided text dataset distillation.

## Project Structure

```text
.
├── configs/               # Hydra experiment configuration
├── scripts/               # Data preparation and reproduction scripts
├── src/                   # Distillation, training, and evaluation code
├── tests/                 # Lightweight protocol and SAAB tests
├── environment.yml        # Conda environment
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # MIT License
└── README.md              # Documentation
```

Generated data, logs, checkpoints, and MLflow records are excluded from Git.

## Environment Setup

The experiments use Python 3.10, PyTorch 2.0.0 with CUDA 11.8, and Transformers 4.28.1.

```bash
conda env create -f environment.yml
conda activate saab
```

## Data Preparation

Prepare the clean and poisoned SST-2 and AG News datasets used by the supplied scripts:

```bash
bash scripts/prepare_all_datasets.sh
```

Generate a single AG News dataset with trigger `the`:

```bash
python src/generate_poison_data.py --task ag_news --trigger the
```

The generator defaults to a poisoning ratio of `0.001`, target label `1`, selection seed `42`, and `source_scope=all`. The auxiliary source-composition setting is generated with:

```bash
python src/generate_poison_data.py \
    --task ag_news \
    --trigger the \
    --source_scope non_target
```

`all` is the default paper-compatible source pool. `non_target` excludes examples already belonging to the target class and is used only for the source-composition sensitivity analysis. Each generated dataset includes `poison_metadata.json` with its selection settings and index hash.

## Reproduction

Run the main SAAB example on SST-2 with the `film` trigger:

```bash
bash scripts/reproduce_main_result.sh
```

Run the clean references and SI, DI-Std, and DI-Attn baselines:

```bash
bash scripts/reproduce_baselines.sh
```

Run both source-composition settings on AG News with the `the` trigger:

```bash
bash scripts/reproduce_source_composition.sh both
```

Use `all` or `non_target` instead of `both` to run one setting only.

The supplied scripts preserve the paper-compatible settings in `configs/default.yaml`: distillation seed `42`, 10 distillation epochs, and five victim-model evaluations. Any mean and standard deviation produced by one run summarize those five victim evaluations of one fixed distilled-data run, not five independent distillation runs. The scripts also preserve the original batching, checkpoint-selection, and separate CTA/ASR evaluation behavior; changing these behaviors defines a different protocol.

Run artifacts are stored under `save/<task>.<model>/<method>/<timestamp>/`. Poison-source provenance is saved automatically in `poison_metadata.json` inside the generated dataset directory.

## Custom Usage

The following command contains only the overrides required for a custom SST-2 SAAB run; all other settings use `configs/default.yaml`.

```bash
python src/main.py -m \
    data.task_name=sst2 \
    data.datasets_path=./data/SST2_R0.001_film_Target1 \
    data.preprocessed_datasets_path=./data/SST2_R0.001_film_Target1/preprocessed_bert \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    distilled_data.attack_strategy=SAAB \
    train.attack_weight=1.0
```

## References

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) and [Datasets](https://huggingface.co/docs/datasets/)
- [Hydra](https://hydra.cc/)
- [MLflow](https://mlflow.org/)
- [Dataset Distillation with Attention Labels](https://github.com/arumaekawa/dataset-distillation-with-attention-labels)

This project is a research derivative of the final reference above, based on the preserved starting snapshot `b479c34`. The upstream copyright notice remains in `LICENSE.txt`. The MIT License covers this repository's source code only; external datasets, pretrained models, and third-party packages remain subject to their own terms. Generated poisoned datasets and distilled checkpoints are intentionally excluded and should be used only for authorized security research.
