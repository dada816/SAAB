# <p align="center">Semantic-Adaptive Attention Backdoor Attacks Against Text Dataset Distillation</p>

<div align="center">

![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

</div>

---

> **Semantic-Adaptive Attention Backdoor Attacks Against Text Dataset Distillation**  
> Hang Ren, Xin Wang, Yudan Chen, Jin Yang, Tong Yue, Wen Chen, and Junqing Le

> **TL;DR:** We identify distilled attention labels as a structural attack surface and propose **Semantic-Adaptive Attention Backdoor (SAAB)**, which combines saturation-based attention injection with structural freezing to preserve a trigger-centered backdoor under highly compressed text dataset distillation.

> **Abstract:** Text dataset distillation compresses large training corpora into compact synthetic sets, enabling efficient model training and data sharing. However, the structural supervision introduced by attention-guided distillation may also create security risks that remain insufficiently understood. This paper identifies distilled attention labels as a structural attack surface and proposes **Semantic-Adaptive Attention Backdoor (SAAB)**, a backdoor framework for text dataset distillation. SAAB combines saturation-based attention injection with structural freezing to convert the trigger position into a fixed attention constraint during bi-level optimization, thereby preserving the backdoor under extreme compression while retaining clean-task utility. We further formulate the *Semantic Anchoring Hypothesis* to characterize how trigger semantics affect the interaction between malicious and benign objectives: domain-related triggers can couple with task-relevant evidence, whereas high-frequency functional triggers may be represented through a more separated structural signal. Experiments on SST-2 and AG News across multiple triggers, synthetic-data budgets, poisoning ratios, and BERT capacities show that SAAB achieves attack success rates of up to 100% while maintaining competitive clean accuracy. Additional analyses of convergence loss, parameter sensitivity, cross-capacity transfer, and post-hoc defenses indicate that the structurally embedded backdoor remains persistent across the evaluated settings.

This repository contains the implementation of **Semantic-Adaptive Attention Backdoor (SAAB)** and its SI, DI-Std, and DI-Attn baselines for attention-guided text dataset distillation.

## Project Structure

```text
.
├── configs/               # Hydra experiment configuration
├── scripts/               # Dataset preparation and reproduction entry points
├── src/                   # Data, model, distillation, training, and evaluation code
├── tests/                 # Protocol and SAAB unit tests
├── environment.yml        # Conda environment specification
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # MIT License
└── README.md              # Documentation
```

Directories such as `data/`, `logs/`, `save/`, and `mlruns/` are created during data generation and training and are excluded from Git.

## Environment Setup

The paper reports experiments with Python 3.10.19, PyTorch 2.0.0/CUDA 11.8, and Transformers 4.28.1. The recommended environment can be prepared with:

```bash
conda env create -f environment.yml
conda activate saab
```

Alternatively, prepare a Python 3.10 environment and install the dependencies manually:

```bash
conda create -n saab python=3.10
conda activate saab
pip install -r requirements.txt
```

The PyTorch and CUDA builds should be selected according to the local GPU driver when necessary.

## Data Preparation

Generate the clean reference datasets and the default SST-2 and AG News poisoned datasets:

```bash
bash scripts/prepare_all_datasets.sh
```

The script covers SST-2 triggers `cf`, `film`, `movie`, and `the`, and AG News triggers `cf`, `said`, and `the`, at poisoning ratios 0.1%, 0.5%, and 1.0%.

A single dataset can also be generated explicitly:

```bash
python src/generate_poison_data.py \
    --task ag_news \
    --trigger the \
    --ratio 0.001 \
    --target_label 1 \
    --source_scope all \
    --selection_seed 42
```

The default `all` setting selects poison sources from all training classes. The optional `non_target` setting restricts the eligible source pool to examples whose original labels differ from the target label. The optional setting is provided for the source-composition sensitivity analysis and does not change the default SAAB protocol.

## Reproduction

### 1. SAAB main example on SST-2

Run SAAB on SST-2 with the `film` trigger, one synthetic sample per class, and the default 0.1% poisoning ratio:

```bash
bash scripts/reproduce_main_result.sh
```

### 2. Baselines and clean references

Run Clean-Std, Clean-Attn, SI, DI-Std, and DI-Attn on the SST-2 `film` setting:

```bash
bash scripts/reproduce_baselines.sh
```

### 3. Source-composition sensitivity analysis

Run the AG News `the` setting with the default all-class source pool and the optional non-target-only source pool:

```bash
bash scripts/reproduce_source_composition.sh both
```

Either setting can be run independently:

```bash
bash scripts/reproduce_source_composition.sh all
bash scripts/reproduce_source_composition.sh non_target
```

The supplied scripts are focused reproduction entry points. Other paper settings can be run by changing the corresponding Hydra overrides.

## Custom Usage

Hydra arguments can be overridden from the command line. For example:

```bash
python src/main.py -m \
    data.task_name=sst2 \
    data.datasets_path=./data/SST2_R0.001_film_Target1 \
    data.preprocessed_datasets_path=./data/SST2_R0.001_film_Target1/preprocessed_bert \
    model.model_name=bert-base-uncased \
    base.seed=42 \
    base.method=SAAB_custom \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    distilled_data.attack_strategy=SAAB \
    distilled_data.trigger_index=1 \
    distilled_data.trigger_length=1 \
    distilled_data.attention_alpha=20.0 \
    train.attack_weight=1.0
```

The reported prefix-trigger experiments use token index 1, immediately after `[CLS]`, with a trigger length of 1 and an attention saturation value of 20.0.

## References

The implementation builds on:

- [PyTorch](https://pytorch.org/) for differentiable optimization.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) and [Datasets](https://huggingface.co/docs/datasets/) for pretrained models, tokenization, datasets, and evaluation.
- [Hydra](https://hydra.cc/) for experiment configuration.
- [MLflow](https://mlflow.org/) for experiment tracking.
- [Dataset Distillation with Attention Labels](https://github.com/arumaekawa/dataset-distillation-with-attention-labels) as the foundational attention-guided text distillation codebase.
