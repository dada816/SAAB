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
├── .gitignore             # Generated files and local artifacts
├── README.md              # Documentation
├── LICENSE.txt            # MIT License
└── requirements.txt       # Python dependencies
```

Directories such as `data/`, `logs/`, and `save/` are created during data generation and training.

## Environment Setup

The paper reports experiments with Python 3.10.19, PyTorch 2.0.0/CUDA 11.8, and Transformers 4.28.1. A Python 3.10 environment can be prepared as follows:

```bash
conda create -n saab_env python=3.10
conda activate saab_env
pip install -r requirements.txt
```

## Data Preparation

Generate the SST-2 and AG News poisoned datasets for the trigger and poisoning-ratio combinations supported by the preparation script:

```bash
bash scripts/prepare_all_datasets.sh
```

The script covers SST-2 triggers `cf`, `film`, `movie`, and `the`, and AG News triggers `cf`, `said`, and `the`, at poisoning ratios 0.1%, 0.5%, and 1.0%.

## Reproduction

### 1. SAAB main example on SST-2

Run SAAB on SST-2 with the `film` trigger, one synthetic sample per class, and the default 0.1% poisoning ratio:

```bash
bash scripts/reproduce_main_result.sh
```

The run performs dataset distillation and reports Clean Test Accuracy (CTA) and Attack Success Rate (ASR).

### 2. Baselines and clean references

Run Clean-Std, Clean-Attn, SI, DI-Std, and DI-Attn on the SST-2 `film` setting:

```bash
bash scripts/reproduce_baselines.sh
```

The supplied reproduction scripts are focused entry points for the default SST-2 `film` experiment. Other paper settings require changing Hydra arguments or adding experiment loops; the defense and full sensitivity experiments are not automated by these scripts.

## Custom Usage

Hydra arguments can be overridden from the command line. For example, the following command runs a custom SAAB experiment with BERT-Tiny:

```bash
python src/main.py -m \
    data.task_name=sst2 \
    data.datasets_path="./data/SST2_R0.001_film_Target1" \
    data.preprocessed_datasets_path="./data/SST2_R0.001_film_Target1/preprocessed_bert_tiny" \
    model.model_name="prajjwal1/bert-tiny" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    distilled_data.attack_strategy=SAAB \
    train.attack_weight=1.0 \
    base.method="SAAB_BERT_Tiny"
```

## References

The implementation builds on:

- [PyTorch](https://pytorch.org/) for differentiable optimization.
- [Hugging Face Transformers](https://huggingface.co/) and [Datasets](https://huggingface.co/docs/datasets/) for models, tokenization, datasets, and metrics.
- [Hydra](https://hydra.cc/) for experiment configuration.
- [Dataset Distillation with Attention Labels](https://github.com/arumaekawa/dataset-distillation-with-attention-labels) as the foundational attention-guided text distillation codebase.
