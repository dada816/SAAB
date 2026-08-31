import glob
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed

from data import DataConfig, DataModule
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig, Evaluator
from model import LearnerModel, ModelConfig
from trainer import TrainConfig, Trainer
from utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # add hydra config
            hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
            for file in hydra_config_files:
                mlflow.log_artifact(file)
            with logging_redirect_tqdm():
                out = func(config, *args, **kwargs)
            # add main.log
            mlflow.log_artifact(os.path.join(output_dir, "main.log"))
        return out

    return wrapper


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@mlflow_start_run_with_hydra
def main(config: Config):
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # log config (mlflow)
    log_params_from_omegaconf_dict(config)

    # Set seed
    set_seed(config.base.seed)

    # DataModule
    logger.info(f"Loading datasets: (`{config.data.task_name}`)")
    data_module = DataModule(config.data)

    # Learner
    logger.info(f"Building Leaner model: (`{config.model.model_name}`)")
    model = LearnerModel(config.model, num_labels=data_module.num_labels)

    # preprocess datasets
    data_module.run_preprocess(tokenizer=model.tokenizer)

    # Distilled data
    if config.distilled_data.pretrained_data_path is not None:
        distilled_data = DistilledData.from_pretrained(
            config.distilled_data.pretrained_data_path
        )
    else:
        distilled_data = DistilledData(
            config=config.distilled_data,
            train_config=config.learner_train,
            num_labels=data_module.num_labels,
            hidden_size=model.bert_model_config.hidden_size,
            num_layers=model.bert_model_config.num_hidden_layers,
            num_heads=model.bert_model_config.num_attention_heads,
        )

    # ==========================================================
    # SAAB (Semantic-Adaptive Attention Backdoor) Injection Logic
    # ==========================================================
    attack_strategy = config.distilled_data.get("attack_strategy", "none")

    if attack_strategy == "SAAB":
        logger.info("Attack strategy: Semantic-Adaptive Attention Backdoor (SAAB) enabled!")
        distilled_data.construct_and_freeze_saab_attention(
            trigger_index=config.distilled_data.trigger_index,
            trigger_length=config.distilled_data.trigger_length,
            attention_alpha=config.distilled_data.attention_alpha,
        )

    # Evaluator
    evaluator = Evaluator(config.evaluate, model=model)

    # Train distilled data
    if not config.train.skip_train:
        trainer = Trainer(config.train)

        # ==========================================================
        # Training Logic: Distinguish between SI and DI modes
        # ==========================================================
        # Check for DI mode: attack_weight > 0 implies Dual Data Streams
        attack_weight = config.train.get("attack_weight", 0.0)

        if attack_weight > 0:
            logger.info(f"Detected DI Mode (Attack Weight: {attack_weight})")
            logger.info("Loading Dual Data Streams: 'train_clean' & 'train_poison'")

            # Use the get_dataloader method added in data.py
            loader_clean = data_module.get_dataloader("train_clean", shuffle=True)
            loader_poison = data_module.get_dataloader("train_poison", shuffle=True)

            trainer.fit(
                distilled_data=distilled_data,
                model=model,
                clean_loader=loader_clean,
                poison_loader=loader_poison,
                valid_loader=data_module.valid_loader(),
                evaluator=evaluator,
            )
        else:
            logger.info("Detected Standard/SI Mode (Normal Training)")
            # Normal SI process: data.py maps 'train_mixed' to 'train' automatically
            trainer.fit(
                distilled_data=distilled_data,
                model=model,
                train_loader=data_module.train_loader(),
                valid_loader=data_module.valid_loader(),
                evaluator=evaluator,
            )

    # ==========================================================
    # Evaluation Logic: Evaluate both CTA (Clean) and ASR (Poison)
    # ==========================================================

    # 1. Evaluate CTA (Clean Test Accuracy)
    logger.info(">>> Evaluating CTA (Clean Test Accuracy)...")
    results_cta = evaluator.evaluate(
        distilled_data, eval_loader=data_module.valid_loader(), verbose=True
    )
    # Log CTA
    mlflow.log_metrics({f"cta.avg.{k}": v[0] for k, v in results_cta.items()})

    final_results = {}
    final_results.update({f"cta_{k}": f"{v[0]:.4f}±{v[1]:.4f}" for k, v in results_cta.items()})

    # 2. Evaluate ASR (Attack Success Rate)
    # Check if 'test_poisoned' split exists
    if "test_poisoned" in data_module.preprocessed_datasets:
        logger.info(">>> Evaluating ASR (Attack Success Rate)...")

        # Get poisoned test loader (shuffle=False)
        loader_asr = data_module.get_dataloader("test_poisoned", shuffle=False)

        # Reuse evaluator to re-train learner on distilled data and test on loader_asr
        results_asr = evaluator.evaluate(
            distilled_data, eval_loader=loader_asr, verbose=True
        )

        # Log ASR
        mlflow.log_metrics({f"asr.avg.{k}": v[0] for k, v in results_asr.items()})
        final_results.update({f"asr_{k}": f"{v[0]:.4f}±{v[1]:.4f}" for k, v in results_asr.items()})
    else:
        logger.warning("No 'test_poisoned' split found. Skipping ASR evaluation.")

    logger.info(f"Final Combined Results: {final_results}")

    save_path = os.path.join(config.base.save_dir, "results.json")
    json.dump(final_results, open(save_path, "w"), indent=4)
    mlflow.log_artifact(save_path)

    return


if __name__ == "__main__":
    main()
