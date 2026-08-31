import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    from distilled_data import (  # noqa: E402
        DistilledData,
        DistilledDataConfig,
        LearnerTrainConfig,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SAABAttentionTest(unittest.TestCase):
    def make_data(self, **overrides):
        config_values = {
            "data_per_label": 1,
            "attention_label_type": "cls",
            "seq_length": 8,
            "label_type": "soft",
            "attack_strategy": "SAAB",
            "trigger_index": 1,
            "trigger_length": 1,
            "attention_alpha": 20.0,
        }
        config_values.update(overrides)
        return DistilledData(
            config=DistilledDataConfig(**config_values),
            train_config=LearnerTrainConfig(train_step=1, batch_size_per_label=1),
            num_labels=2,
            hidden_size=4,
            num_layers=2,
            num_heads=2,
        )

    def test_default_saab_target_is_saturated_and_frozen(self):
        data = self.make_data()
        data.construct_and_freeze_saab_attention()

        logits = data.attention_labels.data
        probabilities = data.attention_labels[torch.arange(2)]
        self.assertTrue(torch.all(logits[..., 1] == 20.0))
        self.assertTrue(torch.all(logits[..., 0] == -20.0))
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1))))
        self.assertTrue(torch.all(probabilities[..., 1] > 1.0 - 1e-6))
        self.assertFalse(logits.requires_grad)

    def test_multi_token_span_splits_mass_evenly(self):
        data = self.make_data(trigger_length=2)
        data.construct_and_freeze_saab_attention()
        probabilities = data.attention_labels[torch.arange(2)]
        self.assertTrue(
            torch.allclose(
                probabilities[..., 1:3],
                torch.full_like(probabilities[..., 1:3], 0.5),
                atol=1e-6,
            )
        )

    def test_invalid_saab_configuration_fails_fast(self):
        with self.assertRaises(ValueError):
            DistilledDataConfig(
                attention_label_type="none",
                attack_strategy="SAAB",
            )

        data = self.make_data(trigger_index=7, trigger_length=2)
        with self.assertRaises(ValueError):
            data.construct_and_freeze_saab_attention()


if __name__ == "__main__":
    unittest.main()
