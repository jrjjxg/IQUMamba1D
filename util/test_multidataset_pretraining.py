import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_loader.dataloader import (
    FixedLengthSignalDataset,
    create_multidataset_data_loaders,
)


class _SignalDataset(Dataset):
    def __init__(self, size, length, marker=0.0):
        self.size = int(size)
        self.length = int(length)
        self.marker = float(marker)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        mixture = torch.full((2, self.length), self.marker)
        target = torch.full((4, self.length), self.marker)
        return mixture, target, torch.tensor(0.0)


class MultiDatasetPretrainingTests(unittest.TestCase):
    def test_strict_policy_rejects_incompatible_length(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            FixedLengthSignalDataset(_SignalDataset(2, 128), 4096, policy="strict")

    def test_crop_and_explicit_padding_preserve_tuple_contract(self):
        cropped = FixedLengthSignalDataset(_SignalDataset(2, 8192), 4096, policy="crop")
        padded = FixedLengthSignalDataset(_SignalDataset(2, 1024), 4096, policy="pad_crop")
        self.assertEqual(cropped[0][0].shape, (2, 4096))
        self.assertEqual(cropped[0][1].shape, (4, 4096))
        self.assertEqual(padded[0][0].shape, (2, 4096))
        self.assertEqual(padded[0][1].shape, (4, 4096))

    def test_balanced_sampling_normalizes_each_dataset_by_its_size(self):
        def fake_create(*, data_choice, batch_size, **_kwargs):
            size = 10 if data_choice == "domain-a" else 30
            marker = 1.0 if data_choice == "domain-a" else 2.0
            train = _SignalDataset(size, 32, marker)
            val = _SignalDataset(2 if data_choice == "domain-a" else 4, 32, marker)
            test = _SignalDataset(2, 32, marker)
            return (
                DataLoader(train, batch_size=batch_size),
                DataLoader(val, batch_size=batch_size),
                {0.0: DataLoader(test, batch_size=batch_size)},
            )

        with patch("data_loader.dataloader.create_data_loaders", side_effect=fake_create):
            train_loader, val_loader, snr_loaders = create_multidataset_data_loaders(
                batch_size=4,
                data_choices=["domain-a", "domain-b"],
                target_length=32,
                sampling="balanced",
                num_workers=0,
                pin_memory=False,
            )

        self.assertIsInstance(train_loader.sampler, WeightedRandomSampler)
        weights = train_loader.sampler.weights
        self.assertAlmostEqual(float(weights[:10].sum()), float(weights[10:].sum()))
        self.assertEqual(len(train_loader.dataset), 40)
        self.assertEqual(len(val_loader.dataset), 8)
        self.assertEqual(len(snr_loaders[0.0].dataset), 4)

    def test_dataset_weights_must_match_dataset_count(self):
        with self.assertRaisesRegex(ValueError, "one value per"):
            create_multidataset_data_loaders(
                batch_size=2,
                data_choices=["a", "b"],
                dataset_weights=[1.0],
            )

    def test_joint_and_single_domain_use_the_same_split_seed(self):
        observed = []

        def fake_create(*, data_choice, batch_size, seed, **_kwargs):
            observed.append((data_choice, seed))
            dataset = _SignalDataset(4, 32)
            loader = DataLoader(dataset, batch_size=batch_size)
            return loader, loader, {0.0: loader}

        with patch("data_loader.dataloader.create_data_loaders", side_effect=fake_create):
            create_multidataset_data_loaders(
                batch_size=2,
                data_choices=["domain-a", "domain-b"],
                target_length=32,
                seed=42,
                num_workers=0,
                pin_memory=False,
            )

        self.assertEqual(observed, [("domain-a", 42), ("domain-b", 42)])


if __name__ == "__main__":
    unittest.main()
