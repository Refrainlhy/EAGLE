import math
import os
import pickle
import torch
from tqdm import tqdm

from utils import get_data_transductive, MYNegEdgeSampler

class DatasetProcessor:
    def __init__(self, dataset_name: str, batch_size: int = 200, device: str = "cpu"):
        self.full_data, self.train_data, self.val_data, self.test_data, _, _ = get_data_transductive(
            dataset_name, use_validation=True
        )

        self._init_samplers(device)
        
        self.batch_size = batch_size
        self.device = device

    def _init_samplers(self, device):
        self.train_sampler = MYNegEdgeSampler(
            destinations=self.train_data.destinations,
            full_destinations=self.train_data.destinations,
            num_neg=1,
            device=device,
            seed=2024
        )
        
        self.test_sampler = MYNegEdgeSampler(
            destinations=self.test_data.destinations,
            full_destinations=self.full_data.destinations,
            num_neg=99,
            device=device,
            seed=2026
        )

    def process_test_data(self, cache_dir: str = "../data/batchneg"):
        test_data = self.test_data
        num_batches = math.ceil(len(test_data.sources) / self.batch_size)
        
        for batch_idx in tqdm(range(num_batches)):
            cache_path = os.path.join(
                cache_dir,
                f"{self.dataset_name}/Test_neg99_bs{self.batch_size}_batch{batch_idx}.pkl"
            )

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    yield pickle.load(f)
                continue

            batch_data = self._generate_test_batch(batch_idx)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(batch_data, f)
            
            yield batch_data

    def _generate_test_batch(self, batch_idx: int):
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, len(self.test_data.sources))

        sources = self.test_data.sources[start:end]
        destinations = self.test_data.destinations[start:end]
        timestamps = self.test_data.timestamps[start:end]

        neg_samples = self.test_sampler.sample(destinations)

        if isinstance(neg_samples, torch.Tensor):
            neg_samples = neg_samples.cpu().numpy()
        
        return {
            "sources": sources,
            "destinations": destinations,
            "timestamps": timestamps,
            "negatives": neg_samples #  [batch_size, 99]
        }


if __name__ == "__main__":
    processor = DatasetProcessor(dataset_name="wikipedia")
    # Support dataset_name in ["Contacts", "lastfm", "wikipedia", "reddit", "askubuntu", "superuser", "wikitalk"]

    test_generator = processor.process_test_data()

    for batch_idx, test_batch in enumerate(test_generator):
        print(f"Batch {batch_idx}:")
        print(f"\t#Pos Sample: {len(test_batch['sources'])}")
        print(f"\t#Neg Sample: {test_batch['negatives'].shape}") # (batch_size, 99)
