import torch
from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler):
    """Shard evaluation data without padding or duplicating samples."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


def distributed_sum(values, device):
    """Sum scalar statistics across ranks and return them on every rank."""

    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return [float(value) for value in values]
    statistics = torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
    return statistics.cpu().tolist()
