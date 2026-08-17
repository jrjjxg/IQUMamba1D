from pathlib import Path
import tempfile

import torch

from util.distributed import DistributedEvalSampler, distributed_sum


def test_distributed_eval_sampler_has_exact_coverage():
    sample_count = 11
    shards = [
        list(DistributedEvalSampler(range(sample_count), 4, rank))
        for rank in range(4)
    ]
    flattened = [index for shard in shards for index in shard]

    assert sorted(flattened) == list(range(sample_count))
    assert len(flattened) == len(set(flattened))


def test_distributed_sum_without_process_group():
    assert distributed_sum([1, 2.5], torch.device("cpu")) == [1.0, 2.5]


def _run_distributed_sum_worker(rank, world_size, init_method):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        reduced = distributed_sum([rank + 1, 1], torch.device("cpu"))
        assert reduced == [3.0, 2.0], reduced
        print(f"rank={rank} reduced={reduced}")
    finally:
        torch.distributed.destroy_process_group()


def run_distributed_sum_smoke():
    world_size = 2
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous_path = Path(temporary_directory) / "gloo_rendezvous"
        torch.multiprocessing.spawn(
            _run_distributed_sum_worker,
            args=(world_size, rendezvous_path.as_uri()),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    test_distributed_eval_sampler_has_exact_coverage()
    run_distributed_sum_smoke()
