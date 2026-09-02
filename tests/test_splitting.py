import torch

from sae_repro.metrics.splitting import discover_splitting_order


def test_first_latent_is_baseline_not_split() -> None:
    """第一个主要 latent 不应被误计为额外 splitting。"""
    target = torch.tensor([True, True, True, True, False, False])
    firing = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [False, False],
            [False, False],
        ]
    )
    report = discover_splitting_order(firing, target, max_latents=2, f1_jump=0.1)
    assert report["latent_order"][0] == 0
    assert report["additional_splitting_count"] == 1
    assert len(report["f1_increments"]) == 2
