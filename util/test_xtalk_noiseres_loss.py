from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for candidate in (WORKSPACE / ".tmp_codex_pydeps", WORKSPACE / ".codex_pydeps", ROOT):
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def test_xtalk_loss_penalizes_cross_source_leakage() -> None:
    import torch

    from util.loss import pit_si_snr_huber_xtalk_loss

    length = 128
    t = torch.linspace(0.0, 1.0, length)
    s1 = torch.stack([torch.sin(2 * torch.pi * 5 * t), torch.cos(2 * torch.pi * 5 * t)], dim=0)
    s2 = torch.stack([torch.sin(2 * torch.pi * 11 * t), torch.cos(2 * torch.pi * 11 * t)], dim=0)
    targets = torch.cat([s1, s2], dim=0).unsqueeze(0)

    clean_outputs = targets.clone()
    leaky_outputs = torch.cat([s1 + 0.45 * s2, s2], dim=0).unsqueeze(0)

    clean_loss = pit_si_snr_huber_xtalk_loss(
        clean_outputs,
        targets,
        xtalk_lambda=1.0,
        alpha=0.0,
        beta=0.0,
    )
    leaky_loss = pit_si_snr_huber_xtalk_loss(
        leaky_outputs,
        targets,
        xtalk_lambda=1.0,
        alpha=0.0,
        beta=0.0,
    )

    assert clean_loss.item() < 1e-4
    assert leaky_loss.item() > clean_loss.item() + 0.05


def test_noiseres_loss_penalizes_source_structured_residual() -> None:
    import torch

    from util.loss import pit_si_snr_huber_xtalk_noiseres_loss

    length = 128
    t = torch.linspace(0.0, 1.0, length)
    s1 = torch.stack([torch.sin(2 * torch.pi * 4 * t), torch.cos(2 * torch.pi * 4 * t)], dim=0)
    s2 = torch.stack([torch.sin(2 * torch.pi * 9 * t), torch.cos(2 * torch.pi * 9 * t)], dim=0)
    targets = torch.cat([s1, s2], dim=0).unsqueeze(0)

    clean_outputs = targets.clone()
    mixture_white_residual = (s1 + s2 + 0.05 * torch.randn_like(s1)).unsqueeze(0)
    mixture_source_residual = (s1 + s2 + 0.5 * s1).unsqueeze(0)

    white_loss = pit_si_snr_huber_xtalk_noiseres_loss(
        clean_outputs,
        targets,
        mixture_white_residual,
        alpha=0.0,
        beta=0.0,
        xtalk_lambda=0.0,
        noiseres_lambda=1.0,
        noiseres_corr_weight=1.0,
        noiseres_whiteness_weight=0.0,
    )
    source_residual_loss = pit_si_snr_huber_xtalk_noiseres_loss(
        clean_outputs,
        targets,
        mixture_source_residual,
        alpha=0.0,
        beta=0.0,
        xtalk_lambda=0.0,
        noiseres_lambda=1.0,
        noiseres_corr_weight=1.0,
        noiseres_whiteness_weight=0.0,
    )

    assert source_residual_loss.item() > white_loss.item() + 0.1


def test_xtalk_noiseres_losses_are_registered_in_main() -> None:
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    loss_text = (ROOT / "util" / "loss.py").read_text(encoding="utf-8")

    for name in (
        "PIT-SI-SNR+Huber+XTALK",
        "PIT-SI-SNR+Huber+XTALK+NOISERES",
    ):
        assert name in main_text

    for token in (
        "pit_si_snr_huber_xtalk_loss",
        "pit_si_snr_huber_xtalk_noiseres_loss",
        "xtalk_lambda",
        "noiseres_lambda",
        "noiseres_corr_weight",
        "noiseres_whiteness_weight",
    ):
        assert token in loss_text or token in main_text


def run_tests() -> int:
    tests = [
        test_xtalk_loss_penalizes_cross_source_leakage,
        test_noiseres_loss_penalizes_source_structured_residual,
        test_xtalk_noiseres_losses_are_registered_in_main,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        else:
            print(f"[PASS] {test.__name__}")
    print(f"Results: {len(tests) - failed} passed, {failed} failed out of {len(tests)}")
    return failed


if __name__ == "__main__":
    raise SystemExit(run_tests())
