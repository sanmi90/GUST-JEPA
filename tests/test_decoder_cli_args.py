"""Smoke tests for ``scripts/session9_train_decoder.py`` argument parsing.

The training entrypoint accumulated a non-trivial number of flags during
Session 10 (decoder type, decoder loss, FiLM toggle, lambda weights, FFL
schedule, encoder-run vs jepa-checkpoint resolution). These tests
exercise the argparse layer end-to-end without running training.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "session9_train_decoder.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("session9_train_decoder", SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec.loader.exec_module(mod)
    return mod


def _parse(argv):
    mod = _load_script_module()
    old = sys.argv
    sys.argv = ["session9_train_decoder.py"] + argv
    try:
        return mod.parse_args()
    finally:
        sys.argv = old


def test_session9_baseline_args_still_parse() -> None:
    """Session 9 / E0-style invocation still parses (legacy --recon-loss-type)."""
    args = _parse([
        "--jepa-checkpoint", "/tmp/dummy.pt",
        "--output-dir", "/tmp/out",
        "--recon-loss-type", "mse",
    ])
    assert args.jepa_checkpoint == "/tmp/dummy.pt"
    assert args.recon_loss_type == "mse"
    assert args.decoder_type == "fukami"


def test_e1_lapfilm_no_ffl_args_parse() -> None:
    """Session 10 E1 launch command parses cleanly."""
    args = _parse([
        "--omega-pipeline-manifest", "outputs/data_pipeline/v1/manifest.json",
        "--encoder-run", "/tmp/run",
        "--decoder-type", "lapfilm",
        "--decoder-upsample", "pixelshuffle",
        "--decoder-use-film", "true",
        "--decoder-loss", "region_pyr_ffl",
        "--lambda-region", "1.0",
        "--lambda-pyramid", "0.4",
        "--lambda-ffl", "0.0",
        "--lambda-enstrophy", "0.02",
        "--lambda-circulation", "0.01",
        "--max-iters", "20000",
        "--B", "16", "--T", "32", "--seed", "42",
        "--output-dir", "/tmp/E1",
    ])
    assert args.decoder_type == "lapfilm"
    assert args.decoder_loss == "region_pyr_ffl"
    assert args.decoder_use_film is True
    assert args.lambda_ffl == 0.0
    assert args.encoder_run == "/tmp/run"


def test_e_nofilm_ablation_args_parse() -> None:
    """E_noFiLM ablation: use_film false. Parsing must accept the string form."""
    args = _parse([
        "--encoder-run", "/tmp/run",
        "--output-dir", "/tmp/E_nofilm",
        "--decoder-type", "lapfilm",
        "--decoder-use-film", "false",
        "--decoder-loss", "region_pyr_ffl",
    ])
    assert args.decoder_use_film is False


def test_e4_coordmlp_args_parse() -> None:
    """Session 10 E4 CoordMLP audit launch command parses cleanly."""
    args = _parse([
        "--omega-pipeline-manifest", "outputs/data_pipeline/v1/manifest.json",
        "--encoder-run", "/tmp/run",
        "--decoder-type", "coord_mlp",
        "--decoder-fourier-bands", "8",
        "--decoder-loss", "region_pyr_ffl",
        "--lambda-region", "1.0",
        "--lambda-pyramid", "0.0",
        "--lambda-ffl", "0.03",
        "--max-iters", "20000",
        "--B", "16", "--T", "32", "--seed", "42",
        "--output-dir", "/tmp/E4",
    ])
    assert args.decoder_type == "coord_mlp"
    assert args.decoder_fourier_bands == 8
    assert args.lambda_pyramid == 0.0


def test_jepa_checkpoint_or_encoder_run_optional_at_argparse_level() -> None:
    """Session 11 Track 0.1 relaxed the mutex group from required=True to
    required=False so omega_direct can run without an encoder checkpoint.
    The latent-mode requirement is now enforced inside main() with a
    clear message; argparse itself no longer rejects."""
    args = _parse(["--output-dir", "/tmp/out"])
    assert args.jepa_checkpoint is None
    assert args.encoder_run is None
    assert args.input_mode == "latent"


def test_input_mode_omega_direct_defaults() -> None:
    """--input-mode omega_direct parses with no encoder source flags."""
    args = _parse([
        "--input-mode", "omega_direct",
        "--decoder-type", "lapfilm",
        "--output-dir", "/tmp/out",
    ])
    assert args.input_mode == "omega_direct"
    assert args.jepa_checkpoint is None
    assert args.encoder_run is None


def test_jepa_checkpoint_and_encoder_run_mutex() -> None:
    """The two encoder-source flags are mutually exclusive."""
    with pytest.raises(SystemExit):
        _parse([
            "--jepa-checkpoint", "/tmp/dummy.pt",
            "--encoder-run", "/tmp/run",
            "--output-dir", "/tmp/out",
        ])


def test_decoder_cond_other_modes_raise_not_implemented() -> None:
    """--decoder-cond params or params_phase must not be acceptable in
    Session 10 (deferred to Session 11)."""
    args = _parse([
        "--encoder-run", "/tmp/run",
        "--output-dir", "/tmp/out",
        "--decoder-cond", "params",
    ])
    assert args.decoder_cond == "params"


def test_ffl_warmup_factor_schedule() -> None:
    """ffl_warmup_factor: 0 before warmup, ramps linearly, then 1 after."""
    mod = _load_script_module()
    f = mod.ffl_warmup_factor
    assert f(0, 2000, 1000) == 0.0
    assert f(1999, 2000, 1000) == 0.0
    assert abs(f(2500, 2000, 1000) - 0.5) < 1e-6
    assert f(3000, 2000, 1000) == 1.0
    assert f(10000, 2000, 1000) == 1.0


def test_build_decoder_dispatch() -> None:
    """build_decoder returns the right class for each decoder-type."""
    mod = _load_script_module()
    import torch

    args = _parse([
        "--encoder-run", "/tmp/run",
        "--output-dir", "/tmp/out",
        "--decoder-type", "fukami",
    ])
    dec = mod.build_decoder(args, latent_dim=32, device=torch.device("cpu"))
    from src.models.decoder import HybridViTConvDecoder
    assert isinstance(dec, HybridViTConvDecoder)

    args = _parse([
        "--encoder-run", "/tmp/run",
        "--output-dir", "/tmp/out",
        "--decoder-type", "lapfilm",
        "--decoder-base-ch", "32",
    ])
    dec = mod.build_decoder(args, latent_dim=32, device=torch.device("cpu"))
    from src.models.lap_film_decoder import LapFiLMDecoder
    assert isinstance(dec, LapFiLMDecoder)
    # base-ch propagated to the channels tuple
    assert dec.channels[0] == 32

    args = _parse([
        "--encoder-run", "/tmp/run",
        "--output-dir", "/tmp/out",
        "--decoder-type", "coord_mlp",
        "--decoder-mlp-hidden", "64",
    ])
    dec = mod.build_decoder(args, latent_dim=32, device=torch.device("cpu"))
    from src.models.coord_mlp_decoder import CoordMLPDecoder
    assert isinstance(dec, CoordMLPDecoder)
    assert dec.hidden == 64
