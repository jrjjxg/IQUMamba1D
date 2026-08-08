# ICASSP 2024 RF Challenge checkpoint compatibility source

This directory is the upstream `src` folder from
[`RFChallenge/icassp2024rfchallenge`](https://github.com/RFChallenge/icassp2024rfchallenge),
branch `0.2.0`, commit `ab1d51b`. It is bundled so released PyTorch WaveNet checkpoints that
pickle classes under `src.config_torchwavenet` can be loaded in an offline
Kaggle notebook.

The only compatibility modification is in `src/config_torchwavenet.py`: its
Python 3.7 dataclass instance defaults use `default_factory`, which preserves
their values while allowing the module to import on Kaggle's Python 3.12.
An otherwise empty `src/__init__.py` makes package discovery explicit.

The competition pipeline registers the `rfchallenge` directory as an import
root only while loading legacy checkpoint metadata, so pickle references such
as `src.config_torchwavenet.Config` resolve without Internet access. Inference
still runs IQUMamba's native WaveNet implementation.
