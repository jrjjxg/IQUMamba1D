# ICASSP 2024 RF Challenge Reproduction

This package runs the public ICASSP 2024 single-channel RF Challenge protocol
inside the existing IQUMamba PyTorch environment. Normal training, inference,
local scoring, fixed validation, and the augmentation path do not require
TensorFlow or Sionna.

## Released Official WaveNet Baseline

The released PyTorch baseline can be evaluated without retraining.  Pass its
case directory directly to ``--checkpoint``; the loader resolves the official
``weights.pt`` file and the ``{'model': state_dict}`` payload automatically.
The reproduction uses the official ``residual_layers.*`` parameter names and
also retains compatibility with older local ``residual_blocks.*`` checkpoints.
The upstream ``src`` checkpoint classes are bundled under
``rfchallenge/src`` and registered automatically when an old
checkpoint requests them. After uploading the complete IQUMamba directory,
Kaggle does not need Internet access, a second repository, or a custom
``PYTHONPATH`` entry for the official starter code.

Run all eight released checkpoints, print every case's 11 SINR rows, and then
print the official leaderboard-style total and eight-case average with one
command:

```bash
python -m rfchallenge.cli benchmark-baseline-all \
  --data-root /kaggle/input/datasets/lilnb666/iqumamba-data/dataset/dataset \
  --weights-root /kaggle/input/datasets/lilnb666/rfweights/torchmodels \
  --output-dir /kaggle/working/rf_baseline_all \
  --n-per-sinr 100 \
  --seed 100 \
  --batch-size 1 \
  --device cuda
```

The command writes per-case ``metrics.json`` files and
``official_baseline_all_cases_summary.json``. Add ``--save-predictions`` only
when the eight large SOI/bit prediction pairs are also needed.

## OneInAMillion WaveNet-ft reproduction

The most reproducible published improvement is OneInAMillion's **WaveNet-ft**:
start each of the eight tasks from its released official WaveNet model and
continue training with a very small learning rate. The pipeline implements
this as a strict **model-only warm start**. It never restores optimizer, AMP
scaler, scheduler, epoch, or best-loss state embedded in an input checkpoint;
those states are created fresh. A later local ``--resume`` still restores the
complete local training state normally.

The following single command verifies all eight released checkpoints before
training, fine-tunes every task, evaluates each best checkpoint on the same
public TestSet1-style seed, prints all per-SINR/per-case results, and prints the
eight-case MSE/BER Result and Average:

```bash
python -m rfchallenge.cli train-test-all \
  --data-root /kaggle/input/datasets/lilnb666/iqumamba-data/dataset/dataset \
  --model-config config/model_config_icassp_baseline_wavenet.yaml \
  --init-checkpoint-root /kaggle/input/datasets/lilnb666/rfweights/torchmodels \
  --output-dir /kaggle/working/oneinamillion_wavenet_ft \
  --epochs 2 \
  --samples-per-epoch 10000 \
  --batch-size 1 \
  --learning-rate 2e-6 \
  --optimizer adam \
  --loss mse \
  --sinr-mode continuous \
  --validation-per-sinr 100 \
  --validation-samples 1100 \
  --seed 42 \
  --test-seed 100 \
  --test-n-per-sinr 100 \
  --test-batch-size 8 \
  --device cuda
```

With batch size 1 this first pass performs 20,000 fine-tuning updates per
case. Increase the target ``--epochs`` only when the fixed validation score is
still improving; using ``--resume --epochs 4`` continues from epoch 3 rather
than restarting. ``--sinr-mode continuous`` follows the public training range
from approximately -33 dB to +3 dB instead of training only on the 11 test
grid points.

Rerun the identical command with ``--resume`` after an interruption. For a
single task, use ``train`` with ``--soi``, ``--interference``, and
``--init-checkpoint <that case directory>``.

This reproduces the paper's disclosed WaveNet-ft warm start and low learning
rate using the public data. The paper also describes a waveform-plus-bit loss,
but does not disclose enough receiver/loss detail for an exact independent
implementation; this command therefore uses the auditable raw-I/Q MSE variant
and does not claim the hidden TestSet2 score.

## TUB RFDEMUCS Stage 358

Stage 358 implements TUB's waveform-estimation RFDEMUCS. The all-case command
automatically uses `H/S/U=80/4/4` and `LR=3e-4` for QPSK+CommSignal3,
`LR=3e-4` for QPSK+CommSignal2, and `H/S/U=64/2/2`, `LR=3e-5` otherwise.
The paper does not publish batch size or maximum epochs; the values below are
explicit single-GPU reproduction assumptions:

```bash
python -m rfchallenge.cli train-test-all \
  --data-root /kaggle/input/datasets/lilnb666/iqumamba-data/dataset/dataset \
  --model-stage 358 \
  --output-dir /kaggle/working/rfdemucs_stage358 \
  --epochs 100 \
  --samples-per-epoch 10000 \
  --batch-size 4 \
  --optimizer adam \
  --loss mse \
  --sinr-mode continuous \
  --lr-patience 3 \
  --lr-factor 0.1 \
  --minimum-learning-rate 1e-7 \
  --early-stopping-patience 12 \
  --validation-per-sinr 100 \
  --validation-samples 1100 \
  --seed 42 \
  --test-seed 100 \
  --test-n-per-sinr 100 \
  --test-batch-size 4 \
  --device cuda
```

Increase `--samples-per-epoch` to `240000` for the paper's disclosed per-case
data volume. The Stage 358 config deliberately overrides the global CLI
learning rate with the paper's case-specific values and records the effective
value in checkpoint metadata.

## One-command IQUMamba training, resume, and test

`train-test-all` trains all eight SOI/interference cases, selects each case's
best fixed-validation checkpoint, then evaluates the eight best models with
the same 11-SINR MSE/BER table and Result/Average aggregation used by
`benchmark-baseline-all`:

```bash
python -m rfchallenge.cli train-test-all \
  --data-root /kaggle/input/datasets/lilnb666/iqumamba-data/dataset/dataset \
  --model-stage 342 \
  --output-dir /kaggle/working/rf_stage342_seed42 \
  --epochs 150 \
  --samples-per-epoch 10000 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --validation-per-sinr 100 \
  --validation-samples 1100 \
  --seed 42 \
  --test-seed 100 \
  --test-n-per-sinr 100 \
  --test-batch-size 8 \
  --device cuda
```

Each case writes IQUMamba-style artifacts under
``<output-dir>/<SOI>_<Interference>/``:

```text
weights/best_model_weights.pth
weights/best_training_checkpoint.pth
weights/latest_training_checkpoint.pth
checkpoint/best_model_weights.pth
best.pt
last.pt
training_history.json
```

After an interruption, rerun the same command with ``--resume``. The value of
``--epochs`` is the target total, not a number of extra epochs. For example, a
case stopped after epoch 37 resumes at epoch 38 and finishes at epoch 150:

```bash
python -m rfchallenge.cli train-test-all \
  --data-root /kaggle/input/datasets/lilnb666/iqumamba-data/dataset/dataset \
  --model-stage 342 \
  --output-dir /kaggle/working/rf_stage342_seed42 \
  --epochs 150 \
  --samples-per-epoch 10000 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --validation-per-sinr 100 \
  --validation-samples 1100 \
  --seed 42 \
  --test-seed 100 \
  --test-n-per-sinr 100 \
  --test-batch-size 8 \
  --device cuda \
  --resume
```

Resume restores model, optimizer, scheduler, AMP scaler, completed epoch,
best validation loss, training history, early-stopping counter, and Python,
NumPy, CPU/CUDA Torch random states. Completed cases are retained and the
command proceeds through the remaining cases before testing all eight best
models. Test outputs are stored under ``<output-dir>/test_results``; the final
machine-readable aggregate is ``train_test_all_summary.json``.

For example, on Kaggle, first generate a scoreable public TestSet1-style set:

```bash
python -m rfchallenge.cli generate-example \
  --data-root /kaggle/input/datasets/lilnb666/iqumamba-data/dataset/dataset \
  --soi OFDMQPSK \
  --interference CommSignal2 \
  --identifier TestSetLocal_seed100 \
  --output-dir /kaggle/working/rf_eval/seed100 \
  --n-per-sinr 100 \
  --seed 100
```

Run the released model.  The checkpoint argument below is deliberately the
directory shown by the Kaggle dataset browser, not a copied file:

```bash
python -m rfchallenge.cli infer \
  --soi OFDMQPSK \
  --interference CommSignal2 \
  --model-config config/model_config_icassp_baseline_wavenet.yaml \
  --mixture /kaggle/working/rf_eval/seed100/TestSetLocal_seed100_testmixture_OFDMQPSK_CommSignal2.npy \
  --checkpoint /kaggle/input/datasets/lilnb666/rfweights/torchmodels/dataset_ofdmqpsk_commsignal2_mixture_wavenet \
  --identifier TestSetLocal_seed100 \
  --method-id Default_Torch_WaveNet \
  --output-dir /kaggle/working/rf_eval/seed100/outputs \
  --batch-size 100 \
  --device cuda
```

Score the resulting arrays against the locally generated truth:

```bash
python -m rfchallenge.cli evaluate \
  --soi OFDMQPSK \
  --interference CommSignal2 \
  --ground-truth /kaggle/working/rf_eval/seed100/GroundTruth_TestSetLocal_seed100_Dataset_OFDMQPSK_CommSignal2.pkl \
  --metadata /kaggle/working/rf_eval/seed100/TestSetLocal_seed100_testmixture_OFDMQPSK_CommSignal2_metadata.npy \
  --estimated-soi /kaggle/working/rf_eval/seed100/outputs/Default_Torch_WaveNet_TestSetLocal_seed100_estimated_soi_OFDMQPSK_CommSignal2.npy \
  --estimated-bits /kaggle/working/rf_eval/seed100/outputs/Default_Torch_WaveNet_TestSetLocal_seed100_estimated_bits_OFDMQPSK_CommSignal2.npy
```

This evaluates the public protocol against locally generated ground truth. It
does not reproduce the hidden TestSet2 leaderboard score. Prefer the public
``testset1_frame`` recordings for this check; ``interferenceset_frame`` is the
training bank and should not be presented as held-out test data.

## KU-TII-Style Model

`config/model_config_rfchallenge_kutii_wavenet.yaml` is a transparent
implementation from the public KU-TII paper, not a claim of access to its
released code. It implements the disclosed design points:

- 30 residual WaveNet blocks and 256 residual channels;
- same-length, three-tap gated dilated convolutions;
- scalar learnable dilation per block, projected to a valid range after each
  optimizer update;
- per-mixture dilation-cycle selection;
- raw I/Q MSE training, optional learning-rate reduction, and optional early
  stopping.

The public paper does not disclose its exact learnable-dilation
parameterization, cycle values per case, CommSignal2 codec, augmentation seed,
or high-SNR acceptance threshold. The corresponding choices in this package
are explicit command-line settings and should be reported as
**KU-TII-inspired**, not byte-level winner reproduction.

## IQUMamba Models in the Challenge Pipeline

The following single-SOI configurations can be trained with the ordinary
`train` and `train-all` commands. They have a two-channel I/Q input and a
two-channel I/Q SOI head, and the RF Challenge builder also enforces those
settings at runtime. It disables deep supervision because the competition
objective contains one waveform target only.

| Project stage | RF Challenge configuration | Model |
| --- | --- | --- |
| 4 | `config/model_config_stage4_rfchallenge.yaml` | IQUMamba1D |
| 12 | `config/model_config_stage12_rfchallenge.yaml` | IQUBiMamba1D |
| 79 | `config/model_config_stage79_rfchallenge.yaml` | IQUMamba1D + mixture-estimated CycloFRESH |
| 197 | `config/model_config_stage197_rfchallenge.yaml` | IQUBiMamba1D + mixture-estimated CycloFRESH |
| 235 | `config/model_config_stage235_rfchallenge.yaml` | IQUBiMamba1D + compressed global cross-scale attention |
| 255 | `config/model_config_stage255_rfchallenge.yaml` | identity-aware physical MoE |
| 261 | `config/model_config_stage261_rfchallenge.yaml` | WaveNet(10) + stride-4 UniMamba + WaveNet(10) |
| 290 | `config/model_config_stage290_rfchallenge.yaml` | strict-complex C1 stem + real Stage-4 backbone |
| 295 | `config/model_config_stage295_rfchallenge.yaml` | complex-state selective SSM |
| 299 | `config/model_config_stage299_rfchallenge.yaml` | Stage-290 complex stem + Stage-295 complex-state SSM |
| 309 | `config/model_config_stage309_rfchallenge.yaml` | Stage-4 + FDConv1D at encoder stage 0 |
| 310 | `config/model_config_stage310_rfchallenge.yaml` | Stage-4 + UniRepLK1D at encoder stages 0/1/2 |
| 333 | `config/model_config_stage333_rfchallenge.yaml` | combined RF Mamba-3 recurrence |
| 336 | `config/model_config_stage336_rfchallenge.yaml` | Stage-4 + standalone S4D-Lin memory |
| 342 | `config/model_config_stage342_rfchallenge.yaml` | CycloFRESH + complex stem + RF Mamba-3 + UniRepLK |
| 350 | `config/model_config_stage350_rfchallenge.yaml` | CycloFRESH + complex stem + Stage-4 Mamba + UniRepLK |
| 354 | `config/model_config_stage354_rfchallenge.yaml` | strict Stage-333 A2+A3 (trapezoidal + cyclic poles) |
| 355 | `config/model_config_stage355_rfchallenge.yaml` | strict Stage-333 A2+A4 (trapezoidal + reliability) |
| 356 | `config/model_config_stage356_rfchallenge.yaml` | strict Stage-333 A3+A4 (cyclic poles + reliability) |
| 357 | `config/model_config_stage357_rfchallenge.yaml` | fully strict-complex U-Net + strict-complex S4D |
| 358 | `config/model_config_stage358_rfchallenge.yaml` | TUB RFDEMUCS waveform estimator (case-specific H/S/U) |

Stages 79, 197, 342, and 350 estimate cyclic frequency from the received I/Q
mixture; they do not consume hidden SPS, modulation, or transmitter-side
labels. The other registered stages likewise consume only the received
waveform. All configurations therefore use the same public-data pipeline,
fixed TestSet1 validation, official MSE logging, and local MSE/BER evaluator.
`train-sweep` remains reserved for the KU-TII learnable-dilation WaveNet.

For example, train Stage 197 on one public case:

```powershell
python -m rfchallenge.cli train `
  --data-root <official-dataset-directory> `
  --soi QPSK `
  --interference CommSignal2 `
  --model-stage 197 `
  --output-dir results/rfchallenge/stage197_qpsk_commsignal2 `
  --epochs 100 `
  --samples-per-epoch 10000 `
  --batch-size 16 `
  --learning-rate 8e-4 `
  --validation-per-sinr 100 `
  --validation-samples 1100 `
  --device cuda
```

Replace only `--model-stage` and `--output-dir` to train any stage in the
table. `--model-config <path>` remains available for unregistered or modified
YAML files and is mutually exclusive with `--model-stage`. If neither is
given, Stage 255 remains the default. The IQUMamba/Mamba runtime is required;
the challenge package does not install or change that environment.

## Fixed Validation

Training now creates one deterministic TestSet1Example-style validation set
before the first epoch and reuses it unchanged for every epoch. The default is
exactly `11 x 100 = 1,100` examples in ascending official SINR order
`-30, -27, ..., 0 dB`:

- source: `testset1_frame/<interference>_test1_raw_data.h5`;
- SOI, phase, crops, and nominal SINR: deterministic from `--seed + 10000`;
- targets and bits: retained with the validation arrays for auditing;
- sweep candidates: each receives the same fixed validation construction.

Every epoch logs the competition metrics rather than SI-SNR: complex waveform
MSE, the official 11-bin truncated MSE score, mean demodulated BER, and the
official BER threshold score. The protocol-matched QPSK/OFDM-QPSK hard
demodulator uses the ground-truth bits already carried by the fixed validation
dataset. Per-SINR MSE/BER arrays and aggregate values are stored in
`training_history.json` and checkpoint metadata. `validation_loss` remains the
selected raw I/Q training loss used by the scheduler, early stopping, and
best-checkpoint selection; with `--loss mse` it differs from complex MSE only
by the I/Q channel reduction convention.

`--validation-per-sinr` defaults to `100`. `--validation-samples` remains for
command compatibility and must equal `11 * --validation-per-sinr`; leave it
at the default `1100` for the paper-style setup. If TestSet1 raw frames are
absent, the CLI explicitly warns and produces the same fixed stratification
from `InterferenceSet`; that fallback is not held-out validation.

Example case-specific cycle search:

```powershell
python -m rfchallenge.cli train-sweep `
  --data-root <official-dataset-directory> `
  --soi QPSK `
  --interference CommSignal2 `
  --model-config config/model_config_rfchallenge_kutii_wavenet.yaml `
  --dilation-cycle-lengths 8 9 10 11 `
  --epochs 100 `
  --samples-per-epoch 10000 `
  --validation-per-sinr 100 `
  --validation-samples 1100 `
  --lr-patience 4 `
  --early-stopping-patience 12
```

The selection summary is written to
`results/rfchallenge/<SOI>_<interference>/dilation_cycle_selection.json`.
To run the same search sequentially for all eight public cases, replace
`train-sweep` with `train-all-sweep` and omit `--soi` and `--interference`.

The supplied model configuration has `max_dilation: 1024`, so a cycle length
of 12 is invalid because it initializes a dilation of 2048. Increase that
limit explicitly before evaluating a 12-layer cycle.

## CommSignal2 Round-Trip Pairs

The paper says it generated 22,000 CommSignal2 augmentation examples by
converting high-SNR, zero-BER waveforms to bits and then back to waveforms. It
does not publish enough information to decode raw CommSignal2 directly. The
implemented reproducible interpretation is therefore:

```text
known QPSK / OFDM-QPSK SOI + raw CommSignal2 interference
  -> high nominal SINR mixture
  -> protocol-matched hard demodulation of the known SOI
  -> retain BER = 0 samples
  -> remodulate recovered SOI bits into the stored target
  -> store complete (mixture, target, bits, SINR metadata) pairs
```

The builder deliberately reads **InterferenceSet** CommSignal2 raw frames,
not TestSet1, so the augmentation bank does not leak the held-out raw frame
collection. Build a bank with a new output directory:

```powershell
python -m rfchallenge.cli build-commsignal2-roundtrip-pairs `
  --data-root <official-dataset-directory> `
  --soi QPSK `
  --output <working-directory>/commsignal2_qpsk_roundtrip_22000 `
  --num-examples 22000 `
  --candidate-sinr-db 0 3 `
  --batch-size 8 `
  --seed 42
```

`0` and `3 dB` are explicit defaults, not values claimed by the paper. If the
acceptance rate is too low, increase `--max-attempts`, revise
`--candidate-sinr-db`, or use a different seed. The output contains
memory-mappable `.npy` arrays and a `manifest.json`; an incomplete bank is
marked `complete: false` and will not be accepted by training.

At 40,960 samples, the mixture and target arrays alone require about 13.4 GiB
for 22,000 complex64 pairs. The bank is sampled one mini-batch at a time, so
this is disk usage rather than per-worker RAM. QPSK bits add about 0.1 GiB;
OFDM-QPSK bits add about 1.2 GiB.

Mix the completed pair bank into a CommSignal2 training case:

```powershell
python -m rfchallenge.cli train `
  --data-root <official-dataset-directory> `
  --soi QPSK `
  --interference CommSignal2 `
  --model-config config/model_config_rfchallenge_kutii_wavenet.yaml `
  --commsignal2-roundtrip-pairs <working-directory>/commsignal2_qpsk_roundtrip_22000 `
  --commsignal2-roundtrip-probability 0.5 `
  --validation-per-sinr 100 `
  --validation-samples 1100
```

`--commsignal2-roundtrip-probability 0.5` replaces approximately half of the
online examples with complete stored pairs. This interface is intentionally
limited to CommSignal2 cases and validates the SOI type, bit width, and frame
length before a pair is injected.

The old `--commsignal2-augmentation-path` option is still available for an
external HDF5/NPY/NPZ **interference waveform** bank. It only replaces the
interference crop; it is not equivalent to the paired round-trip method.

For a known externally supplied waveform/bit set, the small compatibility
command remains available:

```powershell
python -m rfchallenge.cli roundtrip-augment `
  --soi QPSK `
  --waveforms <high-snr-waveforms.npy> `
  --bits <known-bits.npy> `
  --output <clean-roundtrip-bank.npy>
```

## Compatibility Checks

Run the native checks with:

```powershell
python -m rfchallenge.cli compatibility
```

The suite checks the QPSK/RRC/OFDM protocol, hard-demodulation bit mapping,
starter-equivalent MSE/BER aggregation, KU-TII model hooks, old waveform-bank
augmentation, the fixed `11 x 100` validation layout, directory/portable pair
bank round trips, and complete-pair injection. Use `--official-runtime` only
where the original TensorFlow/Sionna starter environment already exists; it
reports a skip rather than altering the IQUMamba environment.
