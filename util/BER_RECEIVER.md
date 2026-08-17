# BER receiver

`ber_receiver.py` is the standalone BER evaluator for the IQUMamba synthetic
protocols. It implements the digital baseband chain required after source
separation:

```text
separated I/Q
  -> known-CFO removal
  -> RRC matched filter (alpha=0.35, span=20)
  -> timing-grid search
  -> target-referenced complex gain/carrier tracking
  -> nearest-symbol decision
  -> exact bit-label comparison
```

The reported quantity is explicitly `reference_assisted` BER. The clean target
and its known labels are used to resolve timing and the unavoidable
source-separation phase/gain ambiguity; decisions on the separated waveform
still use the exact protocol constellation. This is appropriate for comparing
separated waveforms, but it is not a blind over-the-air receiver BER. The
private generators do not save a preamble or pilot that could resolve the
absolute PSK phase by itself.

The evaluator automatically excludes `rrc_span // 2` symbols at the beginning
and end of each evaluated stream. Those symbols are filter-start/filter-end
transients in the saved MATLAB waveform rather than complete, observable
symbols. Pass `--guard-symbols 0` only when an input has already removed those
transients. The report records the effective guard per source.

## Quick checks

Run from `IQUMamba1D` or use the paths below from the repository root:

```powershell
python util/ber_receiver.py --dataset 8PSK-H --root data/synthetic --max-files 1
python util/ber_receiver.py --dataset QAM-C --root data/synthetic --max-files 1 --max-frames 1
python util/ber_receiver.py --dataset QPSK+16APSK-B --root data/synthetic --max-files 1
```

The command without `--pred-file` performs the mandatory clean-target audit.
For a separated output, provide the matching target file and an output file:

```powershell
python util/ber_receiver.py `
  --dataset 8PSK-H `
  --root data/synthetic `
  --target-file data/synthetic/8PSK-H/target/2Source_8PSK-H_Dataset_target_10_SNR=-10dB.mat `
  --pred-file outputs/8PSK-H_prediction.npy `
  --max-frames 500
```

Prediction files may be HDF5/classic MAT, NPY, or NPZ. The preferred array
layout is `(frames, 2 * sources, samples)`. The loader also accepts the
MATLAB v7.3 private layout `(2 * sources, samples, frames)`. For HDF5/NPZ
files with a non-standard dataset key, use `--pred-key`.

The JSON report contains file-level and SNR-level counts. BER is aggregated as
`total_bit_errors / total_compared_bits`, not as an unweighted mean of frame
BERs. The effective bit count can be smaller than the sidecar length because
the generator rounds `symbols_per_frame` while the stored 4096-sample frame
length is not always an integer number of symbols. The receiver preserves the
continuous stream order and reports only symbols that are actually present.
Before interpreting a separated result, run the same command without
`--pred-file`; this target-vs-target audit should report zero errors after the
RRC guard.

## Dataset coverage

The registry includes `8PSK-A/B/C/D/E/H`, the other generated 8PSK variants,
`QAM-A/B/C/D/E`, `QPSK+16APSK-A/B`, and aliases such as `QPSK16APSK-A`.
The private MATLAB mappings are reproduced for 8PSK, QPSK, 16APSK, square
QAM, and MATLAB's 128-QAM cross constellation. A target-vs-target result
must be near zero before a model prediction is interpreted.

`RML2016`, `RML2018`, and `TorchSig` are intentionally reported as
`bits_unavailable` with the current repository data. Their loader has IQ
snapshots and modulation/SNR labels but no transmitted bit sequence. A strict
BER cannot be reconstructed from those labels. The `--bit-file` override is
currently for private MATLAB target files; a public-data result would also
need an explicit waveform protocol adapter (sample rate, SPS, frame ordering)
before a sidecar could be trusted.

The compatibility audit script now uses the same receiver:

```powershell
python util/ber_receiver_audit.py --dataset 8PSK-H --synthetic-root data/synthetic
```
