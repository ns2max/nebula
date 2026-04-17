# Nebula — Audio Feature Extraction Library

C++17 library that extracts a comprehensive set of audio features from a mono/stereo audio file. A benchmark executable measures end-to-end latency for each **individual feature method** independently (e.g. MFCC, RASTA-PLP, CQT chroma, …). Designed for deployment on resource-constrained ARM targets (Raspberry Pi).

---

## Dependencies

| Library | Purpose | Ubuntu/Debian |
|---|---|---|
| [FFTW3](http://www.fftw.org/) (single-precision) | FFT, DCT, STFT | `libfftw3-dev` |
| [libsndfile](http://www.mega-nerd.com/libsndfile/) | Audio file I/O | `libsndfile1-dev` |
| cmake ≥ 3.15 | Build system | `cmake` |
| pkg-config | Library discovery | `pkg-config` |

---

## Build

```bash
# Install dependencies (Raspberry Pi / Debian)
sudo apt-get install libfftw3-dev libsndfile1-dev cmake pkg-config

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Resulting binary
./benchmark
```

The CMakeLists automatically detects ARM targets and adds `-O3 -march=native -mfloat-abi=hard -ffast-math`.

---

## Usage

```
benchmark <audio_file> [n_runs]
```

| Argument | Description | Default |
|---|---|---|
| `audio_file` | Path to any audio format supported by libsndfile (WAV, FLAC, AIFF, OGG, …) | required |
| `n_runs` | Number of timed repetitions per module (one warm-up run is discarded) | `10` |

**Example:**

```bash
./benchmark recording.wav 20
```

**Example output:**

```
Loading: recording.wav
Sample rate : 22050 Hz
Duration    : 3.000 s
Samples     : 66150
Runs        : 10

------------------------------------------------------------------------------------------
  Time-domain features
------------------------------------------------------------------------------------------
zcr                             min=    0.28 ms  mean=    0.30 ms  max=    0.33 ms  std=   0.02 ms
rms                             min=    0.18 ms  mean=    0.19 ms  max=    0.21 ms  std=   0.01 ms
tempo_beats                     min=    0.47 ms  mean=    0.49 ms  max=    0.52 ms  std=   0.02 ms
amplitude_envelope              min=    1.31 ms  mean=    1.34 ms  max=    1.39 ms  std=   0.03 ms
temporal_moments                min=    0.20 ms  mean=    0.21 ms  max=    0.23 ms  std=   0.01 ms
attack_decay                    min=    0.19 ms  mean=    0.20 ms  max=    0.22 ms  std=   0.01 ms
periodicity                     min=    1.82 ms  mean=    1.89 ms  max=    1.97 ms  std=   0.05 ms
envelope_modulation             min=    1.33 ms  mean=    1.36 ms  max=    1.41 ms  std=   0.03 ms

------------------------------------------------------------------------------------------
  Spectral features
------------------------------------------------------------------------------------------
spectral_core                   min=    4.51 ms  mean=    4.62 ms  max=    4.79 ms  std=   0.09 ms
spectral_contrast               min=    4.49 ms  mean=    4.58 ms  max=    4.71 ms  std=   0.07 ms
hnr                             min=   10.31 ms  mean=   10.52 ms  max=   10.77 ms  std=   0.15 ms
inharmonicity                   min=    5.01 ms  mean=    5.13 ms  max=    5.29 ms  std=   0.09 ms
spectral_peaks                  min=    4.55 ms  mean=    4.63 ms  max=    4.76 ms  std=   0.07 ms
formants                        min=    4.65 ms  mean=    4.77 ms  max=    4.91 ms  std=   0.08 ms

------------------------------------------------------------------------------------------
  Time-frequency features
------------------------------------------------------------------------------------------
stft_db                         min=    4.43 ms  mean=    4.52 ms  max=    4.65 ms  std=   0.07 ms
mel_spectrogram                 min=    4.49 ms  mean=    4.58 ms  max=    4.71 ms  std=   0.07 ms
bark_spectrogram                min=    4.51 ms  mean=    4.60 ms  max=    4.73 ms  std=   0.07 ms
erb_spectrogram                 min=    4.49 ms  mean=    4.58 ms  max=    4.71 ms  std=   0.07 ms
cqt                             min=    7.12 ms  mean=    7.28 ms  max=    7.49 ms  std=   0.11 ms
peaks_tf                        min=    4.62 ms  mean=    4.72 ms  max=    4.86 ms  std=   0.08 ms
modulation_spec                 min=    5.89 ms  mean=    6.02 ms  max=    6.19 ms  std=   0.09 ms

------------------------------------------------------------------------------------------
  Cepstral features
------------------------------------------------------------------------------------------
mfcc                            min=    4.62 ms  mean=    4.72 ms  max=    4.85 ms  std=   0.07 ms
lfcc                            min=    4.51 ms  mean=    4.60 ms  max=    4.73 ms  std=   0.07 ms
plp                             min=    4.55 ms  mean=    4.64 ms  max=    4.77 ms  std=   0.07 ms
rasta_plp                       min=    4.57 ms  mean=    4.67 ms  max=    4.80 ms  std=   0.07 ms
gfcc                            min=    4.71 ms  mean=    4.81 ms  max=    4.95 ms  std=   0.08 ms
gtcc                            min=    4.66 ms  mean=    4.76 ms  max=    4.89 ms  std=   0.08 ms
pncc                            min=    4.74 ms  mean=    4.84 ms  max=    4.97 ms  std=   0.08 ms

------------------------------------------------------------------------------------------
  Pitch features
------------------------------------------------------------------------------------------
f0_contours                     min=    4.89 ms  mean=    5.01 ms  max=    5.16 ms  std=   0.08 ms
harmonic_ratios                 min=   14.32 ms  mean=   14.62 ms  max=   14.97 ms  std=   0.21 ms
harmonic_deviation              min=   14.35 ms  mean=   14.65 ms  max=   15.01 ms  std=   0.22 ms
vibrato                         min=    4.91 ms  mean=    5.03 ms  max=    5.18 ms  std=   0.08 ms
pitch_salience                  min=   14.37 ms  mean=   14.68 ms  max=   15.04 ms  std=   0.22 ms
pitch_class                     min=    7.19 ms  mean=    7.35 ms  max=    7.55 ms  std=   0.11 ms

------------------------------------------------------------------------------------------
  Chroma / tonal features
------------------------------------------------------------------------------------------
cqt_chroma                      min=    6.98 ms  mean=    7.13 ms  max=    7.32 ms  std=   0.10 ms
stft_chroma                     min=   11.47 ms  mean=   11.72 ms  max=   12.01 ms  std=   0.17 ms
tonnetz                         min=    7.01 ms  mean=    7.17 ms  max=    7.36 ms  std=   0.10 ms
key                             min=    4.51 ms  mean=    4.61 ms  max=    4.74 ms  std=   0.07 ms
chord_templates                 min=    7.03 ms  mean=    7.19 ms  max=    7.39 ms  std=   0.11 ms
tonal_centroid                  min=    7.04 ms  mean=    7.20 ms  max=    7.40 ms  std=   0.11 ms
harmonic_changes                min=    7.06 ms  mean=    7.22 ms  max=    7.42 ms  std=   0.11 ms

==========================================================================================
  Audio duration: 3.000 s  |  10 runs per feature
==========================================================================================
```

Each feature is timed **independently** — runs alone from raw audio to output. Timings are not cumulative.

---

## Library Structure

```
nebula/
├── CMakeLists.txt
├── main.cpp                    # Benchmark driver
├── include/
│   ├── audio_utils.h           # AudioBuffer type, load/mono helpers
│   ├── fft_utils.h             # STFT, DCT, Hilbert, autocorrelation
│   ├── filterbank.h            # Mel, Bark, ERB, Gammatone filterbanks
│   └── features/
│       ├── time_domain.h
│       ├── spectral.h
│       ├── time_frequency.h
│       ├── cepstral.h
│       ├── pitch.h
│       └── chroma.h
└── src/
    ├── audio_utils.cpp
    ├── fft_utils.cpp
    ├── filterbank.cpp
    └── features/
        ├── time_domain.cpp
        ├── spectral.cpp
        ├── time_frequency.cpp
        ├── cepstral.cpp
        ├── pitch.cpp
        └── chroma.cpp
```

---

## Common Parameters

| Parameter | Value | Description |
|---|---|---|
| `n_fft` | 2048 | FFT window size |
| `hop_length` | 512 | Frame hop size (samples) |
| `frame_length` | 2048 | Analysis frame length |
| Window | Hann | Applied before each FFT |
| Padding | Reflect | Signal padded by `n_fft/2` on each side (librosa-compatible) |

---

## Feature Modules

### 1. Time Domain

Individual functions: `extract_zcr`, `extract_rms`, `extract_tempo_beats`, `extract_amplitude_envelope`, `extract_temporal_moments`, `extract_attack_decay`, `extract_periodicity`, `extract_envelope_modulation`. Aggregate: `extract_time_domain`.

Operates entirely in the sample domain; no FFT except for envelope modulation and the Hilbert transform.

| Feature | Output | Description |
|---|---|---|
| **Zero-Crossing Rate** | per-frame + mean, std, min | Fraction of samples changing sign per frame |
| **RMS Energy** | per-frame + mean, std | `sqrt(mean(x²))` per frame |
| **Tempo / Beat Tracking** | BPM, beat frame indices | Autocorrelation of RMS onset envelope; beats from onset peaks spaced by estimated period |
| **Amplitude Envelope** | per-frame max + mean, std | Hilbert-transform envelope, max over frame |
| **Temporal Centroid** | scalar (seconds) | RMS-energy-weighted centre of mass in time |
| **Temporal Skewness** | scalar | `3·centroid − 2·mean(times)` |
| **Temporal Kurtosis** | scalar | `E[(t−centroid)⁴] / Var(t)²` |
| **Attack Time** | scalar (seconds) | Time from signal onset to RMS peak (smoothed RMS, kernel=10) |
| **Decay Time** | scalar (seconds) | Frames after peak until RMS falls below 50% of peak |
| **Periodicity** | lag (samples), F0 (Hz) | First autocorrelation peak > 0.5 in `[1, min(2·sr, N/2)]` |
| **Envelope Modulation** | mean, max | FFT magnitude of the de-meaned amplitude envelope |

---

### 2. Spectral

Individual functions: `extract_spectral_core`, `extract_spectral_contrast`, `extract_hnr`, `extract_inharmonicity`, `extract_spectral_peaks`, `extract_formants`. Aggregate: `extract_spectral`.

Computed from the magnitude STFT `S[freq][time]`.

| Feature | Output | Description |
|---|---|---|
| **Spectral Centroid** | per-frame + mean, std | `Σ(f·|S(f)|) / Σ|S(f)|` |
| **Spectral Bandwidth** | per-frame + mean | `sqrt(Σ((f−centroid)²·|S(f)|) / Σ|S(f)|)` |
| **Spectral Rolloff** | per-frame + mean | Frequency below which 85% of spectral energy lies |
| **Spectral Flatness** | per-frame + mean | Geometric mean / arithmetic mean of `|S(f)|` |
| **Spectral Flux** | per-frame + mean, std | `Σ(|S(f,t)| − |S(f,t−1)|)²` across frequency bins |
| **Spectral Slope** | per-frame + mean | Linear regression slope of `log10(|S(f)|)` vs frequency |
| **Spectral Contrast** | 7 bands × frames | Peak-to-valley ratio (log) in each of 7 sub-bands (200 Hz base, octave spacing) |
| **HNR** | scalar (dB) | Harmonic energy (harmonics 1–9 of F0) vs total energy, on voiced frames |
| **Inharmonicity** | per-frame + mean | Mean fractional deviation of spectral peaks from ideal harmonic series (first 100 frames) |
| **Spectral Peak Stats** | freq mean, count mean | Per frame: top-5 peaks above 80th-pct threshold averaged across frames; bin count above 90th-pct |
| **Formants F1/F2/F3** | scalar Hz (mean across frames) | LPC order-12 (Hann window, 0.97 pre-emphasis), Levinson-Durbin, spectral envelope peak detection at 512 points, mean of top-3 formants |

---

### 3. Time-Frequency

Individual functions: `extract_stft_db`, `extract_mel_spectrogram`, `extract_bark_spectrogram`, `extract_erb_spectrogram`, `extract_cqt`, `extract_peaks_tf`, `extract_modulation_spec`. Aggregate: `extract_time_frequency`.

Filterbank-based spectrograms and derived statistics.

| Feature | Shape | Parameters |
|---|---|---|
| **STFT (dB)** | mean, std | Amplitude → dB, global ref max |
| **Mel Spectrogram** | `[20][n_frames]` dB | 20 mel bands, `n_fft=2048`, `hop=512` |
| **Bark Spectrogram** | `[25][n_frames]` dB | 25 bands, Traunmuller Bark scale |
| **ERB Spectrogram** | `[30][n_frames]` dB | 30 bands, ERB bandwidth `24.7·(4.37f/1000+1)` |
| **CQT** | `[84][n_frames]` dB | 84 bins, 12 bins/octave, fmin=32.7 Hz; computed as power-summed approximation over `n_fft=4096` STFT |
| **CQT Pitch Classes** | 12 values | Mean energy per pitch class (summed across octaves), L1-normalised |
| **Spectral Peak Stats** | n_atoms, freq/mag mean+std | Per-frame: bins above 90th-percentile threshold |
| **Modulation Spectrogram** | mean, std, dominant rate (Hz) | Log-mel (40 bands) → per-band STFT (`n_fft=128, hop=4`) → mean modulation spectrum |

> **CQT note:** The implementation uses a fast approximation: a 4096-point STFT is computed once, then for each CQT bin the power in the corresponding frequency window `[fc − bw/2, fc + bw/2]` (where `bw = fc / Q`, `Q ≈ 17`) is summed. This is significantly cheaper than a true variable-window CQT while preserving the log-frequency resolution needed for chroma and octave-based analysis.

---

### 4. Cepstral

Individual functions: `extract_mfcc`, `extract_lfcc`, `extract_plp`, `extract_rasta_plp`, `extract_gfcc`, `extract_gtcc`, `extract_pncc` — each returns `CepstralResult{base, delta, delta2}`. Aggregate: `extract_cepstral`.

All cepstral features output `[13][n_frames]` coefficient matrices. Each type also provides first-order delta and second-order delta-delta matrices of the same shape. Per-coefficient means (size-13 vectors) are returned for MFCC and GFCC (base + delta + delta-delta).

Delta computation matches librosa `feature.delta(width=9)`: for each coefficient `k` at frame `t`:
```
delta[k][t] = Σ_{n=1}^{4} n·(C[k][t+n] − C[k][t−n]) / (2·Σ_{n=1}^{4} n²)
```
Edge frames are replicated (not zero-padded).

| Feature | Filterbank | Pipeline |
|---|---|---|
| **MFCC** | 20 Mel bands | Power spec → mel FB → `log10` → DCT-II (ortho) |
| **LFCC** | Linear (20 bins) | Power spec → linear FB → `10·log10` → DCT-II |
| **PLP** | 30 Bark bands | Power spec → bark FB → `log` → `exp(·)³` (loudness law) → DCT-II |
| **RASTA-PLP** | 30 Bark bands | Same as PLP but log-power per band filtered by RASTA IIR `b=[0.0004,−0.0004], a=[1,−0.98]` before DCT |
| **GFCC** | 40 Gammatone | Power spec → gammatone FB → `log10` → DCT-II |
| **GTCC** | 40 Gammatone | Magnitude spec → gammatone FB → `(·)^0.33` → `log10` → DCT-II |
| **PNCC** | 40 Gammatone | Power spec → gammatone FB → power normalisation `(p^γ − noise^γ)`, γ=0.33 → `log` → DCT-II; noise floor = 1% of mean band power |

Each of the 7 types above also produces `*_delta` and `*_delta2` fields of identical shape.

DCT-II uses FFTW's `FFTW_REDFT10` with orthonormal scaling (`1/√(4N)` for k=0, `1/√(2N)` for k>0).

---

### 5. Pitch

Individual functions: `extract_f0_contours`, `extract_harmonic_ratios`, `extract_harmonic_deviation`, `extract_vibrato`, `extract_pitch_salience`, `extract_pitch_class`. Aggregate: `extract_pitch`.

| Feature | Output | Description |
|---|---|---|
| **F0 contour** | per-frame (Hz), 0=unvoiced | YIN algorithm: difference function → cumulative mean normalised difference → first dip < 0.1, parabolic interpolation |
| **Voiced/unvoiced** | per-frame flag | 1 if F0 > 0 |
| **F0 statistics** | mean, std, min, max, voiced ratio | Computed on voiced frames only |
| **Harmonic Energy Ratio** | per-frame + mean, std | Energy in harmonics 1–5 / total frame energy |
| **Harmonic Spectral Deviation** | per-frame | Mean absolute deviation of detected spectral peaks from expected harmonics (nearest-neighbour matching, first 100 frames) |
| **Vibrato Rate** | scalar (Hz) | Dominant frequency of smoothed F0 in 4–12 Hz band via FFT |
| **Vibrato Extent** | scalar (Hz) | Peak-to-peak amplitude / 2 of the vibrato component |
| **Pitch Salience** | mean, std | `voiced_flag × harmonic_ratio_mean` per frame |
| **Pitch Class Profile** | mean + std × 12 | CQT chroma (4096-pt STFT, 4× hop) per semitone, L1-normalised |
| **Chroma Mean / Entropy** | scalar each | Mean and `−Σ p·log2(p)` of normalised mean pitch class vector |

**YIN vs PYIN:** The Python reference uses `librosa.pyin` (probabilistic YIN with HMM smoothing). This implementation uses deterministic YIN, which is faster and suitable for real-time use but may produce more octave errors on noisy or polyphonic audio.

F0 search range: `fmin=65.4 Hz` (C2) – `fmax=2093 Hz` (C7).

---

### 6. Chroma / Tonal

Individual functions: `extract_cqt_chroma`, `extract_stft_chroma`, `extract_tonnetz`, `extract_key`, `extract_chord_templates`, `extract_tonal_centroid`, `extract_harmonic_changes`. Aggregate: `extract_chroma`.

| Feature | Shape / Output | Description |
|---|---|---|
| **CQT Chroma** | `[12][n_frames]` | MIDI-bin power accumulation from 4096-pt STFT at 2× hop, L1-normalised per frame |
| **CQT Chroma PC Stats** | mean + std × 12 | Per-pitch-class mean and std across time |
| **Chroma CQT Mean / Entropy** | scalar each | Global mean and `−Σ p·log2(p)` of normalised mean vector |
| **STFT Chroma** | `[12][n_frames]` | Same mapping from standard `n_fft=2048` STFT |
| **STFT Chroma PC Mean** | 12 scalars | Per-pitch-class mean for STFT chroma |
| **Chroma STFT Correlation** | scalar | Pearson correlation between mean CQT and mean STFT chroma vectors |
| **Tonnetz** | `[6][n_frames]` | 6 tonal centroids: sin/cos projections onto perfect-fifth, minor-third, major-third cycles |
| **Tonnetz Per-Dim Stats** | mean + std × 6 | Per-dimension mean and std |
| **Tonnetz Mean / Spread** | scalar each | Global mean and std across all 6 dimensions |
| **Key Detection** | root (0–11), major/minor, strength, clarity | Krumhansl-Schmuckler: Pearson correlation of mean STFT chroma against all 24 key profiles; clarity = max − mean correlation |
| **Chord Template Matching** | recognition mean, changes detected | Cosine similarity of every 4th chroma frame against 12 major triad templates; unique rounded-value count as change estimate |
| **Tonal Centroid** | pc1, pc2 | 2-D cos/sin projection of mean CQT chroma onto circle of fifths |
| **Harmonic Change Rate** | mean, std, peak count, peaks/s | Frame-to-frame L1 CQT-chroma distance normalised by chroma energy; peaks at 1.5× mean threshold |

Key root encoding: 0=C, 1=C♯, 2=D, …, 11=B.

---

## Using the Library in Your Own Code

Link against `libnebula.a` and include the relevant headers. Call individual feature functions directly — no need to run a full module:

```cpp
#include "audio_utils.h"
#include "features/cepstral.h"
#include "features/chroma.h"

AudioBuffer buf = to_mono(load_audio("file.wav"));
const FloatVec& y = buf.samples;
int sr = buf.sample_rate;

// Individual function — only runs MFCC pipeline
CepstralResult mfcc = extract_mfcc(y, sr);
// mfcc.base[coeff][frame], mfcc.delta[coeff][frame], mfcc.delta2[coeff][frame]

// Individual function — only runs CQT chroma pipeline
CQTChromaResult chroma = extract_cqt_chroma(y, sr);
// chroma.chroma[pitch_class][frame], chroma.pc_mean[pitch_class]

// Aggregate — runs all cepstral methods (MFCC, LFCC, PLP, RASTA-PLP, GFCC, GTCC, PNCC)
CepstralFeatures all = extract_cepstral(y, sr);
```

Each `extract_*` function is stateless and thread-safe (no global state beyond FFTW plans created and destroyed per call).

---

## Internal DSP Utilities

### `fft_utils`

| Function | Description |
|---|---|
| `stft_magnitude(y, n_fft, hop)` | Returns `[freq][time]` magnitude spectrogram |
| `stft_power(y, n_fft, hop)` | Returns `[freq][time]` power spectrogram |
| `stft_complex(y, n_fft, hop)` | Returns `[time][freq]` complex frames |
| `amplitude_to_db_2d(S)` | Element-wise 20·log10, ref=global max, 80 dB floor |
| `power_to_db_2d(S)` | Element-wise 10·log10, ref=global max, 80 dB floor |
| `compute_dct2(x, n_out)` | DCT-II with orthonormal scaling via FFTW REDFT10 |
| `autocorrelate(y, max_size)` | FFT-based linear autocorrelation |
| `analytic_signal(y)` | Hilbert transform via one-sided FFT doubling |
| `envelope_signal(y)` | `|analytic_signal(y)|` |
| `compute_delta(C, width=9)` | Temporal derivative of `[n_coeff][n_frames]` matrix; edge-replicated, matches `librosa.feature.delta` |

### `filterbank`

All filterbanks return `[n_filters][n_fft/2+1]` weight matrices.

| Function | Scale | Bands |
|---|---|---|
| `mel_filterbank` | HTK Mel (`2595·log10(1+f/700)`) | Triangular overlapping filters |
| `bark_filterbank` | Traunmuller Bark | Rectangular filters |
| `erb_filterbank` | ERB rate | Rectangular filters centred at log-spaced freqs |
| `gammatone_filterbank` | ERB (rectangular approx) | Same as ERB, alias used for GFCC/GTCC/PNCC |

`apply_filterbank_frames(fb, S)` multiplies `fb [n_filters × n_bins]` by `S [n_bins × n_frames]` → `[n_filters × n_frames]`.

---



## Performance Tips for Raspberry Pi

- Build with `-DCMAKE_BUILD_TYPE=Release` (enables `-O3 -march=native`).
- If running repeatedly on the same FFT sizes, consider creating persistent FFTW plans with `FFTW_MEASURE` instead of `FFTW_ESTIMATE`, and caching them. The current implementation creates/destroys plans per call, which is safe but sub-optimal for repeated calls.
- Call individual functions (`extract_mfcc`, `extract_cqt_chroma`, …) rather than aggregate wrappers when only one method is needed — avoids recomputing shared spectrograms.
- Costliest individual methods: `hnr` and `harmonic_ratios`/`harmonic_deviation`/`pitch_salience` (each internally runs YIN F0 + STFT). `stft_chroma` also calls `extract_cqt_chroma` internally for correlation.
- Multi-channel files are automatically downmixed to mono before any processing.
