#pragma once
#include "audio_utils.h"
#include <complex>

using CplxVec = std::vector<std::complex<float>>;

// ---- windowing ----
FloatVec hann_window(int n);

// ---- STFT ----
// All spectrograms: [freq_bin][time_frame]

// Complex STFT: n_frames vectors of (n_fft/2+1) complex bins
// Frames stored as [time][freq]; transpose if needed.
std::vector<CplxVec> stft_complex(const FloatVec& y, int n_fft, int hop_length);

// Magnitude spectrogram [freq][time]
Float2D stft_magnitude(const FloatVec& y, int n_fft, int hop_length);

// Power spectrogram [freq][time]
Float2D stft_power(const FloatVec& y, int n_fft, int hop_length);

// ---- dB conversions ----
FloatVec amplitude_to_db(const FloatVec& S, float top_db = 80.f);
FloatVec power_to_db(const FloatVec& S, float top_db = 80.f);
Float2D  amplitude_to_db_2d(const Float2D& S);
Float2D  power_to_db_2d(const Float2D& S);

// ---- DCT-II (ortho normalised) ----
FloatVec compute_dct2(const FloatVec& x, int n_out);

// ---- autocorrelation (FFT-based) ----
FloatVec autocorrelate(const FloatVec& y, int max_size = -1);

// ---- analytic signal via Hilbert transform ----
CplxVec  analytic_signal(const FloatVec& y);
FloatVec envelope_signal(const FloatVec& y);  // |analytic_signal|

// ---- temporal delta of coefficient matrix [n_coeff][n_frames] ----
// width=9 matches librosa default (uses 4 neighbors each side)
Float2D compute_delta(const Float2D& C, int width = 9);
