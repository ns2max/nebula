#pragma once
#include "audio_utils.h"

// All filterbanks return [n_filters][n_fft/2+1] weight matrices.

Float2D mel_filterbank(int n_mels, int n_fft, int sr,
                       float fmin = 0.f, float fmax = -1.f);

Float2D bark_filterbank(int n_bands, int n_fft, int sr);

Float2D erb_filterbank(int n_bands, int n_fft, int sr);

Float2D gammatone_filterbank(int n_filters, int n_fft, int sr);

// Apply filterbank to a single spectrum: (n_fft/2+1,) -> (n_filters,)
FloatVec apply_filterbank(const Float2D& fb, const FloatVec& spectrum);

// Apply filterbank to spectrogram [freq][time] -> [n_filters][time]
Float2D  apply_filterbank_frames(const Float2D& fb, const Float2D& S);
