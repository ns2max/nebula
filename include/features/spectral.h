#pragma once
#include "audio_utils.h"

// ---- aggregate result ----
struct SpectralFeatures {
    FloatVec centroid, bandwidth, rolloff, flatness, flux, slope;
    float centroid_mean = 0, centroid_std = 0;
    float bandwidth_mean = 0, rolloff_mean = 0, flatness_mean = 0;
    float flux_mean = 0, flux_std = 0, slope_mean = 0;
    Float2D contrast;
    float hnr = 0;
    FloatVec inharmonicity;
    float    inharmonicity_mean = 0;
    float spectral_peak_freqs_mean = 0, spectral_peak_count_mean = 0;
    float formant_f1 = 0, formant_f2 = 0, formant_f3 = 0;
};

SpectralFeatures extract_spectral(const FloatVec& y, int sr,
                                  int n_fft = 2048, int hop_length = 512);

// ---- individual feature functions ----
struct SpectralCoreResult {
    FloatVec centroid, bandwidth, rolloff, flatness, flux, slope;
    float centroid_mean, centroid_std, bandwidth_mean, rolloff_mean;
    float flatness_mean, flux_mean, flux_std, slope_mean;
};
struct InharmonicityResult { FloatVec values; float mean; };
struct SpectralPeakResult  { float freq_mean, count_mean; };
struct FormantResult       { float f1, f2, f3; };

SpectralCoreResult   extract_spectral_core    (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
Float2D              extract_spectral_contrast(const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
float                extract_hnr              (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
InharmonicityResult  extract_inharmonicity    (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
SpectralPeakResult   extract_spectral_peaks   (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
FormantResult        extract_formants         (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
