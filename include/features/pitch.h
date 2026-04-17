#pragma once
#include "audio_utils.h"

// ---- aggregate result ----
struct PitchFeatures {
    FloatVec f0, voiced_flags;
    float f0_mean = 0, f0_std = 0, f0_min = 0, f0_max = 0, voiced_ratio = 0;
    FloatVec harmonic_ratio;
    float    harmonic_ratio_mean = 0, harmonic_ratio_std = 0;
    FloatVec harmonic_deviation;
    float vibrato_extent = 0, vibrato_rate = 0;
    float pitch_salience_mean = 0, pitch_salience_std = 0;
    FloatVec pitch_class_mean, pitch_class_std;
    float    chroma_mean_val = 0, chroma_entropy = 0;
};

PitchFeatures extract_pitch(const FloatVec& y, int sr,
                            float fmin = 65.4f, float fmax = 2093.f,
                            int hop_length = 512);

// ---- individual feature functions ----
struct F0Result {
    FloatVec f0, voiced;
    float mean, std, min_val, max_val, voiced_ratio;
};
struct HarmonicRatioResult { FloatVec ratios; float mean, std; };
struct VibratoResult       { float rate, extent; };
struct PitchSalienceResult { float mean, std; };
struct PitchClassResult {
    FloatVec pc_mean, pc_std;
    float chroma_mean, chroma_entropy;
};

F0Result            extract_f0_contours    (const FloatVec& y, int sr, float fmin=65.4f, float fmax=2093.f, int hop_length=512);
HarmonicRatioResult extract_harmonic_ratios(const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
FloatVec            extract_harmonic_deviation(const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
VibratoResult       extract_vibrato        (const FloatVec& y, int sr, float fmin=65.4f, float fmax=2093.f, int hop_length=512);
PitchSalienceResult extract_pitch_salience (const FloatVec& y, int sr, float fmin=65.4f, float fmax=2093.f, int hop_length=512);
PitchClassResult    extract_pitch_class    (const FloatVec& y, int sr, int hop_length=512);
