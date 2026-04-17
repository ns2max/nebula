#pragma once
#include "audio_utils.h"

// ---- aggregate result ----
struct ChromaFeatures {
    Float2D chroma_cqt;
    float   chroma_cqt_mean = 0, chroma_cqt_entropy = 0;
    FloatVec chroma_cqt_pc_mean, chroma_cqt_pc_std;
    Float2D chroma_stft;
    FloatVec chroma_stft_pc_mean;
    float chroma_stft_correlation = 0;
    Float2D tonnetz;
    FloatVec tonnetz_dim_mean, tonnetz_dim_std;
    float    tonnetz_mean_val = 0, tonnetz_spread = 0;
    float key_strength = 0, key_clarity = 0;
    int   key_root = 0;
    bool  key_is_major = true;
    float chord_recognition_mean = 0, chord_changes_detected = 0;
    float tonal_centroid_pc1 = 0, tonal_centroid_pc2 = 0;
    float harmonic_change_mean = 0, harmonic_change_std = 0;
    int   harmonic_change_peaks = 0;
    float harmonic_change_rate = 0;
};

ChromaFeatures extract_chroma(const FloatVec& y, int sr,
                              int n_fft = 2048, int hop_length = 512);

// ---- individual feature functions ----
struct CQTChromaResult {
    Float2D chroma;
    FloatVec pc_mean, pc_std;
    float mean, entropy;
};
struct STFTChromaResult {
    Float2D chroma;
    FloatVec pc_mean;
    float stft_correlation;  // correlation with CQT chroma mean
};
struct TonnetzResult {
    Float2D tonnetz;
    FloatVec dim_mean, dim_std;
    float mean, spread;
};
struct KeyResult   { int root; bool is_major; float strength, clarity; };
struct ChordResult { float recognition_mean, changes_detected; };
struct TonalCentroidResult { float pc1, pc2; };
struct HarmonicChangeResult { float mean, std, rate; int peaks; };

CQTChromaResult      extract_cqt_chroma      (const FloatVec& y, int sr, int hop_length=512);
STFTChromaResult     extract_stft_chroma      (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
TonnetzResult        extract_tonnetz          (const FloatVec& y, int sr, int hop_length=512);
KeyResult            extract_key              (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
ChordResult          extract_chord_templates  (const FloatVec& y, int sr, int hop_length=512);
TonalCentroidResult  extract_tonal_centroid   (const FloatVec& y, int sr, int hop_length=512);
HarmonicChangeResult extract_harmonic_changes (const FloatVec& y, int sr, int hop_length=512);
