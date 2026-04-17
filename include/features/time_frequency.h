#pragma once
#include "audio_utils.h"

// ---- aggregate result ----
struct TimeFrequencyFeatures {
    float   stft_mean = 0, stft_std = 0;
    Float2D mel_spectrogram;
    float   mel_mean = 0, mel_std = 0;
    Float2D bark_spectrogram;
    float   bark_mean = 0, bark_std = 0;
    Float2D erb_spectrogram;
    float   erb_mean = 0, erb_std = 0;
    Float2D cqt;
    float   cqt_mean = 0, cqt_std = 0;
    FloatVec cqt_pitch_classes;
    float n_atoms = 0, freq_mean = 0, freq_std = 0, mag_mean = 0, mag_std = 0;
    float modulation_mean = 0, modulation_std = 0, dominant_rate_hz = 0;
};

TimeFrequencyFeatures extract_time_frequency(const FloatVec& y, int sr,
                                             int n_fft = 2048, int hop_length = 512);

// ---- individual feature functions ----
struct STFTStats   { float mean, std; };
struct CQTResult   { Float2D spectrogram; FloatVec pitch_classes; float mean, std; };
struct PeaksTFResult { float n_atoms, freq_mean, freq_std, mag_mean, mag_std; };
struct ModSpecResult { float mean, std, dominant_rate_hz; };

STFTStats    extract_stft_db           (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
Float2D      extract_mel_spectrogram   (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
Float2D      extract_bark_spectrogram  (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
Float2D      extract_erb_spectrogram   (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
CQTResult    extract_cqt               (const FloatVec& y, int sr, int hop_length=512);
PeaksTFResult extract_peaks_tf         (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
ModSpecResult extract_modulation_spec  (const FloatVec& y, int sr, int n_fft=2048, int hop_length=512);
