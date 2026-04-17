#pragma once
#include "audio_utils.h"

// ---- aggregate result (all time-domain features in one pass) ----
struct TimeDomainFeatures {
    FloatVec zcr;
    float    zcr_mean = 0, zcr_std = 0, zcr_min = 0;
    FloatVec rms;
    float    rms_mean = 0, rms_std = 0;
    float    tempo_bpm = 0;
    FloatVec tempo_beat_frames;
    FloatVec amplitude_envelope;
    float    env_mean = 0, env_std = 0;
    float    temporal_centroid = 0, temporal_skewness = 0, temporal_kurtosis = 0;
    float    attack_time = 0, decay_time = 0;
    float    periodicity_lag = 0, periodicity_f0 = 0;
    float    modulation_mean = 0, modulation_max = 0;
};

TimeDomainFeatures extract_time_domain(const FloatVec& y, int sr,
                                       int frame_length = 2048,
                                       int hop_length   = 512);

// ---- individual feature functions ----
struct ZCRResult   { FloatVec frames; float mean, std, min_val; };
struct RMSResult   { FloatVec frames; float mean, std; };
struct TempoResult { float bpm; FloatVec beat_frames; };
struct EnvResult   { FloatVec frames; float mean, std; };
struct MomentsResult { float centroid, skewness, kurtosis; };
struct AttackDecayResult { float attack_time, decay_time; };
struct PeriodicityResult { float lag, f0; };
struct ModulationResult  { float mean, max_val; };

ZCRResult         extract_zcr               (const FloatVec& y, int sr, int frame_length=2048, int hop_length=512);
RMSResult         extract_rms               (const FloatVec& y, int sr, int frame_length=2048, int hop_length=512);
TempoResult       extract_tempo_beats       (const FloatVec& y, int sr, int hop_length=512);
EnvResult         extract_amplitude_envelope(const FloatVec& y, int sr, int frame_length=2048, int hop_length=512);
MomentsResult     extract_temporal_moments  (const FloatVec& y, int sr, int frame_length=2048, int hop_length=512);
AttackDecayResult extract_attack_decay      (const FloatVec& y, int sr, int frame_length=2048, int hop_length=512);
PeriodicityResult extract_periodicity       (const FloatVec& y, int sr);
ModulationResult  extract_envelope_modulation(const FloatVec& y, int sr, int frame_length=2048, int hop_length=512);
