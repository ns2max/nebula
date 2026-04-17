#include "features/time_domain.h"
#include "fft_utils.h"
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <numeric>

static float mean_vec(const FloatVec& v) {
    if (v.empty()) return 0.f;
    return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
}
static float std_vec(const FloatVec& v, float m) {
    if (v.size() < 2) return 0.f;
    float s = 0.f;
    for (auto x : v) s += (x - m) * (x - m);
    return sqrtf(s / v.size());
}

// Shared onset-strength tempo estimator used by extract_tempo_beats
static float compute_tempo(const FloatVec& y, int sr, int hop_length,
                            FloatVec& beat_frames) {
    int N        = static_cast<int>(y.size());
    int n_frames = 1 + N / hop_length;

    FloatVec onset(n_frames, 0.f);
    float prev_rms = 0.f;
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length;
        int end   = std::min(start + hop_length, N);
        float s   = 0.f;
        for (int i = start; i < end; i++) s += y[i] * y[i];
        float rms = sqrtf(s / (end - start));
        onset[t]  = std::max(0.f, rms - prev_rms);
        prev_rms  = rms;
    }

    FloatVec ac = autocorrelate(onset, n_frames);
    float ac0   = ac[0] > 1e-10f ? ac[0] : 1.f;
    for (auto& v : ac) v /= ac0;

    float frame_rate = static_cast<float>(sr) / hop_length;
    int   lag_min    = std::max(1, static_cast<int>(frame_rate * 60.f / 240.f));
    int   lag_max    = std::min(n_frames - 1, static_cast<int>(frame_rate * 60.f / 60.f));

    int best_lag = lag_min; float best_val = -1.f;
    for (int lag = lag_min; lag <= lag_max; lag++)
        if (ac[lag] > best_val) { best_val = ac[lag]; best_lag = lag; }

    float bpm    = best_lag > 0 ? 60.f * frame_rate / best_lag : 120.f;
    int min_dist = std::max(1, best_lag - best_lag / 4);
    beat_frames.clear();
    for (int t = 1; t < n_frames - 1; t++) {
        if (onset[t] >= onset[t-1] && onset[t] >= onset[t+1]) {
            if (beat_frames.empty() ||
                (t - static_cast<int>(beat_frames.back())) >= min_dist)
                beat_frames.push_back(static_cast<float>(t));
        }
    }
    return bpm;
}

// ============================================================
//  Individual feature functions
// ============================================================

ZCRResult extract_zcr(const FloatVec& y, int sr, int frame_length, int hop_length) {
    int N = static_cast<int>(y.size());
    int n_frames = 1 + N / hop_length;
    ZCRResult r;
    r.frames.resize(n_frames, 0.f);
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length, end = std::min(start + frame_length, N);
        int count = 0;
        for (int i = start + 1; i < end; i++)
            if ((y[i] >= 0) != (y[i-1] >= 0)) ++count;
        int len = end - start;
        r.frames[t] = len > 1 ? static_cast<float>(count) / (len - 1) : 0.f;
    }
    r.mean    = mean_vec(r.frames);
    r.std     = std_vec(r.frames, r.mean);
    r.min_val = *std::min_element(r.frames.begin(), r.frames.end());
    return r;
}

RMSResult extract_rms(const FloatVec& y, int sr, int frame_length, int hop_length) {
    int N = static_cast<int>(y.size());
    int n_frames = 1 + N / hop_length;
    RMSResult r;
    r.frames.resize(n_frames, 0.f);
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length, end = std::min(start + frame_length, N);
        float s = 0.f;
        for (int i = start; i < end; i++) s += y[i] * y[i];
        r.frames[t] = sqrtf(s / (end - start));
    }
    r.mean = mean_vec(r.frames);
    r.std  = std_vec(r.frames, r.mean);
    return r;
}

TempoResult extract_tempo_beats(const FloatVec& y, int sr, int hop_length) {
    TempoResult r;
    r.bpm = compute_tempo(y, sr, hop_length, r.beat_frames);
    return r;
}

EnvResult extract_amplitude_envelope(const FloatVec& y, int sr,
                                      int frame_length, int hop_length) {
    int N = static_cast<int>(y.size());
    int n_frames = 1 + N / hop_length;
    FloatVec env = envelope_signal(y);
    EnvResult r;
    r.frames.resize(n_frames);
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length, end = std::min(start + frame_length, N);
        float mx = 0.f;
        for (int i = start; i < end; i++) mx = std::max(mx, env[i]);
        r.frames[t] = mx;
    }
    r.mean = mean_vec(r.frames);
    r.std  = std_vec(r.frames, r.mean);
    return r;
}

MomentsResult extract_temporal_moments(const FloatVec& y, int sr,
                                        int frame_length, int hop_length) {
    auto rr = extract_rms(y, sr, frame_length, hop_length);
    int n_frames = static_cast<int>(rr.frames.size());
    FloatVec times(n_frames);
    for (int t = 0; t < n_frames; t++)
        times[t] = static_cast<float>(t * hop_length) / sr;

    float num = 0.f, den = 0.f;
    for (int t = 0; t < n_frames; t++) { num += rr.frames[t]*times[t]; den += rr.frames[t]; }
    MomentsResult r;
    r.centroid = (den > 0.f) ? num / den : 0.f;

    float t_mean = mean_vec(times);
    r.skewness = 3.f * r.centroid - 2.f * t_mean;

    float t_var = 0.f;
    for (auto t : times) t_var += (t - t_mean)*(t - t_mean);
    t_var /= times.size();
    r.kurtosis = 0.f;
    if (t_var > 1e-10f) {
        float k4 = 0.f;
        for (auto t : times) k4 += powf(t - r.centroid, 4.f);
        r.kurtosis = (k4 / times.size()) / (t_var * t_var);
    }
    return r;
}

AttackDecayResult extract_attack_decay(const FloatVec& y, int sr,
                                        int frame_length, int hop_length) {
    auto rr = extract_rms(y, sr, frame_length, hop_length);
    int n_frames = static_cast<int>(rr.frames.size());
    const int kernel = 10;
    FloatVec smooth(n_frames, 0.f);
    for (int t = 0; t < n_frames; t++) {
        float s = 0.f; int cnt = 0;
        for (int k = t; k < std::min(t+kernel, n_frames); k++, cnt++) s += rr.frames[k];
        smooth[t] = cnt > 0 ? s / cnt : 0.f;
    }
    int peak_idx = static_cast<int>(
        std::max_element(smooth.begin(), smooth.end()) - smooth.begin());
    float thresh = 0.01f * smooth[peak_idx];
    int silence_start = 0;
    for (int t = 0; t < peak_idx; t++)
        if (smooth[t] > thresh) { silence_start = t; break; }

    AttackDecayResult r;
    r.attack_time = static_cast<float>((peak_idx - silence_start) * hop_length) / sr;
    float sustain = 0.5f * smooth[peak_idx];
    int df = 0;
    for (int t = peak_idx + 1; t < n_frames; t++) {
        if (smooth[t] < sustain) break; ++df;
    }
    r.decay_time = static_cast<float>(df * hop_length) / sr;
    return r;
}

PeriodicityResult extract_periodicity(const FloatVec& y, int sr) {
    int N = static_cast<int>(y.size());
    int max_lag = std::min(2 * sr, N / 2);
    FloatVec acf = autocorrelate(y, max_lag);
    float norm = 0.f;
    for (auto v : acf) norm = std::max(norm, fabsf(v));
    if (norm > 1e-10f) for (auto& v : acf) v /= norm;

    PeriodicityResult r{0.f, 0.f};
    float best_val = -1.f; int best_lag = -1;
    for (int lag = 1; lag < max_lag - 1; lag++) {
        if (acf[lag] > 0.5f && acf[lag] >= acf[lag-1] && acf[lag] >= acf[lag+1])
            if (acf[lag] > best_val) { best_val = acf[lag]; best_lag = lag; }
    }
    if (best_lag > 0) { r.lag = static_cast<float>(best_lag); r.f0 = static_cast<float>(sr) / best_lag; }
    return r;
}

ModulationResult extract_envelope_modulation(const FloatVec& y, int sr,
                                              int frame_length, int hop_length) {
    auto env = extract_amplitude_envelope(y, sr, frame_length, hop_length);
    int n_frames = static_cast<int>(env.frames.size());
    FloatVec centered(n_frames);
    for (int t = 0; t < n_frames; t++) centered[t] = env.frames[t] - env.mean;

    int fft_size = 1;
    while (fft_size < n_frames) fft_size <<= 1;
    FloatVec in(fft_size, 0.f);
    std::copy(centered.begin(), centered.end(), in.begin());
    int n_bins = fft_size / 2 + 1;

    fftwf_complex* spec = fftwf_alloc_complex(n_bins);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(fft_size, in.data(), spec, FFTW_ESTIMATE);
    fftwf_execute(plan); fftwf_destroy_plan(plan);

    FloatVec mag(n_bins);
    for (int k = 0; k < n_bins; k++) {
        float re = spec[k][0], im = spec[k][1];
        mag[k] = 2.f * sqrtf(re*re + im*im) / n_frames;
    }
    fftwf_free(spec);

    ModulationResult r;
    r.mean    = mean_vec(mag);
    r.max_val = *std::max_element(mag.begin(), mag.end());
    return r;
}

// ============================================================
//  Aggregate wrapper
// ============================================================

TimeDomainFeatures extract_time_domain(const FloatVec& y, int sr,
                                        int frame_length, int hop_length) {
    TimeDomainFeatures f;

    auto zcr  = extract_zcr(y, sr, frame_length, hop_length);
    f.zcr = zcr.frames; f.zcr_mean = zcr.mean; f.zcr_std = zcr.std; f.zcr_min = zcr.min_val;

    auto rms  = extract_rms(y, sr, frame_length, hop_length);
    f.rms = rms.frames; f.rms_mean = rms.mean; f.rms_std = rms.std;

    auto tmp  = extract_tempo_beats(y, sr, hop_length);
    f.tempo_bpm = tmp.bpm; f.tempo_beat_frames = tmp.beat_frames;

    auto env  = extract_amplitude_envelope(y, sr, frame_length, hop_length);
    f.amplitude_envelope = env.frames; f.env_mean = env.mean; f.env_std = env.std;

    auto mom  = extract_temporal_moments(y, sr, frame_length, hop_length);
    f.temporal_centroid = mom.centroid; f.temporal_skewness = mom.skewness; f.temporal_kurtosis = mom.kurtosis;

    auto ad   = extract_attack_decay(y, sr, frame_length, hop_length);
    f.attack_time = ad.attack_time; f.decay_time = ad.decay_time;

    auto per  = extract_periodicity(y, sr);
    f.periodicity_lag = per.lag; f.periodicity_f0 = per.f0;

    auto mod  = extract_envelope_modulation(y, sr, frame_length, hop_length);
    f.modulation_mean = mod.mean; f.modulation_max = mod.max_val;

    return f;
}
