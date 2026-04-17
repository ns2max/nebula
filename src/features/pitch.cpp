#include "features/pitch.h"
#include "fft_utils.h"
#include "filterbank.h"
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <numeric>

static float mean_v(const FloatVec& v) {
    if (v.empty()) return 0.f;
    return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
}

static float hz_to_midi(float hz) {
    return 12.f * log2f(hz / 440.f) + 69.f;
}

static float yin_f0(const FloatVec& frame, int sr, float fmin, float fmax) {
    int N       = static_cast<int>(frame.size());
    int tau_min = static_cast<int>(static_cast<float>(sr) / fmax);
    int tau_max = static_cast<int>(static_cast<float>(sr) / fmin);
    tau_max     = std::min(tau_max, N / 2);
    if (tau_min >= tau_max) return 0.f;

    FloatVec d(tau_max + 1, 0.f);
    for (int tau = 1; tau <= tau_max; tau++)
        for (int j = 0; j < N - tau; j++) {
            float diff = frame[j] - frame[j + tau];
            d[tau] += diff * diff;
        }

    FloatVec cmnd(tau_max + 1, 1.f);
    float running_sum = 0.f;
    for (int tau = 1; tau <= tau_max; tau++) {
        running_sum += d[tau];
        cmnd[tau] = (running_sum > 0.f) ? d[tau] * tau / running_sum : 1.f;
    }

    const float threshold = 0.1f;
    int best_tau = -1;
    for (int tau = tau_min; tau <= tau_max - 1; tau++) {
        if (cmnd[tau] < threshold) {
            while (tau + 1 <= tau_max && cmnd[tau + 1] < cmnd[tau]) tau++;
            best_tau = tau; break;
        }
    }
    if (best_tau < 0) {
        best_tau = tau_min;
        for (int tau = tau_min + 1; tau <= tau_max; tau++)
            if (cmnd[tau] < cmnd[best_tau]) best_tau = tau;
        if (cmnd[best_tau] > 0.2f) return 0.f;
    }

    float tau_frac = static_cast<float>(best_tau);
    if (best_tau > 0 && best_tau < tau_max) {
        float a = cmnd[best_tau - 1], b = cmnd[best_tau], c = cmnd[best_tau + 1];
        float denom = 2.f * (a - 2.f * b + c);
        if (fabsf(denom) > 1e-10f) tau_frac += (a - c) / denom;
    }
    return (tau_frac > 0.f) ? static_cast<float>(sr) / tau_frac : 0.f;
}

static void compute_f0_contour(const FloatVec& y, int sr, float fmin, float fmax,
                                int hop_length, FloatVec& f0_out, FloatVec& voiced_out) {
    int N        = static_cast<int>(y.size());
    int n_fft    = 2048;
    int n_frames = 1 + N / hop_length;
    f0_out.resize(n_frames, 0.f);
    voiced_out.resize(n_frames, 0.f);
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length;
        int end   = std::min(start + n_fft, N);
        if (end - start < 64) continue;
        FloatVec frame(y.begin() + start, y.begin() + end);
        frame.resize(n_fft, 0.f);
        float f0 = yin_f0(frame, sr, fmin, fmax);
        f0_out[t]     = f0;
        voiced_out[t] = (f0 > 0.f) ? 1.f : 0.f;
    }
}

// ============================================================
//  Individual feature functions
// ============================================================

F0Result extract_f0_contours(const FloatVec& y, int sr, float fmin, float fmax, int hop_length) {
    F0Result r;
    compute_f0_contour(y, sr, fmin, fmax, hop_length, r.f0, r.voiced);
    int n_frames = static_cast<int>(r.f0.size());
    FloatVec voiced_f0;
    for (int t = 0; t < n_frames; t++)
        if (r.f0[t] > 0.f) voiced_f0.push_back(r.f0[t]);
    r.voiced_ratio = static_cast<float>(voiced_f0.size()) / std::max(n_frames, 1);
    if (!voiced_f0.empty()) {
        float s = 0;
        for (auto v : voiced_f0) s += v;
        r.mean = s / voiced_f0.size();
        float sv = 0;
        for (auto v : voiced_f0) sv += (v - r.mean) * (v - r.mean);
        r.std     = sqrtf(sv / voiced_f0.size());
        r.min_val = *std::min_element(voiced_f0.begin(), voiced_f0.end());
        r.max_val = *std::max_element(voiced_f0.begin(), voiced_f0.end());
    }
    return r;
}

HarmonicRatioResult extract_harmonic_ratios(const FloatVec& y, int sr, int n_fft, int hop_length) {
    FloatVec f0, voiced;
    compute_f0_contour(y, sr, 65.4f, 2093.f, hop_length, f0, voiced);
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins   = static_cast<int>(S.size());
    int n_frames = static_cast<int>(f0.size());

    HarmonicRatioResult r;
    r.ratios.resize(n_frames, 0.f);
    for (int t = 0; t < n_frames; t++) {
        if (f0[t] < 1.f) continue;
        float total = 0, harm = 0;
        for (int k = 0; k < n_bins; k++) total += S[k][t]*S[k][t];
        for (int h = 1; h <= 5; h++) {
            int bin = static_cast<int>(h * f0[t] * n_fft / sr + 0.5f);
            if (bin >= 0 && bin < n_bins) harm += S[bin][t]*S[bin][t];
        }
        r.ratios[t] = (total > 1e-10f) ? harm / total : 0.f;
    }
    float s = 0, s2 = 0; int cnt = 0;
    for (int t = 0; t < n_frames; t++)
        if (voiced[t] > 0) { s += r.ratios[t]; ++cnt; }
    r.mean = cnt > 0 ? s / cnt : 0.f;
    for (int t = 0; t < n_frames; t++)
        if (voiced[t] > 0) s2 += (r.ratios[t] - r.mean)*(r.ratios[t] - r.mean);
    r.std = cnt > 1 ? sqrtf(s2 / cnt) : 0.f;
    return r;
}

FloatVec extract_harmonic_deviation(const FloatVec& y, int sr, int n_fft, int hop_length) {
    FloatVec f0, voiced;
    compute_f0_contour(y, sr, 65.4f, 2093.f, hop_length, f0, voiced);
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins   = static_cast<int>(S.size());
    int n_frames = static_cast<int>(f0.size());
    float df     = static_cast<float>(sr) / n_fft;

    FloatVec deviations;
    int frames_check = std::min(n_frames, 100);
    for (int t = 0; t < frames_check; t++) {
        if (f0[t] < 1.f) continue;
        float frame_max = 0;
        for (int k = 0; k < n_bins; k++) frame_max = std::max(frame_max, S[k][t]);
        float thresh = frame_max * 0.1f;

        FloatVec peak_freqs;
        for (int k = 1; k < n_bins - 1; k++)
            if (S[k][t] > thresh && S[k][t] > S[k-1][t] && S[k][t] > S[k+1][t])
                peak_freqs.push_back(k * df);
        if (peak_freqs.size() < 3) continue;

        int n_harm = std::min(9, static_cast<int>(peak_freqs.size()));
        FloatVec actual_ratios(peak_freqs.size() - 1);
        for (size_t i = 1; i < peak_freqs.size(); i++)
            actual_ratios[i-1] = peak_freqs[i] / f0[t];
        FloatVec expected_ratios(n_harm - 1);
        for (int i = 0; i < n_harm - 1; i++) expected_ratios[i] = static_cast<float>(i + 2);

        int min_len = std::min(static_cast<int>(actual_ratios.size()),
                               static_cast<int>(expected_ratios.size()));
        float dev = 0;
        for (int i = 0; i < min_len; i++) dev += fabsf(actual_ratios[i] - expected_ratios[i]);
        if (min_len > 0) deviations.push_back(dev / min_len);
    }
    return deviations;
}

VibratoResult extract_vibrato(const FloatVec& y, int sr, float fmin, float fmax, int hop_length) {
    FloatVec f0, voiced;
    compute_f0_contour(y, sr, fmin, fmax, hop_length, f0, voiced);
    int N = static_cast<int>(f0.size());
    if (N < 16) return {0.f, 0.f};

    FloatVec smooth(N, 0.f);
    for (int t = 0; t < N; t++) {
        float s = 0.f; int cnt = 0;
        for (int k = std::max(0, t-2); k <= std::min(N-1, t+2); k++) {
            if (voiced[k] > 0) { s += f0[k]; ++cnt; }
        }
        smooth[t] = cnt > 0 ? s / cnt : 0.f;
    }

    int fft_size = 1;
    while (fft_size < N) fft_size <<= 1;
    FloatVec in(fft_size, 0.f);
    std::copy(smooth.begin(), smooth.end(), in.begin());
    int n_bins = fft_size / 2 + 1;

    fftwf_complex* spec = fftwf_alloc_complex(n_bins);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(fft_size, in.data(), spec, FFTW_ESTIMATE);
    fftwf_execute(plan); fftwf_destroy_plan(plan);

    FloatVec mag(n_bins);
    for (int k = 0; k < n_bins; k++) {
        float re = spec[k][0], im = spec[k][1];
        mag[k] = sqrtf(re*re + im*im);
    }
    fftwf_free(spec);

    float frame_rate = static_cast<float>(sr) / hop_length;
    float bin_hz     = frame_rate / fft_size;
    int lo = std::max(1, static_cast<int>(4.f  / bin_hz));
    int hi = std::min(n_bins - 1, static_cast<int>(12.f / bin_hz));
    if (lo >= hi) return {0.f, 0.f};

    int peak_bin = lo;
    for (int k = lo + 1; k <= hi; k++)
        if (mag[k] > mag[peak_bin]) peak_bin = k;

    VibratoResult r;
    r.rate   = peak_bin * bin_hz;
    r.extent = 2.f * mag[peak_bin] / fft_size;
    return r;
}

PitchSalienceResult extract_pitch_salience(const FloatVec& y, int sr,
                                            float fmin, float fmax, int hop_length) {
    FloatVec f0, voiced;
    compute_f0_contour(y, sr, fmin, fmax, hop_length, f0, voiced);
    int n_frames = static_cast<int>(f0.size());

    // Harmonic ratio mean (reuse logic inline for self-contained function)
    int n_fft = 2048;
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins = static_cast<int>(S.size());
    FloatVec hr(n_frames, 0.f);
    for (int t = 0; t < n_frames; t++) {
        if (f0[t] < 1.f) continue;
        float total = 0, harm = 0;
        for (int k = 0; k < n_bins; k++) total += S[k][t]*S[k][t];
        for (int h = 1; h <= 5; h++) {
            int bin = static_cast<int>(h * f0[t] * n_fft / sr + 0.5f);
            if (bin >= 0 && bin < n_bins) harm += S[bin][t]*S[bin][t];
        }
        hr[t] = (total > 1e-10f) ? harm / total : 0.f;
    }
    float hr_mean = 0; int cnt = 0;
    for (int t = 0; t < n_frames; t++) if (voiced[t] > 0) { hr_mean += hr[t]; ++cnt; }
    if (cnt > 0) hr_mean /= cnt;

    FloatVec salience(n_frames);
    for (int t = 0; t < n_frames; t++)
        salience[t] = voiced[t] * hr_mean * voiced[t];

    PitchSalienceResult r;
    r.mean = mean_v(salience);
    float s2 = 0;
    for (auto v : salience) s2 += (v - r.mean)*(v - r.mean);
    r.std = sqrtf(s2 / std::max(n_frames, 1));
    return r;
}

PitchClassResult extract_pitch_class(const FloatVec& y, int sr, int hop_length) {
    int cqt_hop = hop_length * 4, cqt_fft = 4096;
    Float2D Scqt    = stft_magnitude(y, cqt_fft, cqt_hop);
    int ncqt_bins   = static_cast<int>(Scqt.size());
    int ncqt_frames = ncqt_bins > 0 ? static_cast<int>(Scqt[0].size()) : 0;
    float cqt_df    = static_cast<float>(sr) / cqt_fft;

    Float2D chroma(12, FloatVec(ncqt_frames, 0.f));
    for (int k = 1; k < ncqt_bins; k++) {
        float freq = k * cqt_df;
        if (freq < 1.f) continue;
        float midi = hz_to_midi(freq);
        int   pc   = static_cast<int>(fmodf(roundf(midi), 12.f));
        if (pc < 0) pc += 12;
        for (int t = 0; t < ncqt_frames; t++)
            chroma[pc][t] += Scqt[k][t] * Scqt[k][t];
    }
    for (int t = 0; t < ncqt_frames; t++) {
        float s = 0;
        for (int pc = 0; pc < 12; pc++) s += chroma[pc][t];
        if (s > 1e-10f) for (int pc = 0; pc < 12; pc++) chroma[pc][t] /= s;
    }

    PitchClassResult r;
    r.pc_mean.resize(12, 0.f);
    r.pc_std.resize(12, 0.f);
    for (int pc = 0; pc < 12; pc++) {
        float m = mean_v(chroma[pc]);
        r.pc_mean[pc] = m;
        float s2 = 0;
        for (auto v : chroma[pc]) s2 += (v - m)*(v - m);
        r.pc_std[pc] = sqrtf(s2 / std::max(ncqt_frames, 1));
    }

    FloatVec cmean(12);
    for (int pc = 0; pc < 12; pc++) cmean[pc] = r.pc_mean[pc];
    float total = 0;
    for (auto v : cmean) total += v;
    if (total > 1e-10f) for (auto& v : cmean) v /= total;

    r.chroma_mean = mean_v(cmean);
    float entropy = 0;
    for (auto v : cmean)
        if (v > 1e-10f) entropy -= v * log2f(v);
    r.chroma_entropy = entropy;
    return r;
}

// ============================================================
//  Aggregate wrapper
// ============================================================

PitchFeatures extract_pitch(const FloatVec& y, int sr,
                             float fmin, float fmax, int hop_length) {
    PitchFeatures f;

    auto f0r = extract_f0_contours(y, sr, fmin, fmax, hop_length);
    f.f0 = f0r.f0; f.voiced_flags = f0r.voiced;
    f.f0_mean = f0r.mean; f.f0_std = f0r.std;
    f.f0_min  = f0r.min_val; f.f0_max = f0r.max_val;
    f.voiced_ratio = f0r.voiced_ratio;

    auto hrr = extract_harmonic_ratios(y, sr, 2048, hop_length);
    f.harmonic_ratio = hrr.ratios;
    f.harmonic_ratio_mean = hrr.mean; f.harmonic_ratio_std = hrr.std;

    f.harmonic_deviation = extract_harmonic_deviation(y, sr, 2048, hop_length);

    auto vib = extract_vibrato(y, sr, fmin, fmax, hop_length);
    f.vibrato_rate = vib.rate; f.vibrato_extent = vib.extent;

    auto ps = extract_pitch_salience(y, sr, fmin, fmax, hop_length);
    f.pitch_salience_mean = ps.mean; f.pitch_salience_std = ps.std;

    auto pc = extract_pitch_class(y, sr, hop_length);
    f.pitch_class_mean = pc.pc_mean; f.pitch_class_std = pc.pc_std;
    f.chroma_mean_val = pc.chroma_mean; f.chroma_entropy = pc.chroma_entropy;

    return f;
}
