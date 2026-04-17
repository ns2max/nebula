#include "filterbank.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// ---- Mel scale helpers ----
static float hz_to_mel(float hz) {
    return 2595.f * log10f(1.f + hz / 700.f);
}
static float mel_to_hz(float mel) {
    return 700.f * (powf(10.f, mel / 2595.f) - 1.f);
}

Float2D mel_filterbank(int n_mels, int n_fft, int sr,
                       float fmin, float fmax) {
    if (fmax < 0) fmax = sr / 2.f;
    int n_bins = n_fft / 2 + 1;

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    // n_mels+2 linearly spaced mel points
    FloatVec mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        mel_pts[i] = mel_min + i * (mel_max - mel_min) / (n_mels + 1);

    // convert to Hz then to FFT bin indices
    FloatVec bin_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        bin_pts[i] = floorf((n_fft + 1) * mel_to_hz(mel_pts[i]) / sr);

    Float2D fb(n_mels, FloatVec(n_bins, 0.f));
    for (int m = 0; m < n_mels; m++) {
        float lo  = bin_pts[m];
        float ctr = bin_pts[m + 1];
        float hi  = bin_pts[m + 2];
        for (int k = 0; k < n_bins; k++) {
            if (k >= lo && k <= ctr && ctr > lo)
                fb[m][k] = (k - lo) / (ctr - lo);
            else if (k > ctr && k <= hi && hi > ctr)
                fb[m][k] = (hi - k) / (hi - ctr);
        }
    }
    return fb;
}

// ---- Bark scale (Traunmuller approximation) ----
static float hz_to_bark(float hz) {
    return 26.81f / (1.f + 1960.f / hz) - 0.53f;
}

Float2D bark_filterbank(int n_bands, int n_fft, int sr) {
    int n_bins = n_fft / 2 + 1;
    float fmax = sr / 2.f;

    float bark_min = hz_to_bark(20.f);
    float bark_max = hz_to_bark(fmax);

    FloatVec edges(n_bands + 1);
    for (int i = 0; i <= n_bands; i++)
        edges[i] = bark_min + i * (bark_max - bark_min) / n_bands;

    // Convert bark edges back to Hz bin indices
    // Inverse: hz = 1960 / (26.81/(z+0.53) - 1)
    auto bark_to_hz = [](float z) {
        float denom = 26.81f / (z + 0.53f) - 1.f;
        if (denom <= 0) return 1.f;
        return 1960.f / denom;
    };

    Float2D fb(n_bands, FloatVec(n_bins, 0.f));
    for (int m = 0; m < n_bands; m++) {
        float hz_lo  = bark_to_hz(edges[m]);
        float hz_hi  = bark_to_hz(edges[m + 1]);
        int   bin_lo = static_cast<int>(floorf(hz_lo * n_fft / sr));
        int   bin_hi = static_cast<int>(ceilf(hz_hi  * n_fft / sr));
        bin_lo = std::max(0, bin_lo);
        bin_hi = std::min(n_bins - 1, bin_hi);
        for (int k = bin_lo; k <= bin_hi; k++)
            fb[m][k] = 1.f;
    }
    return fb;
}

// ---- ERB filterbank ----
// ERB bandwidth: erb_bw = 24.7 * (4.37 * f_hz/1000 + 1)
// Centres log-spaced from 50 Hz to sr/2

Float2D erb_filterbank(int n_bands, int n_fft, int sr) {
    int n_bins = n_fft / 2 + 1;
    float fmin = 50.f;
    float fmax = sr / 2.f;

    // Log-spaced centre frequencies
    FloatVec centres(n_bands);
    float log_min = logf(fmin), log_max = logf(fmax);
    for (int i = 0; i < n_bands; i++)
        centres[i] = expf(log_min + i * (log_max - log_min) / (n_bands - 1));

    Float2D fb(n_bands, FloatVec(n_bins, 0.f));
    float df = static_cast<float>(sr) / n_fft;

    for (int m = 0; m < n_bands; m++) {
        float cf     = centres[m];
        float erb_bw = 24.7f * (4.37f * cf / 1000.f + 1.f);
        float hz_lo  = cf - erb_bw / 2.f;
        float hz_hi  = cf + erb_bw / 2.f;
        for (int k = 0; k < n_bins; k++) {
            float f = k * df;
            if (f >= hz_lo && f <= hz_hi)
                fb[m][k] = 1.f;
        }
        // ensure at least one bin active
        if (std::all_of(fb[m].begin(), fb[m].end(), [](float v){ return v == 0; })) {
            int ctr_bin = static_cast<int>(cf / df + 0.5f);
            ctr_bin = std::clamp(ctr_bin, 0, n_bins - 1);
            fb[m][ctr_bin] = 1.f;
        }
    }
    return fb;
}

// ---- Gammatone filterbank (rectangular approx) ----
Float2D gammatone_filterbank(int n_filters, int n_fft, int sr) {
    // identical structure to ERB but used independently
    return erb_filterbank(n_filters, n_fft, sr);
}

// ---- application helpers ----

FloatVec apply_filterbank(const Float2D& fb, const FloatVec& spectrum) {
    int n_filters = static_cast<int>(fb.size());
    int n_bins    = static_cast<int>(spectrum.size());
    FloatVec out(n_filters, 0.f);
    for (int m = 0; m < n_filters; m++) {
        float s = 0.f;
        int   bins = std::min(n_bins, static_cast<int>(fb[m].size()));
        for (int k = 0; k < bins; k++)
            s += fb[m][k] * spectrum[k];
        out[m] = s;
    }
    return out;
}

// S: [freq_bin][time_frame]  ->  out: [n_filters][time_frame]
Float2D apply_filterbank_frames(const Float2D& fb, const Float2D& S) {
    if (S.empty()) return {};
    int n_filters = static_cast<int>(fb.size());
    int n_frames  = static_cast<int>(S[0].size());
    int n_bins    = static_cast<int>(S.size());

    Float2D out(n_filters, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < n_filters; m++) {
            float s = 0.f;
            int bins = std::min(n_bins, static_cast<int>(fb[m].size()));
            for (int k = 0; k < bins; k++)
                s += fb[m][k] * S[k][t];
            out[m][t] = s;
        }
    }
    return out;
}
