#include "features/cepstral.h"
#include "fft_utils.h"
#include "filterbank.h"
#include <cmath>
#include <algorithm>
#include <numeric>

static float mean_v(const FloatVec& v) {
    if (v.empty()) return 0.f;
    return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
}

static FloatVec coeff_means(const Float2D& C) {
    FloatVec m(C.size());
    for (size_t k = 0; k < C.size(); k++) m[k] = mean_v(C[k]);
    return m;
}

static FloatVec rasta_filter(const FloatVec& sig) {
    const float a1 = -0.98f, b0 = 0.0004f, b1 = -0.0004f;
    FloatVec out(sig.size(), 0.f);
    float y_prev = 0.f, x_prev = 0.f;
    for (size_t i = 0; i < sig.size(); i++) {
        float x = sig[i];
        float y = b0*x + b1*x_prev - a1*y_prev;
        out[i] = y; y_prev = y; x_prev = x;
    }
    return out;
}

// ============================================================
//  Individual feature functions
// ============================================================

CepstralResult extract_mfcc(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_mels = 20;
    Float2D S = stft_power(y, n_fft, hop_length);
    int n_frames = S.empty() ? 0 : static_cast<int>(S[0].size());
    Float2D fb  = mel_filterbank(n_mels, n_fft, sr);
    Float2D mel = apply_filterbank_frames(fb, S);

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec lm(n_mels);
        for (int m = 0; m < n_mels; m++) lm[m] = log10f(mel[m][t] + 1e-10f);
        auto c = compute_dct2(lm, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

CepstralResult extract_lfcc(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_mels = 20;
    Float2D S = stft_power(y, n_fft, hop_length);
    int n_bins   = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    int step = std::max(1, n_bins / n_mels);

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec ll(n_mels, 0.f);
        for (int m = 0; m < n_mels; m++) {
            int k = m * step;
            ll[m] = 10.f * log10f((k < n_bins ? S[k][t] : 0.f) + 1e-10f);
        }
        auto c = compute_dct2(ll, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

CepstralResult extract_plp(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_bark = 30;
    Float2D S    = stft_power(y, n_fft, hop_length);
    int n_frames = S.empty() ? 0 : static_cast<int>(S[0].size());
    Float2D fb   = bark_filterbank(n_bark, n_fft, sr);
    Float2D bark = apply_filterbank_frames(fb, S);

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec pw(n_bark);
        for (int m = 0; m < n_bark; m++)
            pw[m] = powf(expf(logf(bark[m][t] + 1e-10f)), 3.f);
        auto c = compute_dct2(pw, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

CepstralResult extract_rasta_plp(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_bark = 30;
    Float2D S    = stft_power(y, n_fft, hop_length);
    int n_frames = S.empty() ? 0 : static_cast<int>(S[0].size());
    Float2D fb   = bark_filterbank(n_bark, n_fft, sr);
    Float2D bark = apply_filterbank_frames(fb, S);

    Float2D rasta_bark(n_bark, FloatVec(n_frames));
    for (int m = 0; m < n_bark; m++) {
        FloatVec lb(n_frames);
        for (int t = 0; t < n_frames; t++) lb[t] = logf(bark[m][t] + 1e-10f);
        auto filt = rasta_filter(lb);
        for (int t = 0; t < n_frames; t++) rasta_bark[m][t] = filt[t];
    }

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec pw(n_bark);
        for (int m = 0; m < n_bark; m++) pw[m] = powf(expf(rasta_bark[m][t]), 3.f);
        auto c = compute_dct2(pw, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

CepstralResult extract_gfcc(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_gt = 40;
    Float2D S    = stft_power(y, n_fft, hop_length);
    int n_frames = S.empty() ? 0 : static_cast<int>(S[0].size());
    Float2D fb   = gammatone_filterbank(n_gt, n_fft, sr);
    Float2D gt   = apply_filterbank_frames(fb, S);

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec lg(n_gt);
        for (int m = 0; m < n_gt; m++) lg[m] = log10f(gt[m][t] + 1e-10f);
        auto c = compute_dct2(lg, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

CepstralResult extract_gtcc(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_gt = 40;
    Float2D S    = stft_magnitude(y, n_fft, hop_length);
    int n_frames = S.empty() ? 0 : static_cast<int>(S[0].size());
    Float2D fb   = gammatone_filterbank(n_gt, n_fft, sr);
    Float2D gt   = apply_filterbank_frames(fb, S);
    for (auto& row : gt) for (auto& v : row) v = powf(v, 0.33f);

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec lg(n_gt);
        for (int m = 0; m < n_gt; m++) lg[m] = log10f(gt[m][t] + 1e-10f);
        auto c = compute_dct2(lg, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

CepstralResult extract_pncc(const FloatVec& y, int sr, int n_mfcc, int n_fft, int hop_length) {
    const int n_gt = 40;
    const float gamma = 0.33f;
    Float2D S    = stft_power(y, n_fft, hop_length);
    int n_frames = S.empty() ? 0 : static_cast<int>(S[0].size());
    Float2D fb   = gammatone_filterbank(n_gt, n_fft, sr);
    Float2D gt   = apply_filterbank_frames(fb, S);

    FloatVec noise_floor(n_gt);
    for (int m = 0; m < n_gt; m++) {
        float s = 0;
        for (int t = 0; t < n_frames; t++) s += gt[m][t];
        noise_floor[m] = 0.01f * (n_frames > 0 ? s / n_frames : 0.f);
    }

    CepstralResult r;
    r.base.assign(n_mfcc, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        FloatVec lp(n_gt);
        for (int m = 0; m < n_gt; m++) {
            float val = powf(std::max(gt[m][t], 0.f), gamma)
                      - powf(noise_floor[m], gamma);
            lp[m] = logf(std::max(val, 0.f) + 1e-10f);
        }
        auto c = compute_dct2(lp, n_mfcc);
        for (int k = 0; k < n_mfcc; k++) r.base[k][t] = c[k];
    }
    r.delta  = compute_delta(r.base);
    r.delta2 = compute_delta(r.delta);
    return r;
}

// ============================================================
//  Aggregate wrapper
// ============================================================

CepstralFeatures extract_cepstral(const FloatVec& y, int sr,
                                   int n_mfcc, int n_fft, int hop_length) {
    CepstralFeatures f;

    auto mfcc = extract_mfcc(y, sr, n_mfcc, n_fft, hop_length);
    f.mfcc = mfcc.base; f.mfcc_delta = mfcc.delta; f.mfcc_delta2 = mfcc.delta2;
    f.mfcc_mean = coeff_means(f.mfcc);
    f.mfcc_delta_mean = coeff_means(f.mfcc_delta);
    f.mfcc_delta2_mean = coeff_means(f.mfcc_delta2);

    auto lfcc = extract_lfcc(y, sr, n_mfcc, n_fft, hop_length);
    f.lfcc = lfcc.base; f.lfcc_delta = lfcc.delta; f.lfcc_delta2 = lfcc.delta2;
    f.lfcc_mean = coeff_means(f.lfcc);

    auto plp = extract_plp(y, sr, n_mfcc, n_fft, hop_length);
    f.plp = plp.base; f.plp_delta = plp.delta; f.plp_delta2 = plp.delta2;
    f.plp_mean = coeff_means(f.plp);

    auto rplp = extract_rasta_plp(y, sr, n_mfcc, n_fft, hop_length);
    f.rasta_plp = rplp.base; f.rasta_plp_delta = rplp.delta; f.rasta_plp_delta2 = rplp.delta2;

    auto gfcc = extract_gfcc(y, sr, n_mfcc, n_fft, hop_length);
    f.gfcc = gfcc.base; f.gfcc_delta = gfcc.delta; f.gfcc_delta2 = gfcc.delta2;
    f.gfcc_mean = coeff_means(f.gfcc);
    f.gfcc_delta_mean = coeff_means(f.gfcc_delta);
    f.gfcc_delta2_mean = coeff_means(f.gfcc_delta2);

    auto gtcc = extract_gtcc(y, sr, n_mfcc, n_fft, hop_length);
    f.gtcc = gtcc.base; f.gtcc_delta = gtcc.delta; f.gtcc_delta2 = gtcc.delta2;

    auto pncc = extract_pncc(y, sr, n_mfcc, n_fft, hop_length);
    f.pncc = pncc.base; f.pncc_delta = pncc.delta; f.pncc_delta2 = pncc.delta2;

    return f;
}
