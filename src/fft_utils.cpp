#include "fft_utils.h"
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

// ---- window ----

FloatVec hann_window(int n) {
    FloatVec w(n);
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.f - cosf(2.f * M_PI * i / (n - 1)));
    return w;
}

// ---- STFT ----

// Returns [n_frames][n_bins] complex frames; n_bins = n_fft/2+1
// Signal is centre-padded (reflect) by n_fft/2 on each side (librosa default).
std::vector<CplxVec> stft_complex(const FloatVec& y, int n_fft, int hop_length) {
    int n_bins   = n_fft / 2 + 1;
    int pad      = n_fft / 2;
    int n_y      = static_cast<int>(y.size());
    int n_frames = 1 + n_y / hop_length;

    // reflect-pad
    FloatVec padded(pad + n_y + pad, 0.f);
    for (int i = 0; i < pad; i++) {
        int src = pad - 1 - i;
        padded[i] = (src < n_y) ? y[src] : 0.f;
    }
    std::copy(y.begin(), y.end(), padded.begin() + pad);
    for (int i = 0; i < pad; i++) {
        int src = n_y - 1 - i;
        padded[pad + n_y + i] = (src >= 0) ? y[src] : 0.f;
    }

    FloatVec window = hann_window(n_fft);
    FloatVec frame(n_fft);

    fftwf_complex* out  = fftwf_alloc_complex(n_bins);
    fftwf_plan     plan = fftwf_plan_dft_r2c_1d(n_fft, frame.data(), out, FFTW_ESTIMATE);

    std::vector<CplxVec> result(n_frames, CplxVec(n_bins));
    int padded_size = static_cast<int>(padded.size());

    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length;
        for (int i = 0; i < n_fft; i++) {
            int idx  = start + i;
            frame[i] = (idx < padded_size) ? padded[idx] * window[i] : 0.f;
        }
        fftwf_execute(plan);
        for (int k = 0; k < n_bins; k++)
            result[t][k] = {out[k][0], out[k][1]};
    }

    fftwf_destroy_plan(plan);
    fftwf_free(out);
    return result;
}

// [freq_bin][time_frame]
Float2D stft_magnitude(const FloatVec& y, int n_fft, int hop_length) {
    auto cs      = stft_complex(y, n_fft, hop_length);
    int n_frames = static_cast<int>(cs.size());
    int n_bins   = n_fft / 2 + 1;
    Float2D S(n_bins, FloatVec(n_frames));
    for (int t = 0; t < n_frames; t++)
        for (int k = 0; k < n_bins; k++)
            S[k][t] = std::abs(cs[t][k]);
    return S;
}

Float2D stft_power(const FloatVec& y, int n_fft, int hop_length) {
    auto S = stft_magnitude(y, n_fft, hop_length);
    for (auto& row : S)
        for (auto& v : row)
            v = v * v;
    return S;
}

// ---- dB conversions ----

static float vec_max(const FloatVec& v) {
    return *std::max_element(v.begin(), v.end());
}

FloatVec amplitude_to_db(const FloatVec& S, float top_db) {
    float ref = vec_max(S);
    if (ref < 1e-10f) ref = 1e-10f;
    FloatVec out(S.size());
    for (size_t i = 0; i < S.size(); i++)
        out[i] = 20.f * log10f(std::max(S[i], 1e-10f) / ref);
    float max_val = vec_max(out);
    for (auto& v : out)
        v = std::max(v, max_val - top_db);
    return out;
}

FloatVec power_to_db(const FloatVec& S, float top_db) {
    float ref = vec_max(S);
    if (ref < 1e-10f) ref = 1e-10f;
    FloatVec out(S.size());
    for (size_t i = 0; i < S.size(); i++)
        out[i] = 10.f * log10f(std::max(S[i], 1e-10f) / ref);
    float max_val = vec_max(out);
    for (auto& v : out)
        v = std::max(v, max_val - top_db);
    return out;
}

// apply element-wise across a 2D spectrogram [freq][time]
Float2D amplitude_to_db_2d(const Float2D& S) {
    // find global max
    float ref = 1e-10f;
    for (auto& row : S)
        for (auto v : row)
            ref = std::max(ref, v);
    Float2D out(S.size(), FloatVec(S[0].size()));
    for (size_t i = 0; i < S.size(); i++)
        for (size_t j = 0; j < S[i].size(); j++)
            out[i][j] = 20.f * log10f(std::max(S[i][j], 1e-10f) / ref);
    return out;
}

Float2D power_to_db_2d(const Float2D& S) {
    float ref = 1e-10f;
    for (auto& row : S)
        for (auto v : row)
            ref = std::max(ref, v);
    Float2D out(S.size(), FloatVec(S[0].size()));
    for (size_t i = 0; i < S.size(); i++)
        for (size_t j = 0; j < S[i].size(); j++)
            out[i][j] = 10.f * log10f(std::max(S[i][j], 1e-10f) / ref);
    return out;
}

// ---- DCT-II (ortho) ----
// Uses FFTW REDFT10 which computes the unscaled DCT-II.
FloatVec compute_dct2(const FloatVec& x, int n_out) {
    int N = static_cast<int>(x.size());
    FloatVec in(N);
    std::copy(x.begin(), x.end(), in.begin());
    FloatVec out_buf(N);

    fftwf_plan plan = fftwf_plan_r2r_1d(N, in.data(), out_buf.data(),
                                         FFTW_REDFT10, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // ortho normalisation
    float s0 = 1.f / sqrtf(4.f * N);       // k=0
    float sk = 1.f / sqrtf(2.f * N);       // k>0
    FloatVec result(n_out);
    for (int k = 0; k < n_out && k < N; k++)
        result[k] = out_buf[k] * (k == 0 ? s0 : sk);
    return result;
}

// ---- autocorrelation (FFT-based) ----
FloatVec autocorrelate(const FloatVec& y, int max_size) {
    int N = static_cast<int>(y.size());
    if (max_size < 0 || max_size > N) max_size = N;

    // zero-pad to next power of 2 >= 2N for linear autocorrelation
    int fft_size = 1;
    while (fft_size < 2 * N) fft_size <<= 1;

    int n_bins = fft_size / 2 + 1;
    FloatVec in(fft_size, 0.f);
    std::copy(y.begin(), y.end(), in.begin());

    fftwf_complex* spec  = fftwf_alloc_complex(n_bins);
    fftwf_plan     fwd   = fftwf_plan_dft_r2c_1d(fft_size, in.data(), spec, FFTW_ESTIMATE);
    fftwf_execute(fwd);
    fftwf_destroy_plan(fwd);

    // |X|^2
    for (int k = 0; k < n_bins; k++) {
        float re = spec[k][0], im = spec[k][1];
        spec[k][0] = re * re + im * im;
        spec[k][1] = 0.f;
    }

    FloatVec acf(fft_size);
    fftwf_plan inv = fftwf_plan_dft_c2r_1d(fft_size, spec, acf.data(), FFTW_ESTIMATE);
    fftwf_execute(inv);
    fftwf_destroy_plan(inv);
    fftwf_free(spec);

    // normalise and truncate
    float norm = acf[0] > 0 ? 1.f / (acf[0] * fft_size) : 0.f;
    FloatVec result(max_size);
    for (int i = 0; i < max_size; i++)
        result[i] = acf[i] / fft_size * norm * fft_size; // just divide by fft_size
    // simpler: just return raw acf / fft_size, normalise caller-side
    for (int i = 0; i < max_size; i++)
        result[i] = acf[i] / fft_size;
    return result;
}

// ---- Hilbert transform (via FFT) ----
CplxVec analytic_signal(const FloatVec& y) {
    int N     = static_cast<int>(y.size());
    int n_bins = N / 2 + 1;

    FloatVec in(N);
    std::copy(y.begin(), y.end(), in.begin());

    fftwf_complex* spec = fftwf_alloc_complex(n_bins);
    fftwf_plan fwd = fftwf_plan_dft_r2c_1d(N, in.data(), spec, FFTW_ESTIMATE);
    fftwf_execute(fwd);
    fftwf_destroy_plan(fwd);

    // one-sided: multiply positive freqs by 2, keep DC and Nyquist as-is
    // DC
    // Positive freqs [1 .. n_bins-2] * 2
    // Nyquist (n_bins-1): keep as-is for even N
    // Negative freqs (would be N-k) -> set to 0 but r2c doesn't store them

    // We work in full complex domain for IFFT c2c
    fftwf_complex* full = fftwf_alloc_complex(N);
    for (int k = 0; k < N; k++) { full[k][0] = 0.f; full[k][1] = 0.f; }

    full[0][0] = spec[0][0]; full[0][1] = spec[0][1];
    for (int k = 1; k < n_bins - 1; k++) {
        full[k][0] = 2.f * spec[k][0];
        full[k][1] = 2.f * spec[k][1];
    }
    if (N % 2 == 0) {
        full[n_bins - 1][0] = spec[n_bins - 1][0];
        full[n_bins - 1][1] = spec[n_bins - 1][1];
    } else {
        full[n_bins - 1][0] = 2.f * spec[n_bins - 1][0];
        full[n_bins - 1][1] = 2.f * spec[n_bins - 1][1];
    }

    fftwf_free(spec);

    fftwf_complex* out_c = fftwf_alloc_complex(N);
    fftwf_plan inv = fftwf_plan_dft_1d(N, full, out_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(inv);
    fftwf_destroy_plan(inv);
    fftwf_free(full);

    CplxVec result(N);
    float norm = 1.f / N;
    for (int i = 0; i < N; i++)
        result[i] = {out_c[i][0] * norm, out_c[i][1] * norm};
    fftwf_free(out_c);
    return result;
}

FloatVec envelope_signal(const FloatVec& y) {
    auto as = analytic_signal(y);
    FloatVec env(as.size());
    for (size_t i = 0; i < as.size(); i++)
        env[i] = std::abs(as[i]);
    return env;
}

// ---- temporal delta (librosa-compatible) ----
// C: [n_coeff][n_frames]; half = width/2 neighbors each side
Float2D compute_delta(const Float2D& C, int width) {
    if (C.empty() || C[0].empty()) return C;
    int n_coeff  = static_cast<int>(C.size());
    int n_frames = static_cast<int>(C[0].size());
    int half     = width / 2;

    // denominator: 2 * sum_{n=1}^{half} n^2
    float denom = 0.f;
    for (int n = 1; n <= half; n++) denom += static_cast<float>(n * n);
    denom *= 2.f;
    if (denom < 1e-10f) denom = 1.f;

    Float2D D(n_coeff, FloatVec(n_frames, 0.f));
    for (int k = 0; k < n_coeff; k++) {
        for (int t = 0; t < n_frames; t++) {
            float num = 0.f;
            for (int n = 1; n <= half; n++) {
                // edge-pad: clamp index
                int t_fwd = std::min(t + n, n_frames - 1);
                int t_bwd = std::max(t - n, 0);
                num += static_cast<float>(n) * (C[k][t_fwd] - C[k][t_bwd]);
            }
            D[k][t] = num / denom;
        }
    }
    return D;
}
