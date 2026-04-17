#include "features/chroma.h"
#include "fft_utils.h"
#include <cmath>
#include <algorithm>
#include <numeric>

static float mean_v(const FloatVec& v) {
    if (v.empty()) return 0.f;
    return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
}
static float std_v(const FloatVec& v, float m) {
    if (v.size() < 2) return 0.f;
    float s = 0;
    for (auto x : v) s += (x - m)*(x - m);
    return sqrtf(s / v.size());
}
static float pearson(const FloatVec& a, const FloatVec& b) {
    int n = static_cast<int>(std::min(a.size(), b.size()));
    if (n < 2) return 0.f;
    float ma = mean_v(a), mb = mean_v(b);
    float num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) {
        num += (a[i] - ma)*(b[i] - mb);
        da  += (a[i] - ma)*(a[i] - ma);
        db  += (b[i] - mb)*(b[i] - mb);
    }
    float denom = sqrtf(da * db);
    return (denom > 1e-10f) ? num / denom : 0.f;
}
static float hz_to_midi(float hz) { return 12.f * log2f(hz / 440.f) + 69.f; }

static Float2D chroma_from_S(const Float2D& S, int n_fft, int sr) {
    int n_bins   = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    float df     = static_cast<float>(sr) / n_fft;
    Float2D chroma(12, FloatVec(n_frames, 0.f));
    for (int k = 1; k < n_bins; k++) {
        float freq = k * df;
        if (freq < 1.f) continue;
        int pc = static_cast<int>(fmodf(roundf(hz_to_midi(freq)), 12.f));
        if (pc < 0) pc += 12;
        for (int t = 0; t < n_frames; t++) chroma[pc][t] += S[k][t]*S[k][t];
    }
    for (int t = 0; t < n_frames; t++) {
        float s = 0;
        for (int pc = 0; pc < 12; pc++) s += chroma[pc][t];
        if (s > 1e-10f) for (int pc = 0; pc < 12; pc++) chroma[pc][t] /= s;
    }
    return chroma;
}

// ============================================================
//  Individual feature functions
// ============================================================

CQTChromaResult extract_cqt_chroma(const FloatVec& y, int sr, int hop_length) {
    int cqt_fft = 4096, cqt_hop = hop_length * 2;
    Float2D S = stft_magnitude(y, cqt_fft, cqt_hop);
    Float2D chroma = chroma_from_S(S, cqt_fft, sr);
    int n_frames = chroma.empty() ? 0 : static_cast<int>(chroma[0].size());

    CQTChromaResult r;
    r.chroma = chroma;
    r.pc_mean.resize(12); r.pc_std.resize(12);
    for (int pc = 0; pc < 12; pc++) {
        float m = mean_v(chroma[pc]);
        r.pc_mean[pc] = m;
        r.pc_std[pc]  = std_v(chroma[pc], m);
    }

    FloatVec cmean(12);
    float total = 0;
    for (int pc = 0; pc < 12; pc++) { cmean[pc] = r.pc_mean[pc]; total += cmean[pc]; }
    if (total > 1e-10f) for (auto& v : cmean) v /= total;

    r.mean = mean_v(cmean);
    float entropy = 0;
    for (auto v : cmean) if (v > 1e-10f) entropy -= v * log2f(v);
    r.entropy = entropy;
    return r;
}

STFTChromaResult extract_stft_chroma(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S      = stft_magnitude(y, n_fft, hop_length);
    Float2D chroma = chroma_from_S(S, n_fft, sr);

    STFTChromaResult r;
    r.chroma = chroma;
    r.pc_mean.resize(12);
    for (int pc = 0; pc < 12; pc++) r.pc_mean[pc] = mean_v(chroma[pc]);

    // Compute CQT chroma mean for correlation
    auto cqt = extract_cqt_chroma(y, sr, hop_length);
    FloatVec cqt_norm = cqt.pc_mean;
    float t1 = 0;
    for (auto v : cqt_norm) t1 += v;
    if (t1 > 1e-10f) for (auto& v : cqt_norm) v /= t1;

    FloatVec stft_norm = r.pc_mean;
    float t2 = 0;
    for (auto v : stft_norm) t2 += v;
    if (t2 > 1e-10f) for (auto& v : stft_norm) v /= t2;

    r.stft_correlation = pearson(cqt_norm, stft_norm);
    return r;
}

TonnetzResult extract_tonnetz(const FloatVec& y, int sr, int hop_length) {
    auto cqt = extract_cqt_chroma(y, sr, hop_length);
    Float2D& chroma  = cqt.chroma;
    int n_frames_cqt = chroma.empty() ? 0 : static_cast<int>(chroma[0].size());

    const float f5  = 2.f * static_cast<float>(M_PI) * 7.f / 12.f;
    const float f3m = 2.f * static_cast<float>(M_PI) * 3.f / 12.f;
    const float f3M = 2.f * static_cast<float>(M_PI) * 4.f / 12.f;

    TonnetzResult r;
    r.tonnetz.assign(6, FloatVec(n_frames_cqt, 0.f));
    for (int t = 0; t < n_frames_cqt; t++) {
        float r5s=0,r5c=0,r3ms=0,r3mc=0,r3Ms=0,r3Mc=0;
        for (int pc = 0; pc < 12; pc++) {
            float ch = chroma[pc][t];
            r5s  += ch*sinf(pc*f5);  r5c  += ch*cosf(pc*f5);
            r3ms += ch*sinf(pc*f3m); r3mc += ch*cosf(pc*f3m);
            r3Ms += ch*sinf(pc*f3M); r3Mc += ch*cosf(pc*f3M);
        }
        r.tonnetz[0][t]=r5s; r.tonnetz[1][t]=r5c;
        r.tonnetz[2][t]=r3ms; r.tonnetz[3][t]=r3mc;
        r.tonnetz[4][t]=r3Ms; r.tonnetz[5][t]=r3Mc;
    }
    r.dim_mean.resize(6); r.dim_std.resize(6);
    float tz_sum = 0, tz_sum2 = 0; size_t tz_cnt = 0;
    for (int d = 0; d < 6; d++) {
        float m = mean_v(r.tonnetz[d]);
        r.dim_mean[d] = m;
        r.dim_std[d]  = std_v(r.tonnetz[d], m);
        for (auto v : r.tonnetz[d]) { tz_sum += v; tz_sum2 += v*v; ++tz_cnt; }
    }
    r.mean   = tz_cnt > 0 ? tz_sum / tz_cnt : 0.f;
    float tz_var = tz_cnt > 1 ? tz_sum2/tz_cnt - (tz_sum/tz_cnt)*(tz_sum/tz_cnt) : 0.f;
    r.spread = sqrtf(std::max(0.f, tz_var));
    return r;
}

KeyResult extract_key(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S      = stft_magnitude(y, n_fft, hop_length);
    Float2D chroma = chroma_from_S(S, n_fft, sr);

    FloatVec pc_mean(12);
    for (int pc = 0; pc < 12; pc++) pc_mean[pc] = mean_v(chroma[pc]);

    const float major_profile[12] = {6.35f,2.23f,3.48f,2.33f,4.38f,4.09f,
                                      2.52f,5.19f,2.39f,3.66f,2.29f,2.88f};
    const float minor_profile[12] = {6.33f,2.68f,3.52f,5.38f,2.60f,3.53f,
                                      2.54f,4.75f,3.98f,2.69f,3.34f,3.17f};

    auto correlate = [&](const float* prof, int root) {
        float mx=0, my=0;
        for (int i=0;i<12;i++) { mx+=pc_mean[(i+root)%12]; my+=prof[i]; }
        mx/=12; my/=12;
        float num=0,dx=0,dy=0;
        for (int i=0;i<12;i++) {
            float a=pc_mean[(i+root)%12]-mx, b=prof[i]-my;
            num+=a*b; dx+=a*a; dy+=b*b;
        }
        float denom=sqrtf(dx*dy);
        return (denom>1e-10f) ? num/denom : 0.f;
    };

    float best=-2.f, sum_c=0.f; int best_root=0; bool best_major=true;
    for (int root=0; root<12; root++) {
        float cm=correlate(major_profile,root), cn=correlate(minor_profile,root);
        sum_c += cm+cn;
        if (cm>best) { best=cm; best_root=root; best_major=true; }
        if (cn>best) { best=cn; best_root=root; best_major=false; }
    }
    KeyResult r;
    r.strength = best;
    r.clarity  = best - sum_c/24.f;
    r.root     = best_root;
    r.is_major = best_major;
    return r;
}

ChordResult extract_chord_templates(const FloatVec& y, int sr, int hop_length) {
    auto cqt = extract_cqt_chroma(y, sr, hop_length);
    Float2D& chroma  = cqt.chroma;
    int n_frames_cqt = chroma.empty() ? 0 : static_cast<int>(chroma[0].size());

    const int triad_intervals[3] = {0, 4, 7};
    FloatVec matches;

    for (int t = 0; t < n_frames_cqt; t += 4) {
        FloatVec frame_ch(12);
        for (int pc = 0; pc < 12; pc++) frame_ch[pc] = chroma[pc][t];
        float norm_ch = 0;
        for (auto v : frame_ch) norm_ch += v*v;
        norm_ch = sqrtf(norm_ch) + 1e-10f;

        for (int root = 0; root < 12; root++) {
            FloatVec tmpl(12, 0.f);
            for (int iv : triad_intervals) tmpl[(root + iv) % 12] = 1.f;
            float dot = 0;
            for (int pc = 0; pc < 12; pc++) dot += frame_ch[pc] * tmpl[pc];
            matches.push_back(dot / norm_ch);
        }
    }

    ChordResult r;
    r.recognition_mean = mean_v(matches);
    FloatVec rounded(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
        rounded[i] = roundf(matches[i] * 10.f) / 10.f;
    std::sort(rounded.begin(), rounded.end());
    auto last = std::unique(rounded.begin(), rounded.end());
    r.changes_detected = static_cast<float>(std::distance(rounded.begin(), last));
    return r;
}

TonalCentroidResult extract_tonal_centroid(const FloatVec& y, int sr, int hop_length) {
    auto cqt = extract_cqt_chroma(y, sr, hop_length);
    FloatVec cmean = cqt.pc_mean;
    float total = 0;
    for (auto v : cmean) total += v;
    if (total > 1e-10f) for (auto& v : cmean) v /= total;

    TonalCentroidResult r;
    r.pc1 = 0; r.pc2 = 0;
    for (int pc = 0; pc < 12; pc++) {
        float angle = 2.f * static_cast<float>(M_PI) * pc / 12.f;
        r.pc1 += cmean[pc] * cosf(angle);
        r.pc2 += cmean[pc] * sinf(angle);
    }
    return r;
}

HarmonicChangeResult extract_harmonic_changes(const FloatVec& y, int sr, int hop_length) {
    auto cqt = extract_cqt_chroma(y, sr, hop_length);
    Float2D& chroma  = cqt.chroma;
    int n_frames_cqt = chroma.empty() ? 0 : static_cast<int>(chroma[0].size());

    FloatVec hcdf(n_frames_cqt, 0.f);
    for (int t = 1; t < n_frames_cqt; t++) {
        float diff = 0, energy = 0;
        for (int pc = 0; pc < 12; pc++) {
            diff   += fabsf(chroma[pc][t] - chroma[pc][t-1]);
            energy += chroma[pc][t] + chroma[pc][t-1];
        }
        hcdf[t] = diff / (energy / 2.f + 1e-10f);
    }

    HarmonicChangeResult r;
    r.mean   = mean_v(hcdf);
    r.std    = std_v(hcdf, r.mean);
    float thresh = 1.5f * r.mean;
    r.peaks  = 0;
    for (int t = 1; t < n_frames_cqt - 1; t++)
        if (hcdf[t] > thresh && hcdf[t] >= hcdf[t-1] && hcdf[t] >= hcdf[t+1])
            ++r.peaks;
    float dur_s = static_cast<float>(n_frames_cqt * hop_length * 2) / sr;
    r.rate = (dur_s > 0.f) ? r.peaks / dur_s : 0.f;
    return r;
}

// ============================================================
//  Aggregate wrapper
// ============================================================

ChromaFeatures extract_chroma(const FloatVec& y, int sr, int n_fft, int hop_length) {
    ChromaFeatures f;

    auto cqtc = extract_cqt_chroma(y, sr, hop_length);
    f.chroma_cqt         = cqtc.chroma;
    f.chroma_cqt_pc_mean = cqtc.pc_mean;
    f.chroma_cqt_pc_std  = cqtc.pc_std;
    f.chroma_cqt_mean    = cqtc.mean;
    f.chroma_cqt_entropy = cqtc.entropy;

    auto stftc = extract_stft_chroma(y, sr, n_fft, hop_length);
    f.chroma_stft             = stftc.chroma;
    f.chroma_stft_pc_mean     = stftc.pc_mean;
    f.chroma_stft_correlation = stftc.stft_correlation;

    auto tz = extract_tonnetz(y, sr, hop_length);
    f.tonnetz          = tz.tonnetz;
    f.tonnetz_dim_mean = tz.dim_mean;
    f.tonnetz_dim_std  = tz.dim_std;
    f.tonnetz_mean_val = tz.mean;
    f.tonnetz_spread   = tz.spread;

    auto key = extract_key(y, sr, n_fft, hop_length);
    f.key_strength = key.strength; f.key_clarity = key.clarity;
    f.key_root = key.root; f.key_is_major = key.is_major;

    auto chord = extract_chord_templates(y, sr, hop_length);
    f.chord_recognition_mean  = chord.recognition_mean;
    f.chord_changes_detected  = chord.changes_detected;

    auto tc = extract_tonal_centroid(y, sr, hop_length);
    f.tonal_centroid_pc1 = tc.pc1; f.tonal_centroid_pc2 = tc.pc2;

    auto hc = extract_harmonic_changes(y, sr, hop_length);
    f.harmonic_change_mean  = hc.mean;  f.harmonic_change_std = hc.std;
    f.harmonic_change_peaks = hc.peaks; f.harmonic_change_rate = hc.rate;

    return f;
}
