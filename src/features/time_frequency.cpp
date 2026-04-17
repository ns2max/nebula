#include "features/time_frequency.h"
#include "fft_utils.h"
#include "filterbank.h"
#include <cmath>
#include <algorithm>
#include <numeric>

static float mean_all(const Float2D& M) {
    float s=0; size_t cnt=0;
    for(auto& row:M) for(auto v:row){s+=v;++cnt;}
    return cnt ? s/cnt : 0.f;
}
static float std_all(const Float2D& M, float m) {
    float s=0; size_t cnt=0;
    for(auto& row:M) for(auto v:row){s+=(v-m)*(v-m);++cnt;}
    return cnt>1 ? sqrtf(s/cnt) : 0.f;
}

static Float2D cqt_approx_power(const FloatVec& y, int sr, int hop_length,
                                 int n_bins=84, int bins_per_octave=12,
                                 float fmin=32.7f) {
    int cqt_fft = 4096;
    Float2D S  = stft_power(y, cqt_fft, hop_length);
    int n_stft = static_cast<int>(S.size());
    int n_frm  = n_stft > 0 ? static_cast<int>(S[0].size()) : 0;
    float df   = static_cast<float>(sr) / cqt_fft;
    float Q    = 1.f / (powf(2.f, 1.f/bins_per_octave) - 1.f);

    Float2D cqt(n_bins, FloatVec(n_frm, 0.f));
    for (int b = 0; b < n_bins; b++) {
        float fc  = fmin * powf(2.f, static_cast<float>(b) / bins_per_octave);
        float bw  = fc / Q;
        int   lo  = std::max(0, static_cast<int>((fc-bw/2.f)/df));
        int   hi  = std::min(n_stft-1, static_cast<int>((fc+bw/2.f)/df)+1);
        int   cnt = std::max(1, hi-lo+1);
        for (int t = 0; t < n_frm; t++) {
            float s=0; for(int k=lo;k<=hi&&k<n_stft;k++) s+=S[k][t];
            cqt[b][t]=s/cnt;
        }
    }
    return cqt;
}

// ============================================================
//  Individual feature functions
// ============================================================

STFTStats extract_stft_db(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S   = stft_magnitude(y, n_fft, hop_length);
    Float2D Sdb = amplitude_to_db_2d(S);
    STFTStats r;
    r.mean = mean_all(Sdb);
    r.std  = std_all(Sdb, r.mean);
    return r;
}

Float2D extract_mel_spectrogram(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S  = stft_power(y, n_fft, hop_length);
    Float2D fb = mel_filterbank(20, n_fft, sr);
    return power_to_db_2d(apply_filterbank_frames(fb, S));
}

Float2D extract_bark_spectrogram(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S  = stft_power(y, n_fft, hop_length);
    Float2D fb = bark_filterbank(25, n_fft, sr);
    return power_to_db_2d(apply_filterbank_frames(fb, S));
}

Float2D extract_erb_spectrogram(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S  = stft_power(y, n_fft, hop_length);
    Float2D fb = erb_filterbank(30, n_fft, sr);
    return power_to_db_2d(apply_filterbank_frames(fb, S));
}

CQTResult extract_cqt(const FloatVec& y, int sr, int hop_length) {
    Float2D cqt_power = cqt_approx_power(y, sr, hop_length);
    CQTResult r;
    r.spectrogram = power_to_db_2d(cqt_power);
    r.mean = mean_all(r.spectrogram);
    r.std  = std_all(r.spectrogram, r.mean);

    const int bpo = 12;
    r.pitch_classes.assign(bpo, 0.f);
    int n_frm = r.spectrogram.empty() ? 0 : static_cast<int>(r.spectrogram[0].size());
    for (int b = 0; b < static_cast<int>(r.spectrogram.size()); b++) {
        int pc = b % bpo;
        for (int t = 0; t < n_frm; t++) r.pitch_classes[pc] += cqt_power[b][t];
    }
    float pc_sum = 0; for (auto v : r.pitch_classes) pc_sum += v;
    if (pc_sum > 1e-10f) for (auto& v : r.pitch_classes) v /= pc_sum;
    return r;
}

PeaksTFResult extract_peaks_tf(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins  = static_cast<int>(S.size());
    int n_frm   = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    float df    = static_cast<float>(sr) / n_fft;

    FloatVec all_f, all_m;
    int total = 0;
    for (int t = 0; t < n_frm; t++) {
        FloatVec col(n_bins); for(int k=0;k<n_bins;k++) col[k]=S[k][t];
        FloatVec sc=col; std::sort(sc.begin(),sc.end());
        float thresh=sc[static_cast<int>(sc.size()*0.9f)];
        for(int k=0;k<n_bins;k++) if(col[k]>thresh){all_f.push_back(k*df);all_m.push_back(col[k]);++total;}
    }
    PeaksTFResult r{};
    r.n_atoms = static_cast<float>(total) / std::max(n_frm, 1);
    if (!all_f.empty()) {
        float fm=0,mm=0;
        for(auto v:all_f) fm+=v; for(auto v:all_m) mm+=v;
        fm/=all_f.size(); mm/=all_m.size();
        r.freq_mean=fm; r.mag_mean=mm;
        float fs=0,ms=0;
        for(auto v:all_f) fs+=(v-fm)*(v-fm); for(auto v:all_m) ms+=(v-mm)*(v-mm);
        r.freq_std=sqrtf(fs/all_f.size()); r.mag_std=sqrtf(ms/all_m.size());
    }
    return r;
}

ModSpecResult extract_modulation_spec(const FloatVec& y, int sr, int n_fft, int hop_length) {
    const int n_mels=40, mod_fft=128, mod_hop=4;
    Float2D S   = stft_power(y, n_fft, hop_length);
    Float2D fb  = mel_filterbank(n_mels, n_fft, sr);
    Float2D mel = apply_filterbank_frames(fb, S);
    for (auto& row : mel) for (auto& v : row) v = log10f(v + 1e-10f);

    int n_frm = mel.empty() ? 0 : static_cast<int>(mel[0].size());
    if (n_frm < mod_fft) return {0.f, 0.f, 0.f};

    Float2D mod_spec;
    for (int m = 0; m < n_mels; m++) {
        Float2D ms = stft_magnitude(mel[m], mod_fft, mod_hop);
        if (!ms.empty()) mod_spec.insert(mod_spec.end(), ms.begin(), ms.end());
    }
    if (mod_spec.empty()) return {0.f, 0.f, 0.f};

    float mm = mean_all(mod_spec);
    ModSpecResult r;
    r.mean = mm; r.std = std_all(mod_spec, mm);
    FloatVec mean_mod(mod_spec[0].size(), 0.f);
    for (auto& row : mod_spec) for(size_t k=0;k<row.size();k++) mean_mod[k]+=row[k];
    int peak_bin = static_cast<int>(
        std::max_element(mean_mod.begin()+1, mean_mod.end()) - mean_mod.begin());
    r.dominant_rate_hz = static_cast<float>(peak_bin) * (static_cast<float>(sr)/hop_length) / mod_fft;
    return r;
}

// ============================================================
//  Aggregate wrapper
// ============================================================

TimeFrequencyFeatures extract_time_frequency(const FloatVec& y, int sr,
                                              int n_fft, int hop_length) {
    TimeFrequencyFeatures f;

    auto sd = extract_stft_db(y, sr, n_fft, hop_length);
    f.stft_mean=sd.mean; f.stft_std=sd.std;

    f.mel_spectrogram = extract_mel_spectrogram(y, sr, n_fft, hop_length);
    f.mel_mean=mean_all(f.mel_spectrogram); f.mel_std=std_all(f.mel_spectrogram,f.mel_mean);

    f.bark_spectrogram = extract_bark_spectrogram(y, sr, n_fft, hop_length);
    f.bark_mean=mean_all(f.bark_spectrogram); f.bark_std=std_all(f.bark_spectrogram,f.bark_mean);

    f.erb_spectrogram = extract_erb_spectrogram(y, sr, n_fft, hop_length);
    f.erb_mean=mean_all(f.erb_spectrogram); f.erb_std=std_all(f.erb_spectrogram,f.erb_mean);

    auto cqt = extract_cqt(y, sr, hop_length);
    f.cqt=cqt.spectrogram; f.cqt_mean=cqt.mean; f.cqt_std=cqt.std; f.cqt_pitch_classes=cqt.pitch_classes;

    auto pk = extract_peaks_tf(y, sr, n_fft, hop_length);
    f.n_atoms=pk.n_atoms; f.freq_mean=pk.freq_mean; f.freq_std=pk.freq_std;
    f.mag_mean=pk.mag_mean; f.mag_std=pk.mag_std;

    auto ms = extract_modulation_spec(y, sr, n_fft, hop_length);
    f.modulation_mean=ms.mean; f.modulation_std=ms.std; f.dominant_rate_hz=ms.dominant_rate_hz;

    return f;
}
