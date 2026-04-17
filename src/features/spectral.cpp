#include "features/spectral.h"
#include "fft_utils.h"
#include "features/pitch.h"
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
    for (auto x : v) s += (x-m)*(x-m);
    return sqrtf(s / v.size());
}

// ---- LPC formant helpers ----
static FloatVec levinson_durbin(const FloatVec& r, int order) {
    FloatVec a(order + 1, 0.f); a[0] = 1.f;
    float err = r[0];
    if (err < 1e-12f) return a;
    for (int m = 1; m <= order; m++) {
        float lam = 0.f;
        for (int j = 1; j < m; j++) lam += a[j] * r[m-j];
        lam = -(r[m] + lam) / err;
        FloatVec an(a); for (int j = 1; j < m; j++) an[j] = a[j] + lam*a[m-j];
        an[m] = lam; a = an; err *= (1.f - lam*lam);
        if (err < 1e-12f) break;
    }
    return a;
}

static std::vector<float> frame_formants(const FloatVec& frame, int sr,
                                          int lpc_order = 12, int n_pts = 512) {
    int N = static_cast<int>(frame.size());
    if (N < lpc_order + 2) return {0.f, 0.f, 0.f};
    FloatVec win(N);
    float pe = 0.97f;
    win[0] = frame[0];
    for (int i = 1; i < N; i++)
        win[i] = (frame[i] - pe*frame[i-1]) * 0.5f*(1.f - cosf(2.f*M_PI*i/(N-1)));
    FloatVec r(lpc_order + 1, 0.f);
    for (int lag = 0; lag <= lpc_order; lag++) {
        float s = 0.f;
        for (int i = lag; i < N; i++) s += win[i]*win[i-lag];
        r[lag] = s;
    }
    FloatVec a = levinson_durbin(r, lpc_order);
    FloatVec H(n_pts);
    for (int k = 0; k < n_pts; k++) {
        float w = M_PI * k / (n_pts - 1), re = 0.f, im = 0.f;
        for (int p = 0; p <= lpc_order; p++) { re += a[p]*cosf(-w*p); im += a[p]*sinf(-w*p); }
        float mag2 = re*re + im*im;
        H[k] = (mag2 > 1e-20f) ? 1.f/sqrtf(mag2) : 0.f;
    }
    std::vector<std::pair<float,float>> peaks;
    for (int k = 1; k < n_pts-1; k++) {
        if (H[k] > H[k-1] && H[k] > H[k+1]) {
            float hz = static_cast<float>(k) * sr / (2.f*(n_pts-1));
            if (hz > 50.f && hz < sr/2.f) peaks.push_back({hz, H[k]});
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](auto& a, auto& b){ return a.second > b.second; });
    std::vector<float> fhz;
    for (auto& p : peaks) fhz.push_back(p.first);
    std::sort(fhz.begin(), fhz.end());
    while (fhz.size() < 3) fhz.push_back(0.f);
    return {fhz[0], fhz[1], fhz[2]};
}

// ============================================================
//  Individual feature functions
// ============================================================

SpectralCoreResult extract_spectral_core(const FloatVec& y, int sr,
                                          int n_fft, int hop_length) {
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    float df = static_cast<float>(sr) / n_fft;
    FloatVec freqs(n_bins);
    for (int k = 0; k < n_bins; k++) freqs[k] = k * df;

    SpectralCoreResult r;
    r.centroid.resize(n_frames); r.bandwidth.resize(n_frames);
    r.rolloff.resize(n_frames);  r.flatness.resize(n_frames);
    r.flux.resize(n_frames, 0.f); r.slope.resize(n_frames, 0.f);

    for (int t = 0; t < n_frames; t++) {
        float num=0,den=0;
        for (int k=0;k<n_bins;k++) { num+=freqs[k]*S[k][t]; den+=S[k][t]; }
        r.centroid[t] = den>1e-10f ? num/den : 0.f;

        float c=r.centroid[t], bnum=0;
        for (int k=0;k<n_bins;k++) { float d=freqs[k]-c; bnum+=d*d*S[k][t]; }
        r.bandwidth[t] = den>1e-10f ? sqrtf(bnum/den) : 0.f;

        float total=0;
        for (int k=0;k<n_bins;k++) total+=S[k][t];
        float target=0.85f*total, cumsum=0;
        for (int k=0;k<n_bins;k++) { cumsum+=S[k][t]; if(cumsum>=target){r.rolloff[t]=freqs[k];break;} }

        float logs=0,lins=0; int cnt=0;
        for (int k=0;k<n_bins;k++) if(S[k][t]>1e-10f){logs+=logf(S[k][t]);lins+=S[k][t];++cnt;}
        if(cnt>0&&lins>1e-10f) r.flatness[t]=expf(logs/cnt)/(lins/cnt);

        if (t > 0) {
            float fs=0;
            for (int k=0;k<n_bins;k++){float d=S[k][t]-S[k][t-1]; fs+=d*d;}
            r.flux[t]=fs;
        }

        FloatVec vf,vm;
        for (int k=0;k<n_bins;k++) if(S[k][t]>1e-10f){vf.push_back(freqs[k]);vm.push_back(log10f(S[k][t]));}
        int n=static_cast<int>(vf.size());
        if(n>=2){float sx=0,sy=0,sxx=0,sxy=0;
            for(int i=0;i<n;i++){sx+=vf[i];sy+=vm[i];sxx+=vf[i]*vf[i];sxy+=vf[i]*vm[i];}
            float denom=n*sxx-sx*sx;
            if(fabsf(denom)>1e-20f) r.slope[t]=(n*sxy-sx*sy)/denom;
        }
    }
    r.centroid_mean=mean_v(r.centroid); r.centroid_std=std_v(r.centroid,r.centroid_mean);
    r.bandwidth_mean=mean_v(r.bandwidth); r.rolloff_mean=mean_v(r.rolloff);
    r.flatness_mean=mean_v(r.flatness);
    r.flux_mean=mean_v(r.flux); r.flux_std=std_v(r.flux,r.flux_mean);
    r.slope_mean=mean_v(r.slope);
    return r;
}

Float2D extract_spectral_contrast(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    float df = static_cast<float>(sr) / n_fft;
    FloatVec freqs(n_bins);
    for (int k = 0; k < n_bins; k++) freqs[k] = k * df;

    const int n_bands = 6;
    Float2D contrast(n_bands + 1, FloatVec(n_frames, 0.f));
    for (int t = 0; t < n_frames; t++) {
        for (int b = 0; b <= n_bands; b++) {
            float lo = 200.f * powf(2.f, static_cast<float>(b));
            float hi = (b == n_bands) ? sr/2.f : 200.f*powf(2.f,static_cast<float>(b+1));
            FloatVec vals;
            for (int k=0;k<n_bins;k++) if(freqs[k]>=lo&&freqs[k]<hi) vals.push_back(S[k][t]);
            if (vals.empty()) continue;
            std::sort(vals.begin(), vals.end());
            int n_top = std::max(1, static_cast<int>(vals.size()*0.1f));
            float peak=0, valley=0;
            for (int i=static_cast<int>(vals.size())-n_top;i<static_cast<int>(vals.size());i++) peak+=vals[i];
            for (int i=0;i<n_top;i++) valley+=vals[i];
            peak/=n_top; valley/=n_top;
            contrast[b][t] = (valley>1e-10f) ? log10f(peak/valley+1e-10f) : 0.f;
        }
    }
    return contrast;
}

float extract_hnr(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;

    auto pf = extract_f0_contours(y, sr, 65.4f, 2093.f, hop_length);
    float total_harm = 0, total_energy = 0;
    for (int t = 0; t < n_frames && t < static_cast<int>(pf.f0.size()); t++) {
        if (pf.f0[t] < 1.f) continue;
        float f0=pf.f0[t], tot=0, harm=0;
        for (int k=0;k<n_bins;k++) tot+=S[k][t]*S[k][t];
        for (int h=1;h<=9;h++) {
            int bin=static_cast<int>(h*f0*n_fft/sr+0.5f);
            if(bin>=0&&bin<n_bins) harm+=S[bin][t]*S[bin][t];
        }
        total_harm+=harm; total_energy+=tot;
    }
    if (total_energy > 1e-10f && total_harm < total_energy) {
        float noise = total_energy - total_harm;
        return (noise > 1e-10f) ? 10.f*log10f(total_harm/noise) : 60.f;
    }
    return 0.f;
}

InharmonicityResult extract_inharmonicity(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    float df = static_cast<float>(sr) / n_fft;

    InharmonicityResult r;
    for (int t = 0; t < std::min(n_frames, 100); t++) {
        FloatVec pf;
        for (int k=1;k<n_bins-1;k++) {
            if(S[k][t]>S[k-1][t]&&S[k][t]>S[k+1][t]) {
                float fm=0; for(int kk=0;kk<n_bins;kk++) fm+=S[kk][t]; fm/=n_bins;
                if(S[k][t]>fm*2.f) pf.push_back(k*df);
            }
        }
        if (pf.size() < 3) continue;
        float f0e=pf[0]; if(f0e<1.f) continue;
        float dev=0; int cnt=0;
        for(float ff:pf){float ratio=ff/f0e,nearest=roundf(ratio);if(nearest<1)nearest=1;dev+=fabsf(ratio/nearest-1.f);++cnt;}
        if(cnt>0) r.values.push_back(dev/cnt);
    }
    r.mean = mean_v(r.values);
    return r;
}

SpectralPeakResult extract_spectral_peaks(const FloatVec& y, int sr, int n_fft, int hop_length) {
    Float2D S = stft_magnitude(y, n_fft, hop_length);
    int n_bins = static_cast<int>(S.size());
    int n_frames = n_bins > 0 ? static_cast<int>(S[0].size()) : 0;
    float df = static_cast<float>(sr) / n_fft;

    FloatVec all_freqs, peak_counts;
    for (int t = 0; t < n_frames; t++) {
        FloatVec col(n_bins);
        for (int k=0;k<n_bins;k++) col[k]=S[k][t];
        FloatVec sc=col; std::sort(sc.begin(),sc.end());
        float t80=sc[static_cast<int>(sc.size()*0.8f)];
        float t90=sc[static_cast<int>(sc.size()*0.9f)];
        std::vector<std::pair<float,int>> peaks_m;
        for(int k=1;k<n_bins-1;k++)
            if(col[k]>t80&&col[k]>col[k-1]&&col[k]>col[k+1])
                peaks_m.push_back({col[k],k});
        std::sort(peaks_m.begin(),peaks_m.end(),[](auto&a,auto&b){return a.first>b.first;});
        int n_top=std::min(5,static_cast<int>(peaks_m.size()));
        for(int i=0;i<n_top;i++) all_freqs.push_back(peaks_m[i].second*df);
        int cnt90=0; for(int k=0;k<n_bins;k++) if(col[k]>t90) ++cnt90;
        peak_counts.push_back(static_cast<float>(cnt90));
    }
    SpectralPeakResult r;
    r.freq_mean  = mean_v(all_freqs);
    r.count_mean = mean_v(peak_counts);
    return r;
}

FormantResult extract_formants(const FloatVec& y, int sr, int n_fft, int hop_length) {
    int N = static_cast<int>(y.size());
    int n_frames = 1 + N / hop_length;
    FloatVec f1s, f2s, f3s;
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop_length, end = std::min(start + n_fft, N);
        if (end - start < 64) continue;
        FloatVec frame(y.begin()+start, y.begin()+end);
        frame.resize(n_fft, 0.f);
        auto fm = frame_formants(frame, sr);
        if(fm[0]>50.f) f1s.push_back(fm[0]);
        if(fm[1]>50.f) f2s.push_back(fm[1]);
        if(fm[2]>50.f) f3s.push_back(fm[2]);
    }
    FormantResult r;
    r.f1 = mean_v(f1s); r.f2 = mean_v(f2s); r.f3 = mean_v(f3s);
    return r;
}

// ============================================================
//  Aggregate wrapper
// ============================================================

SpectralFeatures extract_spectral(const FloatVec& y, int sr, int n_fft, int hop_length) {
    SpectralFeatures f;

    auto core = extract_spectral_core(y, sr, n_fft, hop_length);
    f.centroid=core.centroid; f.centroid_mean=core.centroid_mean; f.centroid_std=core.centroid_std;
    f.bandwidth=core.bandwidth; f.bandwidth_mean=core.bandwidth_mean;
    f.rolloff=core.rolloff; f.rolloff_mean=core.rolloff_mean;
    f.flatness=core.flatness; f.flatness_mean=core.flatness_mean;
    f.flux=core.flux; f.flux_mean=core.flux_mean; f.flux_std=core.flux_std;
    f.slope=core.slope; f.slope_mean=core.slope_mean;

    f.contrast = extract_spectral_contrast(y, sr, n_fft, hop_length);
    f.hnr      = extract_hnr(y, sr, n_fft, hop_length);

    auto inh = extract_inharmonicity(y, sr, n_fft, hop_length);
    f.inharmonicity=inh.values; f.inharmonicity_mean=inh.mean;

    auto pk = extract_spectral_peaks(y, sr, n_fft, hop_length);
    f.spectral_peak_freqs_mean=pk.freq_mean; f.spectral_peak_count_mean=pk.count_mean;

    auto fm = extract_formants(y, sr, n_fft, hop_length);
    f.formant_f1=fm.f1; f.formant_f2=fm.f2; f.formant_f3=fm.f3;

    return f;
}
