#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <functional>

#include "audio_utils.h"
#include "features/time_domain.h"
#include "features/spectral.h"
#include "features/time_frequency.h"
#include "features/cepstral.h"
#include "features/pitch.h"
#include "features/chroma.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct BenchResult {
    std::string name;
    double min_ms, max_ms, mean_ms, std_ms;
};

static BenchResult benchmark(const std::string& name,
                              std::function<void()> fn,
                              int n_runs = 10) {
    std::vector<double> times;
    times.reserve(n_runs);
    fn(); // warm-up
    for (int i = 0; i < n_runs; i++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times.push_back(Ms(t1 - t0).count());
    }
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    var /= times.size();
    return {name,
            *std::min_element(times.begin(), times.end()),
            *std::max_element(times.begin(), times.end()),
            mean, std::sqrt(var)};
}

static void print_result(const BenchResult& r) {
    std::cout << std::left  << std::setw(30) << r.name
              << std::right << std::fixed << std::setprecision(2)
              << "  min=" << std::setw(8)  << r.min_ms  << " ms"
              << "  mean=" << std::setw(8) << r.mean_ms << " ms"
              << "  max=" << std::setw(8)  << r.max_ms  << " ms"
              << "  std=" << std::setw(7)  << r.std_ms  << " ms"
              << "\n";
}

static void print_section(const char* title) {
    std::cout << "\n" << std::string(90, '-') << "\n"
              << "  " << title << "\n"
              << std::string(90, '-') << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark <audio_file> [n_runs=10]\n";
        return 1;
    }

    std::string path = argv[1];
    int n_runs = (argc >= 3) ? std::stoi(argv[2]) : 10;

    std::cout << "Loading: " << path << "\n";
    AudioBuffer buf;
    try { buf = to_mono(load_audio(path)); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }

    int sr = buf.sample_rate;
    const FloatVec& y = buf.samples;
    double duration_s = static_cast<double>(y.size()) / sr;

    std::cout << "Sample rate : " << sr << " Hz\n"
              << "Duration    : " << std::fixed << std::setprecision(3) << duration_s << " s\n"
              << "Samples     : " << y.size() << "\n"
              << "Runs        : " << n_runs << "\n";

    const int nfft = 2048, hop = 512, flen = 2048;

    // ====================================================================
    //  Time-domain
    // ====================================================================
    print_section("Time-domain features");
    print_result(benchmark("zcr",
        [&]{ auto _ = extract_zcr(y, sr, flen, hop); (void)_; }, n_runs));
    print_result(benchmark("rms",
        [&]{ auto _ = extract_rms(y, sr, flen, hop); (void)_; }, n_runs));
    print_result(benchmark("tempo_beats",
        [&]{ auto _ = extract_tempo_beats(y, sr, hop); (void)_; }, n_runs));
    print_result(benchmark("amplitude_envelope",
        [&]{ auto _ = extract_amplitude_envelope(y, sr, flen, hop); (void)_; }, n_runs));
    print_result(benchmark("temporal_moments",
        [&]{ auto _ = extract_temporal_moments(y, sr, flen, hop); (void)_; }, n_runs));
    print_result(benchmark("attack_decay",
        [&]{ auto _ = extract_attack_decay(y, sr, flen, hop); (void)_; }, n_runs));
    print_result(benchmark("periodicity",
        [&]{ auto _ = extract_periodicity(y, sr); (void)_; }, n_runs));
    print_result(benchmark("envelope_modulation",
        [&]{ auto _ = extract_envelope_modulation(y, sr, flen, hop); (void)_; }, n_runs));

    // ====================================================================
    //  Spectral
    // ====================================================================
    print_section("Spectral features");
    print_result(benchmark("spectral_core",
        [&]{ auto _ = extract_spectral_core(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("spectral_contrast",
        [&]{ auto _ = extract_spectral_contrast(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("hnr",
        [&]{ auto _ = extract_hnr(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("inharmonicity",
        [&]{ auto _ = extract_inharmonicity(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("spectral_peaks",
        [&]{ auto _ = extract_spectral_peaks(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("formants",
        [&]{ auto _ = extract_formants(y, sr, nfft, hop); (void)_; }, n_runs));

    // ====================================================================
    //  Time-frequency
    // ====================================================================
    print_section("Time-frequency features");
    print_result(benchmark("stft_db",
        [&]{ auto _ = extract_stft_db(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("mel_spectrogram",
        [&]{ auto _ = extract_mel_spectrogram(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("bark_spectrogram",
        [&]{ auto _ = extract_bark_spectrogram(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("erb_spectrogram",
        [&]{ auto _ = extract_erb_spectrogram(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("cqt",
        [&]{ auto _ = extract_cqt(y, sr, hop); (void)_; }, n_runs));
    print_result(benchmark("peaks_tf",
        [&]{ auto _ = extract_peaks_tf(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("modulation_spec",
        [&]{ auto _ = extract_modulation_spec(y, sr, nfft, hop); (void)_; }, n_runs));

    // ====================================================================
    //  Cepstral
    // ====================================================================
    print_section("Cepstral features");
    print_result(benchmark("mfcc",
        [&]{ auto _ = extract_mfcc(y, sr, 13, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("lfcc",
        [&]{ auto _ = extract_lfcc(y, sr, 13, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("plp",
        [&]{ auto _ = extract_plp(y, sr, 13, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("rasta_plp",
        [&]{ auto _ = extract_rasta_plp(y, sr, 13, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("gfcc",
        [&]{ auto _ = extract_gfcc(y, sr, 13, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("gtcc",
        [&]{ auto _ = extract_gtcc(y, sr, 13, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("pncc",
        [&]{ auto _ = extract_pncc(y, sr, 13, nfft, hop); (void)_; }, n_runs));

    // ====================================================================
    //  Pitch
    // ====================================================================
    print_section("Pitch features");
    print_result(benchmark("f0_contours",
        [&]{ auto _ = extract_f0_contours(y, sr); (void)_; }, n_runs));
    print_result(benchmark("harmonic_ratios",
        [&]{ auto _ = extract_harmonic_ratios(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("harmonic_deviation",
        [&]{ auto _ = extract_harmonic_deviation(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("vibrato",
        [&]{ auto _ = extract_vibrato(y, sr); (void)_; }, n_runs));
    print_result(benchmark("pitch_salience",
        [&]{ auto _ = extract_pitch_salience(y, sr); (void)_; }, n_runs));
    print_result(benchmark("pitch_class",
        [&]{ auto _ = extract_pitch_class(y, sr, hop); (void)_; }, n_runs));

    // ====================================================================
    //  Chroma / tonal
    // ====================================================================
    print_section("Chroma / tonal features");
    print_result(benchmark("cqt_chroma",
        [&]{ auto _ = extract_cqt_chroma(y, sr, hop); (void)_; }, n_runs));
    print_result(benchmark("stft_chroma",
        [&]{ auto _ = extract_stft_chroma(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("tonnetz",
        [&]{ auto _ = extract_tonnetz(y, sr, hop); (void)_; }, n_runs));
    print_result(benchmark("key",
        [&]{ auto _ = extract_key(y, sr, nfft, hop); (void)_; }, n_runs));
    print_result(benchmark("chord_templates",
        [&]{ auto _ = extract_chord_templates(y, sr, hop); (void)_; }, n_runs));
    print_result(benchmark("tonal_centroid",
        [&]{ auto _ = extract_tonal_centroid(y, sr, hop); (void)_; }, n_runs));
    print_result(benchmark("harmonic_changes",
        [&]{ auto _ = extract_harmonic_changes(y, sr, hop); (void)_; }, n_runs));

    std::cout << "\n" << std::string(90, '=') << "\n"
              << "  Audio duration: " << std::fixed << std::setprecision(3)
              << duration_s << " s  |  " << n_runs << " runs per feature\n"
              << std::string(90, '=') << "\n";

    return 0;
}
