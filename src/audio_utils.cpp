#include "audio_utils.h"
#include <sndfile.h>
#include <stdexcept>
#include <numeric>

AudioBuffer load_audio(const std::string& path) {
    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf)
        throw std::runtime_error("Cannot open: " + path);

    AudioBuffer buf;
    buf.sample_rate  = info.samplerate;
    buf.num_channels = info.channels;
    buf.samples.resize(static_cast<size_t>(info.frames) * info.channels);
    sf_readf_float(sf, buf.samples.data(), info.frames);
    sf_close(sf);
    return buf;
}

AudioBuffer to_mono(const AudioBuffer& buf) {
    if (buf.num_channels == 1)
        return buf;

    AudioBuffer mono;
    mono.sample_rate  = buf.sample_rate;
    mono.num_channels = 1;
    int n_frames = static_cast<int>(buf.samples.size()) / buf.num_channels;
    mono.samples.resize(n_frames);

    for (int i = 0; i < n_frames; i++) {
        float sum = 0;
        for (int c = 0; c < buf.num_channels; c++)
            sum += buf.samples[i * buf.num_channels + c];
        mono.samples[i] = sum / buf.num_channels;
    }
    return mono;
}
