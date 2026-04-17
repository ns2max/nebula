#pragma once
#include <vector>
#include <string>
#include <unordered_map>

using FloatVec = std::vector<float>;
using Float2D  = std::vector<FloatVec>;

struct AudioBuffer {
    FloatVec samples;
    int      sample_rate  = 0;
    int      num_channels = 0;
};

AudioBuffer load_audio(const std::string& path);
AudioBuffer to_mono(const AudioBuffer& buf);
