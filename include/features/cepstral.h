#pragma once
#include "audio_utils.h"

// ---- aggregate result ----
struct CepstralFeatures {
    Float2D mfcc, lfcc, plp, rasta_plp, gfcc, gtcc, pncc;
    Float2D mfcc_delta, lfcc_delta, plp_delta, rasta_plp_delta;
    Float2D gfcc_delta, gtcc_delta, pncc_delta;
    Float2D mfcc_delta2, lfcc_delta2, plp_delta2, rasta_plp_delta2;
    Float2D gfcc_delta2, gtcc_delta2, pncc_delta2;
    FloatVec mfcc_mean, lfcc_mean, plp_mean, gfcc_mean;
    FloatVec mfcc_delta_mean, mfcc_delta2_mean;
    FloatVec gfcc_delta_mean,  gfcc_delta2_mean;
};

CepstralFeatures extract_cepstral(const FloatVec& y, int sr,
                                  int n_mfcc = 13, int n_fft = 2048, int hop_length = 512);

// ---- individual feature functions ----
// Each returns base [n_mfcc][n_frames], delta, and delta-delta matrices.
struct CepstralResult {
    Float2D base, delta, delta2;
};

CepstralResult extract_mfcc     (const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
CepstralResult extract_lfcc     (const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
CepstralResult extract_plp      (const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
CepstralResult extract_rasta_plp(const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
CepstralResult extract_gfcc     (const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
CepstralResult extract_gtcc     (const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
CepstralResult extract_pncc     (const FloatVec& y, int sr, int n_mfcc=13, int n_fft=2048, int hop_length=512);
