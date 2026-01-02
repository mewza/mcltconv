/*-----------------------------------------------------------------------------*\
**   MCLT Convolver v0.9 -- MCLT Convolver Tester                              **
**   (c) 2021-2025 Dmitry Boldyrev. All rights reserved.                       **
**                                                                             **
**   mclt_convolver_test.cpp                                                   **
**                                                                             **
**   FFT MCLT Bundle v2.0 -- PUBLIC EDITION --                                 **
**                                                                             **
**    @contact  E-mail: subband@gmail.com or subband@protonmail.com            **
**    @home https://github.com/mewza/mcltconv                                  **
**                                                                             **
**    This software relies on use of FFTReal class available at:               **
**    https://github.com/mewza/realfft/                                        **
\*-----------------------------------------------------------------------------*/


#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

// Your headers
#include "const1.h"
#include "mclt.h"
#include "mclt_convolver.h"

using namespace std;

//=============================================================================
// Test utilities
//=============================================================================

template<typename T>
void print_stats(const char* name, MCLTConvolverHybrid<T>& conv) {
    cout << name << ":\n";
    cout << "  Cross-terms: " << conv.cross_count() << " / " << conv.full_size() << "\n";
    cout << "  Sparsity: " << fixed << setprecision(1) << conv.sparsity() * 100 << "%\n";
}

template<typename F>
double benchmark_us(F&& func, int iters) {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        func();
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, micro>(end - start).count() / iters;
}

//=============================================================================
// Test 1: Round-trip (no convolution)
//=============================================================================

void test_roundtrip() {
    cout << "=== Test: MCLT Round-trip ===\n";
    
    constexpr int L = 512;
    constexpr int M = L / 2;
    
    MCLTReal<float> mclt(L);
    
    vector<float> in(L), out(M);
    vector<cmplxT<float>> coeffs(M);
    
    // Sine input
    for (int i = 0; i < L; i++) {
        in[i] = sin(2.0 * M_PI * i / 32.0);
    }
    
    mclt.analyze(in.data(), coeffs.data());
    mclt.synthesize(coeffs.data());
    mclt.overlap_add_to_buffer(out.data());
    
    cout << "  First 4 output: ";
    for (int i = 0; i < 4; i++) cout << out[i] << " ";
    cout << "\n  PASS\n\n";
}

//=============================================================================
// Test 2: Subband identity
//=============================================================================

void test_subband_identity() {
    cout << "=== Test: Subband Identity ===\n";
    
    constexpr int L = 512;
    constexpr int M = L / 2;
    
    MCLTReal<float> mclt(L);
    
    // Unit impulse IR
    vector<float> ir = {1.0f};
    MCLTConvolverSubband<float> conv(mclt, ir.data(), ir.size());
    
    vector<cmplxT<float>> in(M), out(M);
    for (int k = 0; k < M; k++) {
        in[k] = cmplxT<float>(sin(k * 0.1f), cos(k * 0.1f));
    }
    
    conv.process(in.data(), out.data());
    
    float err = 0;
    for (int k = 0; k < M; k++) {
        float dr = in[k].re - out[k].re;
        float di = in[k].im - out[k].im;
        err += dr*dr + di*di;
    }
    
    cout << "  MSE: " << err / M << "\n";
    cout << (err / M < 0.01 ? "  PASS" : "  FAIL") << "\n\n";
}

//=============================================================================
// Test 3: Hybrid sparsity
//=============================================================================

void test_hybrid_sparsity() {
    cout << "=== Test: Hybrid Sparsity ===\n";
    
    constexpr int L = 512;
    
    MCLTReal<float> mclt(L);
    
    // Smooth IR -> high sparsity expected
    vector<float> ir(256);
    for (int i = 0; i < 256; i++) {
        ir[i] = exp(-float(i) / 50.0f) * cos(2.0 * M_PI * i / 64.0f);
    }
    
    MCLTConvolverHybrid<float> conv(mclt, ir.data(), ir.size(), 0.01f);
    
    print_stats("Smooth IR", conv);
    cout << (conv.sparsity() > 0.8 ? "  PASS" : "  FAIL") << "\n\n";
}

//=============================================================================
// Test 4: Performance benchmark
//=============================================================================

void test_benchmark() {
    cout << "=== Test: Performance ===\n";
    
    constexpr int L = 1024;
    constexpr int M = L / 2;
    constexpr int ITERS = 1000;
    
    MCLTReal<float> mclt(L);
    
    vector<float> ir(512);
    for (int i = 0; i < 512; i++) {
        ir[i] = exp(-float(i) / 100.0f);
    }
    
    MCLTConvolverSubband<float> sub(mclt, ir.data(), ir.size());
    MCLTConvolverHybrid<float> hyb(mclt, ir.data(), ir.size());
    
    vector<cmplxT<float>> in(M), out(M);
    for (int k = 0; k < M; k++) {
        in[k] = cmplxT<float>(sin(k * 0.1f), cos(k * 0.1f));
    }
    
    double us_sub = benchmark_us([&]{ sub.process(in.data(), out.data()); }, ITERS);
    double us_hyb = benchmark_us([&]{ hyb.process(in.data(), out.data()); }, ITERS);
    
    double rt_budget = 1e6 * M / 48000.0;  // Real-time budget at 48kHz
    
    cout << "  Subband: " << fixed << setprecision(2) << us_sub << " µs/frame\n";
    cout << "  Hybrid:  " << us_hyb << " µs/frame\n";
    cout << "  RT budget @ 48kHz: " << rt_budget << " µs\n";
    cout << (us_hyb < rt_budget ? "  PASS" : "  FAIL") << "\n\n";
}

//=============================================================================
// Test 5: Full chain
//=============================================================================

void test_full_chain() {
    cout << "=== Test: Full Chain ===\n";
    
    constexpr int L = 512;
    constexpr int HOP = 256;
    
    MCLTReal<float> mclt(L, HOP, WINTYPE_SINE);
    
    vector<float> ir(128);
    for (int i = 0; i < 128; i++) {
        ir[i] = exp(-float(i) / 30.0f);
    }
    
    MCLTConvolver<float> conv(mclt, ir.data(), ir.size(), MCLTConvolver<float>::HYBRID);
    
    // Process 10 frames
    vector<float> in(L, 0), out(HOP);
    in[L/2] = 1.0f;  // Impulse
    
    float peak = 0;
    for (int f = 0; f < 10; f++) {
        conv.process(in.data(), out.data());
        for (int i = 0; i < HOP; i++) {
            peak = max(peak, fabs(out[i]));
        }
        fill(in.begin(), in.end(), 0.0f);  // Zero after first frame
    }
    
    cout << "  Peak output: " << peak << "\n";
    cout << (peak > 0.01f ? "  PASS" : "  FAIL") << "\n\n";
}

//=============================================================================
// Main
//=============================================================================

int main() {
    cout << "MCLT Convolver Test Suite\n";
    cout << "=========================\n\n";
    
    test_roundtrip();
    test_subband_identity();
    test_hybrid_sparsity();
    test_benchmark();
    test_full_chain();
    
    cout << "=== Done ===\n";
    return 0;
}
