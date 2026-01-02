/*-----------------------------------------------------------------------------*\
**   MCLTReal v2.0 -- Modulated complex lapped forward / inverse transform     **
**   (c) 2021-2025 Dmitry Boldyrev. All rights reserved.                       **
**                                                                             **
**   mclt.h - Main include header (do not include mclt_neon.h directly)        **
**                                                                             **
**   FFT MCLT Bundle v2.0 -- PUBLIC EDITION --                                 **
**                                                                             **
**    Features include:                                                        **
**        - DCT-IV/DST-IV based complex MCLT with orthonormal scaling          **
**        - FFT-accelerated transforms via N/2-point complex FFT algorithm     **
**        - Supports arbitrary HOP sizes (optimized for M, M/2, M/4, M/8)      **
**        - Accumulator-based overlap-add for flexible reconstruction          **
**        - Multiple window types (Sine, Kaiser-Bessel, Hanning, Vorbis)       **
**        - NEON ARM64 / SSE intrinsics optimizations                          **
**        - Reconstruction error < 1e-9 across all hop sizes                   **
**                                                                             **
**    @contact  E-mail: subband@gmail.com or subband@protonmail.com            **
**    @home https://github.com/mewza/mcltconv                                  **
**                                                                             **
**    This software relies on use of FFTReal class available at:               **
**    https://github.com/mewza/realfft/                                        **
\*-----------------------------------------------------------------------------*/


#pragma once

#include <memory>
#include <complex>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include "const1.h"
#include "fftreal.h"

#ifndef D_WINTYPE
#define D_WINTYPE

enum WindowType : int {
    WINTYPE_NONE = 0,
    WINTYPE_HANNING,
    WINTYPE_KAISER,
    WINTYPE_SINE,
    WINTYPE_VORBIS,
    WINTYPE_BLACKMAN
};

#endif // D_WINTYPE

#if !TARGET_OS_MACCATALYST && TARGET_CPU_ARM64 && defined(__ARM_NEON)
#include "mclt_neon.h"
#else

template<typename T>
class MCLTReal {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    static constexpr int PREFETCH_DISTANCE = 16;
    
    int _length;
    int _M;
    int _hop;
    
    struct AlignedDeleterT { void operator()(T* ptr) const { free(ptr); } };
    struct AlignedDeleterT1 { void operator()(T1* ptr) const { free(ptr); } };
    struct AlignedDeleterCmplxTT { void operator()(cmplxTT* ptr) const { free(ptr); } };
    
    std::unique_ptr<T[], AlignedDeleterT> _prev_prev;
    std::unique_ptr<T[], AlignedDeleterT> _prev;
    std::unique_ptr<T[], AlignedDeleterT> _current;
    std::unique_ptr<cmplxTT[], AlignedDeleterCmplxTT> _temp_complex;
    std::unique_ptr<T[], AlignedDeleterT> _temp_real;
    std::unique_ptr<T[], AlignedDeleterT> _u_buf;
    std::unique_ptr<T[], AlignedDeleterT> _v_buf;
    std::unique_ptr<T1[], AlignedDeleterT1> _window;
    std::unique_ptr<T[], AlignedDeleterT> _overlap_buf;
    WindowType _window_type;
    
    FFTReal<T> _fft;
    
    static T* aligned_alloc_T(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, MALLOC_ALIGN, count * sizeof(T)) != 0)
            return nullptr;
        return static_cast<T*>(ptr);
    }
    static cmplxTT* aligned_alloc_cmplxTT(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, MALLOC_ALIGN, count * sizeof(cmplxTT)) != 0)
            return nullptr;
        return static_cast<cmplxTT*>(ptr);
    }
    static T1* aligned_alloc_T1(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, MALLOC_ALIGN, count * sizeof(T1)) != 0)
            return nullptr;
        return static_cast<T1*>(ptr);
    }
    
public:
    explicit MCLTReal(int length, int hop = -1, WindowType window_type = WINTYPE_SINE)
        : _length(length)
        , _M(length / 2)
        , _hop((hop < 0) ? length / 2 : hop)
        , _window_type(window_type)
        , _fft(length / 4)
        , _prev_prev(aligned_alloc_T(length))
        , _prev(aligned_alloc_T(length))
        , _current(aligned_alloc_T(length))
        , _u_buf(aligned_alloc_T(_M))
        , _v_buf(aligned_alloc_T(_M))
        , _overlap_buf(aligned_alloc_T(2 * _M))
        , _temp_complex(aligned_alloc_cmplxTT(length))
        , _temp_real(aligned_alloc_T(6 * length))
        , _window(aligned_alloc_T1(length))
    {
        if (_hop <= 0 || _hop > length) {
            throw std::invalid_argument("Hop size must be between 1 and 2M");
        }
        
        init_window();
        
        reset();
    }
    
    void reset() {
        memset(_prev_prev.get(), 0, _length * sizeof(T));
        memset(_prev.get(), 0, _length * sizeof(T));
        memset(_current.get(), 0, _length * sizeof(T));
        memset(_u_buf.get(), 0, _M * sizeof(T));
        memset(_v_buf.get(), 0, _M * sizeof(T));
        memset(_overlap_buf.get(), 0, 2 * _M * sizeof(T));
        memset(_temp_real.get(), 0, (2 * _length) * sizeof(T));
    }
    
    // ========== FORWARD MCLT (Analysis) ==========
    void analyze(const T* __restrict__ x, cmplxTT* __restrict__ X) {
        T* __restrict__ u = _u_buf.get();
        T* __restrict__ v = _v_buf.get();
        const T1* __restrict__ h = _window.get();
        
        const int M = _M;
        const int M_half = M / 2;
        const int M_three_halves = 3 * M_half;
        
        // Folding
        for (int n = 0; n < M; n++) {
            u[n] = h[M_half + n] * x[M_three_halves - n - 1];
        }
        memcpy(v, u, M * sizeof(T));
        
        for (int n = 0; n < M_half; n++) {
            const T t = h[M_half - 1 - n] * x[M_three_halves + n];
            u[n] += t;
            v[n] -= t;
        }
        
        for (int n = 0; n < M_half; n++) {
            const T t = h[n] * x[n];
            u[n + M_half] -= t;
            v[n + M_half] += t;
        }
        
        // Transforms
        do_dct_iv_fft(u, M);
        do_dst_iv_fft(v, M);
        
        // Pack output
        for (int k = 0; k < M; k++) {
            X[k].re = u[k];
            X[k].im = v[k];
        }
    }

    // ========== INVERSE MCLT (Synthesis) ==========
    void synthesize(const cmplxTT* __restrict__ X) {
        T* __restrict__ u = _u_buf.get();
        T* __restrict__ v = _v_buf.get();
        T* __restrict__ current = _current.get();
        const T1* __restrict__ h = _window.get();
        
        const int M = _M;
        const int M_half = M / 2;
        const int M_three_halves = 3 * M_half;
        const int L = 2 * M;
        
        for (int k = 0; k < M; k++) {
            u[k] = X[k].re;
            v[k] = X[k].im;
        }
        
        do_dct_iv_fft(u, M);
        do_dst_iv_fft(v, M);
        
        // Unfolding
        // Unfolding - scale by 0.5 * hop/M for overlap normalization
       const T1 scale = T1(0.5) * T1(_hop) / T1(_M);
       
       for (int i = 0; i < M_half; i++) {
           current[i] = h[i] * (v[M_half + i] - u[M_half + i]) * scale;
       }
        for (int i = M_half; i < M_three_halves; i++) {
            current[i] = h[i] * (u[M_three_halves - i - 1] + v[M_three_halves - i - 1]) * scale;
        }
        for (int i = M_three_halves; i < L; i++) {
            current[i] = h[i] * (u[i - M_three_halves] - v[i - M_three_halves]) * scale;
        }
    }
    
    void synthesize_orthonormal(const cmplxTT* __restrict__ X, T* __restrict__ output) {
        T* __restrict__ u = _u_buf.get();
        T* __restrict__ v = _v_buf.get();
        const T1* __restrict__ h = _window.get();
        
        const int M = _M;
        const int M_half = M / 2;
        const int M_three_halves = 3 * M_half;
        const int L = 2 * M;
        
        for (int k = 0; k < M; k++) {
            u[k] = X[k].re;
            v[k] = X[k].im;
        }
        
        do_dct_iv_fft(u, M);
        do_dst_iv_fft(v, M);
        
        // NO scale factor - pure orthonormal unfolding
        for (int i = 0; i < M_half; i++) {
            output[i] = h[i] * (v[M_half + i] - u[M_half + i]);
        }
        for (int i = M_half; i < M_three_halves; i++) {
            output[i] = h[i] * (u[M_three_halves - i - 1] + v[M_three_halves - i - 1]);
        }
        for (int i = M_three_halves; i < L; i++) {
            output[i] = h[i] * (u[i - M_three_halves] - v[i - M_three_halves]);
        }
    }
    
    // ========== OVERLAP-ADD OUTPUT ==========
    
    void overlap_add_to_buffer(T* __restrict__ output) {
        overlap_add_to_buffer(output, _hop);
    }

    void overlap_add_to_buffer(T* __restrict__ output, int hop) {
        const int L = 2 * _M;
        
        // 1. FIRST add current frame to overlap buffer
        for (int i = 0; i < L; i++) {
            _overlap_buf[i] += _current[i];
        }
        
        // 2. THEN output first hop samples
        for (int i = 0; i < hop; i++) {
            output[i] = _overlap_buf[i];
        }
        
        // 3. Shift buffer left by hop
        memmove(_overlap_buf.get(), _overlap_buf.get() + hop, (L - hop) * sizeof(T));
        memset(_overlap_buf.get() + L - hop, 0, hop * sizeof(T));
    }
    
    void skip_frame() {
        memset(_current.get(), 0, _length * sizeof(T));
        // Shift overlap buffer
        memmove(_overlap_buf.get(), _overlap_buf.get() + _hop, (_length - _hop) * sizeof(T));
        memset(_overlap_buf.get() + _length - _hop, 0, _hop * sizeof(T));
    }
    void real_mclt(const T* in, cmplxTT* out) { analyze(in, out); }
    void mclt(const T* in, cmplxTT* out) { analyze(in, out); }

    void real_imclt(const cmplxTT* in, T* out) {
        synthesize(in);
        overlap_add_to_buffer(out);
    }
    void imclt(const cmplxTT* in, T* out) { real_imclt(in, out); }

    void synthesize_no_tdac(const cmplxTT* X, T* output) {
        synthesize(X);
        memcpy(output, _current.get(), _length * sizeof(T));
    }
   
    // ========== WINDOW TYPE ==========
    
    void set_window_type(WindowType type) {
        _window_type = type;
        init_window();
    }

    WindowType get_window_type() const { return _window_type; }
    void set_hop(int hop) {
        if (hop <= 0 || hop > _length) {
            throw std::invalid_argument("Hop size must be between 1 and 2M");
        }
        _hop = hop;
        reset();
    }
    // ========== GETTERS ==========
    inline const T1* window() const { return _window.get(); }
    inline const T* get_current_frame() const { return _current.get(); }
    inline int get_length() const { return _length; }
    inline int get_half_length() const { return _M; }
    inline int get_M() const { return _M; }
    inline int get_hop() const { return _hop; }
    inline void get_output(T* output) { memcpy(output, _current.get(), _length * sizeof(T)); }

private:

    void init_window() {
        const int N = _length;
        
        switch (_window_type) {
            case WINTYPE_SINE: {
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    _window[i] = F_SIN((i + 0.5) * K);
                }
                break;
            }
                
            case WINTYPE_KAISER: {
                generate_kbd_window(_window.get(), N, 4.0);
                break;
            }
                
            case WINTYPE_HANNING: {
                for (int i = 0; i < N; i++) {
                    T1 h = 0.5 * (1.0 - F_COS(2.0 * M_PI * (i + 0.5) / T1(N)));
                    _window[i] = F_SQRT(h);
                }
                break;
            }
                
            case WINTYPE_VORBIS: {
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    T1 s = F_SIN((i + 0.5) * K);
                    _window[i] = F_SIN(M_PI * 0.5 * s * s);
                }
                break;
            }
                
            case WINTYPE_BLACKMAN: {
                for (int i = 0; i < N; i++) {
                    T1 t = 2.0 * M_PI * i / T1(N - 1);
                    _window[i] = 0.42 - 0.5 * F_COS(t) + 0.08 * F_COS(2.0 * t);
                }
                break;
            }
                
            default: {
                // Default to sine window
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    _window[i] = F_SIN((i + 0.5) * K);
                }
                break;
            }
        }
    }
    
    
    void do_dst_iv_fft(T* __restrict__ x, int N) {
        // Reverse input
        for (int i = 0; i < N/2; i++) {
            std::swap(x[i], x[N - 1 - i]);
        }
        // DCT-IV
        do_dct_iv_fft(x, N);
        // Apply (-1)^k
        for (int k = 1; k < N; k += 2) {
            x[k] = -x[k];
        }
    }

    void do_dct_iv_fft(T* __restrict__ x, int N) {
        const int N2 = N / 2;
        cmplxTT* y = _temp_complex.get();
        T* fft_temp = _temp_real.get();
        
        // Pack
        for (int n = 0; n < N2; n++) {
            y[n].re = x[2*n];
            y[n].im = x[N - 1 - 2*n];
        }
        
        // Pre-twiddle
        for (int n = 0; n < N2; n++) {
            const T1 angle = -M_PI * T1(n) / T1(N);
            y[n] = y[n] * cmplxTT(F_COS(angle), F_SIN(angle));
        }
        
        // N/2-point complex FFT
        _fft.do_fft_complex(y, fft_temp);
        
        // Post-twiddle
        for (int k = 0; k < N2; k++) {
            const T1 angle = -M_PI * T1(2*k + 0.5) / T1(2 * N);
            y[k] = y[k] * cmplxTT(F_COS(angle), F_SIN(angle));
        }
        
        // Unpack
        const T1 scale = F_SQRT(T1(2.0) / T1(N));
        for (int k = 0; k < N2; k++) {
            x[2*k]         =  y[k].re * scale;
            x[N - 1 - 2*k] = -y[k].im * scale;
        }
    }
    // Direct DCT-IV (O(N²) - use for correctness verification)
    void do_dct_iv_direct(T* __restrict__ x, int n) {
        T* temp = _temp_real.get();
        const T scale = F_SQRT(2.0 / n);
        const T pi_4n = M_PI / (4.0 * n);
        
        for (int k = 0; k < n; k++) {
            T sum = 0;
            const T kk = (2 * k + 1) * pi_4n;
            for (int i = 0; i < n; i++) {
                sum += x[i] * F_COS(kk * (2 * i + 1));
            }
            temp[k] = sum * scale;
        }
        memcpy(x, temp, n * sizeof(T));
    }
    
    void do_dst_iv_direct(T* __restrict__ x, int n) {
        T* temp = _temp_real.get();
        const T scale = F_SQRT(2.0 / n);
        
        for (int k = 0; k < n; k++) {
            T sum = 0;
            for (int i = 0; i < n; i++) {
                sum += x[i] * F_SIN(M_PI * (2*k + 1) * (2*i + 1) / (4.0 * n));
            }
            temp[k] = sum * scale;
        }
        memcpy(x, temp, n * sizeof(T));
    }
    
    inline void generate_kbd_window(T1* window, int length, T1 alpha = 4.0) {
        const int N = length;
        const int N2 = N / 2;
        
        // Generate Kaiser window for N/2 + 1 points
        std::vector<T1> kaiser(N2 + 1);
        T1 i0_alpha = bessel_i0(M_PI * alpha);
        
        for (int i = 0; i <= N2; i++) {
            T1 x = (2.0 * i / T1(N2)) - 1.0;  // Maps [0, N2] to [-1, 1]
            T1 arg = M_PI * alpha * F_SQRT(F_MAX(T1(0), 1.0 - x * x));
            kaiser[i] = bessel_i0(arg) / i0_alpha;
        }
        
        // Cumulative sum
        std::vector<T1> cum_sum(N2 + 1);
        cum_sum[0] = 0;
        for (int i = 1; i <= N2; i++) {
            cum_sum[i] = cum_sum[i - 1] + kaiser[i - 1];
        }
        T1 total = cum_sum[N2] + kaiser[N2];
        
        // First half: sqrt of normalized cumsum
        for (int i = 0; i < N2; i++) {
            window[i] = F_SQRT(cum_sum[i + 1] / total);
        }
        
        // Second half: mirror for Princen-Bradley
        for (int i = 0; i < N2; i++) {
            window[N - 1 - i] = window[i];
        }
    }
    
    inline T1 bessel_i0(const T1 x) {
        T1 sum = 1.0;
        T1 term = 1.0;
        T1 m = 1.0;
        
        while (term > 1e-12 * sum) {
            T1 y = x / (2.0 * m);
            term *= y * y;
            sum += term;
            m += 1.0;
        }
        return sum;
    }

};

#endif // MCLT_HAS_NEON
