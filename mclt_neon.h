/*-----------------------------------------------------------------------------*\
**   MCLTReal v2.0 -- Modulated complex lapped forward / inverse transform     **
**   (c) 2021-2025 Dmitry Boldyrev. All rights reserved.                       **
**                                                                             **
**   mclt_neon.h - Do not include this file directly, instead include mclt.h   **
**                                                                             **
**   FFT MCLT Bundle v2.0 -- PUBLIC EDITION --                                 **
**                                                                             **
**    Features include:                                                        **
**        - DCT-IV/DST-IV based complex MCLT with orthonormal scaling          **
**        - FFT-accelerated transforms via N/2-point complex FFT algorithm     **
**        - Supports arbitrary HOP sizes (optimized for M, M/2, M/4, M/8)      **
**        - Accumulator-based overlap-add for flexible reconstruction          **
**        - Multiple window types: Sine, Kaiser-Bessel, Hanning, Vorbis        **
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

#include <arm_neon.h>
#define MCLT_BEAM_HAS_NEON

template<typename T>
class MCLTReal {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    static constexpr size_t ALIGNMENT = 128;
    
    int _length;
    int _M;
    int _hop;
    WindowType _window_type;
    
    struct AlignedDeleterT { void operator()(T* ptr) const { free(ptr); } };
    struct AlignedDeleterT1 { void operator()(T1* ptr) const { free(ptr); } };
    struct AlignedDeleterCmplxTT { void operator()(cmplxTT* ptr) const { free(ptr); } };
    
    std::unique_ptr<T[], AlignedDeleterT> _current;
    std::unique_ptr<T[], AlignedDeleterT> _overlap_buf;
    std::unique_ptr<cmplxTT[], AlignedDeleterCmplxTT> _temp_complex;
    std::unique_ptr<T[], AlignedDeleterT> _temp_real;
    std::unique_ptr<T[], AlignedDeleterT> _u_buf;
    std::unique_ptr<T[], AlignedDeleterT> _v_buf;
    std::unique_ptr<T1[], AlignedDeleterT1> _window;
    
    FFTReal<T> _fft;  // Size M/2 for DCT-IV FFT
    
    // ============================================================================
    // MEMORY ALLOCATION
    // ============================================================================
    
    static T* aligned_alloc_T(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T)) != 0)
            return nullptr;
        return static_cast<T*>(ptr);
    }
    
    static cmplxTT* aligned_alloc_cmplxTT(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(cmplxTT)) != 0)
            return nullptr;
        return static_cast<cmplxTT*>(ptr);
    }
    
    static T1* aligned_alloc_T1(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T1)) != 0)
            return nullptr;
        return static_cast<T1*>(ptr);
    }

    // ============================================================================
    // NEON OPTIMIZATIONS
    // ============================================================================

#ifdef MCLT_HAS_NEON

    // ----------------------------------------------------------------------------
    // Base type generators (f1, d1)
    // ----------------------------------------------------------------------------

    #define DEFINE_VADD_INPLACE_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VADD) \
    inline void vadd_inplace_##SUFFIX(SCALAR_T* dst, const SCALAR_T* src, int length) { \
        int i = 0; \
        for (; i + (LANES * 2) <= length; i += (LANES * 2)) { \
            VEC_T d0 = VLD(dst + i); \
            VEC_T d1 = VLD(dst + i + LANES); \
            VEC_T s0 = VLD(src + i); \
            VEC_T s1 = VLD(src + i + LANES); \
            VST(dst + i, VADD(d0, s0)); \
            VST(dst + i + LANES, VADD(d1, s1)); \
        } \
        for (; i + LANES <= length; i += LANES) { \
            VST(dst + i, VADD(VLD(dst + i), VLD(src + i))); \
        } \
        for (; i < length; i++) { \
            dst[i] += src[i]; \
        } \
    }

    #define DEFINE_APPLY_WINDOW_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL) \
    inline void apply_window_##SUFFIX( \
        const SCALAR_T* __restrict in, \
        const SCALAR_T* __restrict window, \
        SCALAR_T* __restrict out, \
        int length) \
    { \
        int i = 0; \
        for (; i + (LANES * 2) <= length; i += (LANES * 2)) { \
            VEC_T v_in0 = VLD(in + i); \
            VEC_T v_in1 = VLD(in + i + LANES); \
            VEC_T v_win0 = VLD(window + i); \
            VEC_T v_win1 = VLD(window + i + LANES); \
            VST(out + i, VMUL(v_in0, v_win0)); \
            VST(out + i + LANES, VMUL(v_in1, v_win1)); \
        } \
        for (; i + LANES <= length; i += LANES) { \
            VST(out + i, VMUL(VLD(in + i), VLD(window + i))); \
        } \
        for (; i < length; i++) { \
            out[i] = in[i] * window[i]; \
        } \
    }

    #define DEFINE_APPLY_SCALE_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL, VDUP) \
    inline void apply_scale_##SUFFIX( \
        const SCALAR_T* __restrict x, \
        SCALAR_T* __restrict out, \
        SCALAR_T scale, \
        int N) \
    { \
        int i = 0; \
        VEC_T vscale = VDUP(scale); \
        for (; i + (LANES * 2) <= N; i += (LANES * 2)) { \
            VEC_T x0 = VLD(x + i); \
            VEC_T x1 = VLD(x + i + LANES); \
            VST(out + i, VMUL(x0, vscale)); \
            VST(out + i + LANES, VMUL(x1, vscale)); \
        } \
        for (; i + LANES <= N; i += LANES) { \
            VST(out + i, VMUL(VLD(x + i), vscale)); \
        } \
        for (; i < N; i++) { \
            out[i] = x[i] * scale; \
        } \
    }

    #define DEFINE_APPLY_SCALED_WINDOW_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL, VDUP) \
    inline void apply_scaled_window_##SUFFIX( \
        const SCALAR_T* __restrict x, \
        const SCALAR_T* __restrict h, \
        SCALAR_T* __restrict out, \
        SCALAR_T scale, \
        int N) \
    { \
        int i = 0; \
        VEC_T vscale = VDUP(scale); \
        for (; i + (LANES * 2) <= N; i += (LANES * 2)) { \
            VEC_T x0 = VLD(x + i); \
            VEC_T x1 = VLD(x + i + LANES); \
            VEC_T h0 = VLD(h + i); \
            VEC_T h1 = VLD(h + i + LANES); \
            VST(out + i, VMUL(VMUL(x0, h0), vscale)); \
            VST(out + i + LANES, VMUL(VMUL(x1, h1), vscale)); \
        } \
        for (; i + LANES <= N; i += LANES) { \
            VST(out + i, VMUL(VMUL(VLD(x + i), VLD(h + i)), vscale)); \
        } \
        for (; i < N; i++) { \
            out[i] = x[i] * h[i] * scale; \
        } \
    }

    #define DEFINE_ALL_BASE_FUNCS(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL, VADD, VDUP) \
        DEFINE_VADD_INPLACE_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VADD) \
        DEFINE_APPLY_WINDOW_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL) \
        DEFINE_APPLY_SCALE_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL, VDUP) \
        DEFINE_APPLY_SCALED_WINDOW_BASE(SUFFIX, VEC_T, SCALAR_T, LANES, VLD, VST, VMUL, VDUP)

    // Generate f1 and d1
    DEFINE_ALL_BASE_FUNCS(f1, float32x4_t, float,  4, vld1q_f32, vst1q_f32, vmulq_f32, vaddq_f32, vdupq_n_f32)
    DEFINE_ALL_BASE_FUNCS(d1, float64x2_t, double, 2, vld1q_f64, vst1q_f64, vmulq_f64, vaddq_f64, vdupq_n_f64)

    // ----------------------------------------------------------------------------
    // SIMD vector type generators (f2, f4, f8, d2, d4, d8)
    // ----------------------------------------------------------------------------

    #define DEFINE_SIMD_VADD_INPLACE(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VADD) \
    inline void vadd_inplace_##SUFFIX(SIMD_T* dst, const SIMD_T* src, int N) { \
        SCALAR_T* dstf = reinterpret_cast<SCALAR_T*>(dst); \
        const SCALAR_T* srcf = reinterpret_cast<const SCALAR_T*>(src); \
        constexpr int VLANES = ELEM_COUNT / VECS_PER_ELEM; \
        for (int i = 0; i < N; i++) { \
            SCALAR_T* di = dstf + i * ELEM_COUNT; \
            const SCALAR_T* si = srcf + i * ELEM_COUNT; \
            _Pragma("unroll") \
            for (int v = 0; v < VECS_PER_ELEM; v++) { \
                VST(di + v * VLANES, VADD(VLD(di + v * VLANES), VLD(si + v * VLANES))); \
            } \
        } \
    }

    #define DEFINE_SIMD_APPLY_WINDOW(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VDUP) \
    inline void apply_window_##SUFFIX( \
        const SIMD_T* __restrict x, \
        const SCALAR_T* __restrict h, \
        SIMD_T* __restrict out, \
        int N) \
    { \
        const SCALAR_T* xf = reinterpret_cast<const SCALAR_T*>(x); \
        SCALAR_T* outf = reinterpret_cast<SCALAR_T*>(out); \
        constexpr int VLANES = ELEM_COUNT / VECS_PER_ELEM; \
        for (int i = 0; i < N; i++) { \
            const SCALAR_T* xi = xf + i * ELEM_COUNT; \
            SCALAR_T* oi = outf + i * ELEM_COUNT; \
            VEC_T hv = VDUP(h[i]); \
            _Pragma("unroll") \
            for (int v = 0; v < VECS_PER_ELEM; v++) { \
                VST(oi + v * VLANES, VMUL(VLD(xi + v * VLANES), hv)); \
            } \
        } \
    }

    #define DEFINE_SIMD_APPLY_SCALE(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VDUP) \
    inline void apply_scale_##SUFFIX( \
        const SIMD_T* __restrict x, \
        SIMD_T* __restrict out, \
        SCALAR_T scale, \
        int N) \
    { \
        const SCALAR_T* xf = reinterpret_cast<const SCALAR_T*>(x); \
        SCALAR_T* outf = reinterpret_cast<SCALAR_T*>(out); \
        VEC_T vscale = VDUP(scale); \
        constexpr int VLANES = ELEM_COUNT / VECS_PER_ELEM; \
        for (int i = 0; i < N; i++) { \
            const SCALAR_T* xi = xf + i * ELEM_COUNT; \
            SCALAR_T* oi = outf + i * ELEM_COUNT; \
            _Pragma("unroll") \
            for (int v = 0; v < VECS_PER_ELEM; v++) { \
                VST(oi + v * VLANES, VMUL(VLD(xi + v * VLANES), vscale)); \
            } \
        } \
    }

    #define DEFINE_SIMD_APPLY_SCALED_WINDOW(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VDUP) \
    inline void apply_scaled_window_##SUFFIX( \
        const SIMD_T* __restrict x, \
        const SCALAR_T* __restrict h, \
        SIMD_T* __restrict out, \
        SCALAR_T scale, \
        int N) \
    { \
        const SCALAR_T* xf = reinterpret_cast<const SCALAR_T*>(x); \
        SCALAR_T* outf = reinterpret_cast<SCALAR_T*>(out); \
        VEC_T vscale = VDUP(scale); \
        constexpr int VLANES = ELEM_COUNT / VECS_PER_ELEM; \
        for (int i = 0; i < N; i++) { \
            const SCALAR_T* xi = xf + i * ELEM_COUNT; \
            SCALAR_T* oi = outf + i * ELEM_COUNT; \
            VEC_T sh = VMUL(VDUP(h[i]), vscale); \
            _Pragma("unroll") \
            for (int v = 0; v < VECS_PER_ELEM; v++) { \
                VST(oi + v * VLANES, VMUL(VLD(xi + v * VLANES), sh)); \
            } \
        } \
    }

    #define DEFINE_ALL_SIMD_FUNCS(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VADD, VDUP) \
        DEFINE_SIMD_VADD_INPLACE(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VADD) \
        DEFINE_SIMD_APPLY_WINDOW(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VDUP) \
        DEFINE_SIMD_APPLY_SCALE(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VDUP) \
        DEFINE_SIMD_APPLY_SCALED_WINDOW(SUFFIX, SIMD_T, VEC_T, SCALAR_T, ELEM_COUNT, VECS_PER_ELEM, VLD, VST, VMUL, VDUP)

    // Generate f2, f4, f8
    DEFINE_ALL_SIMD_FUNCS(f2, simd_float2,  float32x2_t, float,  2, 1, vld1_f32,  vst1_f32,  vmul_f32,  vadd_f32,  vdup_n_f32)
    DEFINE_ALL_SIMD_FUNCS(f4, simd_float4,  float32x4_t, float,  4, 1, vld1q_f32, vst1q_f32, vmulq_f32, vaddq_f32, vdupq_n_f32)
    DEFINE_ALL_SIMD_FUNCS(f8, simd_float8,  float32x4_t, float,  8, 2, vld1q_f32, vst1q_f32, vmulq_f32, vaddq_f32, vdupq_n_f32)

    // Generate d2, d4, d8
    DEFINE_ALL_SIMD_FUNCS(d2, simd_double2, float64x2_t, double, 2, 1, vld1q_f64, vst1q_f64, vmulq_f64, vaddq_f64, vdupq_n_f64)
    DEFINE_ALL_SIMD_FUNCS(d4, simd_double4, float64x2_t, double, 4, 2, vld1q_f64, vst1q_f64, vmulq_f64, vaddq_f64, vdupq_n_f64)
    DEFINE_ALL_SIMD_FUNCS(d8, simd_double8, float64x2_t, double, 8, 4, vld1q_f64, vst1q_f64, vmulq_f64, vaddq_f64, vdupq_n_f64)

    // ----------------------------------------------------------------------------
    // Dispatch functions
    // ----------------------------------------------------------------------------

    void vadd_inplace_dispatch(T* dst, const T* src, int len) {
        if constexpr (std::is_same_v<T, float>) {
            vadd_inplace_f1(dst, src, len);
        } else if constexpr (std::is_same_v<T, double>) {
            vadd_inplace_d1(dst, src, len);
        } else if constexpr (std::is_same_v<T, simd_float2>) {
            vadd_inplace_f2(dst, src, len);
        } else if constexpr (std::is_same_v<T, simd_float4>) {
            vadd_inplace_f4(dst, src, len);
        } else if constexpr (std::is_same_v<T, simd_float8>) {
            vadd_inplace_f8(dst, src, len);
        } else if constexpr (std::is_same_v<T, simd_double2>) {
            vadd_inplace_d2(dst, src, len);
        } else if constexpr (std::is_same_v<T, simd_double4>) {
            vadd_inplace_d4(dst, src, len);
        } else if constexpr (std::is_same_v<T, simd_double8>) {
            vadd_inplace_d8(dst, src, len);
        } else {
            for (int i = 0; i < len; i++) dst[i] += src[i];
        }
    }

#else // No NEON

    void vadd_inplace_dispatch(T* dst, const T* src, int len) {
        for (int i = 0; i < len; i++) dst[i] += src[i];
    }

#endif // MCLT_HAS_NEON

    // ============================================================================
    // WINDOW GENERATION
    // ============================================================================

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
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    T1 s = F_SIN((i + 0.5) * K);
                    _window[i] = s * s;
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
                    T1 t = 2.0 * M_PI * (i + 0.5) / T1(N);
                    _window[i] = 0.42 - 0.5 * F_COS(t) + 0.08 * F_COS(2.0 * t);
                }
                break;
            }
            default: {
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    _window[i] = F_SIN((i + 0.5) * K);
                }
                break;
            }
        }
    }
    
    static void generate_kbd_window(T1* window, int length, T1 alpha = 4.0) {
        const int N = length;
        const int N2 = N / 2;
        
        T1* kaiser = new T1[N2 + 1];
        T1 i0_alpha = bessel_i0(M_PI * alpha);
        
        for (int i = 0; i <= N2; i++) {
            T1 x = (2.0 * i / T1(N2)) - 1.0;
            T1 arg = M_PI * alpha * F_SQRT(F_MAX(T1(0), 1.0 - x * x));
            kaiser[i] = bessel_i0(arg) / i0_alpha;
        }
        
        T1* cum_sum = new T1[N2 + 1];
        cum_sum[0] = 0;
        for (int i = 1; i <= N2; i++) {
            cum_sum[i] = cum_sum[i - 1] + kaiser[i - 1];
        }
        T1 total = cum_sum[N2] + kaiser[N2];
        
        for (int i = 0; i < N2; i++) {
            window[i] = F_SQRT(cum_sum[i + 1] / total);
        }
        
        for (int i = 0; i < N2; i++) {
            window[N - 1 - i] = window[i];
        }
        
        delete[] kaiser;
        delete[] cum_sum;
    }
    
    static T1 bessel_i0(T1 x) {
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

public:
    
    // ============================================================================
    // CONSTRUCTOR / RESET
    // ============================================================================
    
    explicit MCLTReal(int length, int hop = -1, WindowType window_type = WINTYPE_SINE)
        : _length(length)
        , _M(length / 2)
        , _hop(hop < 0 ? _M : hop)
        , _window_type(window_type)
        , _fft(length / 4)  // M/2 for DCT-IV FFT
        , _current(aligned_alloc_T(length))
        , _overlap_buf(aligned_alloc_T(length))
        , _u_buf(aligned_alloc_T(_M))
        , _v_buf(aligned_alloc_T(_M))
        , _temp_complex(aligned_alloc_cmplxTT(2 * _M))
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
        memset(_current.get(), 0, _length * sizeof(T));
        memset(_overlap_buf.get(), 0, _length * sizeof(T));
        memset(_u_buf.get(), 0, _M * sizeof(T));
        memset(_v_buf.get(), 0, _M * sizeof(T));
        memset(_temp_real.get(), 0, (6 * _length) * sizeof(T));
    }
    
    // ============================================================================
    // GETTERS
    // ============================================================================
    
    inline const T* get_current_frame() const { return _current.get(); }
    inline int get_length() const { return _length; }
    inline int get_half_length() const { return _M; }
    inline int get_M() const { return _M; }
    inline int get_hop() const { return _hop; }
    inline const T1* window() const { return _window.get(); }
    inline WindowType get_window_type() const { return _window_type; }
    
    inline void get_output(T* output) {
        memcpy(output, _current.get(), _length * sizeof(T));
    }
    
    // ============================================================================
    // SETTERS
    // ============================================================================
    
    void set_hop(int hop) {
        if (hop <= 0 || hop > _length) {
            throw std::invalid_argument("Hop size must be between 1 and 2M");
        }
        _hop = hop;
        reset();
    }
    
    void set_window_type(WindowType type) {
        _window_type = type;
        init_window();
    }
    
    // ============================================================================
    // FORWARD MCLT (Analysis)
    // ============================================================================
    
    void analyze(const T* __restrict__ x, cmplxTT* __restrict__ X, bool apply_window = true) {
        T* __restrict__ u = _u_buf.get();
        T* __restrict__ v = _v_buf.get();
        const T1* __restrict__ h = _window.get();
        
        const int M = _M;
        const int M_half = M / 2;
        const int M_three_halves = 3 * M_half;
        
        if (apply_window) {
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
        } else {
            for (int n = 0; n < M; n++) {
                u[n] = x[M_three_halves - n - 1];
            }
            memcpy(v, u, M * sizeof(T));
            
            for (int n = 0; n < M_half; n++) {
                const T t = x[M_three_halves + n];
                u[n] += t;
                v[n] -= t;
            }
            
            for (int n = 0; n < M_half; n++) {
                const T t = x[n];
                u[n + M_half] -= t;
                v[n + M_half] += t;
            }
        }
        
        do_dct_iv_fft(u, M);
        do_dst_iv_fft(v, M);
        
        for (int k = 0; k < M; k++) {
            X[k].re = u[k];
            X[k].im = v[k];
        }
    }

    // ============================================================================
    // INVERSE MCLT (Synthesis)
    // ============================================================================
    
    void synthesize(const cmplxTT* __restrict__ X, bool apply_window = true) {
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
        
        const T1 scale = T1(0.5) * T1(_hop) / T1(_M);
        
        if (apply_window) {
            for (int i = 0; i < M_half; i++) {
                current[i] = h[i] * (v[M_half + i] - u[M_half + i]) * scale;
            }
            for (int i = M_half; i < M_three_halves; i++) {
                current[i] = h[i] * (u[M_three_halves - i - 1] + v[M_three_halves - i - 1]) * scale;
            }
            for (int i = M_three_halves; i < L; i++) {
                current[i] = h[i] * (u[i - M_three_halves] - v[i - M_three_halves]) * scale;
            }
        } else {
            for (int i = 0; i < M_half; i++) {
                current[i] = (v[M_half + i] - u[M_half + i]) * scale;
            }
            for (int i = M_half; i < M_three_halves; i++) {
                current[i] = (u[M_three_halves - i - 1] + v[M_three_halves - i - 1]) * scale;
            }
            for (int i = M_three_halves; i < L; i++) {
                current[i] = (u[i - M_three_halves] - v[i - M_three_halves]) * scale;
            }
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
    
    // ============================================================================
    // OVERLAP-ADD
    // ============================================================================
    
    void overlap_add_to_buffer(T* __restrict__ output) {
        overlap_add_to_buffer(output, _hop);
    }

    void overlap_add_to_buffer(T* __restrict__ output, int hop) {
        const int L = 2 * _M;
        
        // 1. Add current frame to overlap buffer
        vadd_inplace_dispatch(_overlap_buf.get(), _current.get(), L);
        
        // 2. Output first hop samples
        memcpy(output, _overlap_buf.get(), hop * sizeof(T));
        
        // 3. Shift buffer left by hop
        memmove(_overlap_buf.get(), _overlap_buf.get() + hop, (L - hop) * sizeof(T));
        memset(_overlap_buf.get() + L - hop, 0, hop * sizeof(T));
    }
    
    // ============================================================================
    // API ALIASES
    // ============================================================================
    
    void real_mclt(const T* in, cmplxTT* out, bool apply_window = true) {
        analyze(in, out, apply_window);
    }
    
    void mclt(const T* in, cmplxTT* out, bool apply_window = true) {
        analyze(in, out, apply_window);
    }

    void real_imclt(const cmplxTT* in, T* out, bool apply_window = true) {
        synthesize(in, apply_window);
        overlap_add_to_buffer(out);
    }
    
    void imclt(const cmplxTT* in, T* out, bool apply_window = true) {
        real_imclt(in, out, apply_window);
    }

    void synthesize_no_tdac(const cmplxTT* X, T* output, bool apply_window = true) {
        synthesize(X, apply_window);
        memcpy(output, _current.get(), _length * sizeof(T));
    }

    // ============================================================================
    // UTILITY
    // ============================================================================
    
    void skip_frame() {
        memset(_current.get(), 0, _length * sizeof(T));
        memmove(_overlap_buf.get(), _overlap_buf.get() + _hop, (_length - _hop) * sizeof(T));
        memset(_overlap_buf.get() + _length - _hop, 0, _hop * sizeof(T));
    }
    
    // ============================================================================
    // DCT-IV / DST-IV TRANSFORMS
    // ============================================================================
    
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
        
        // Pack y[n] = x[2n] + j*x[N-1-2n]
        for (int n = 0; n < N2; n++) {
            y[n].re = x[2*n];
            y[n].im = x[N - 1 - 2*n];
        }
        
        // Pre-twiddle by e^{-jπn/N}
        for (int n = 0; n < N2; n++) {
            const T1 angle = -M_PI * T1(n) / T1(N);
            y[n] = y[n] * cmplxTT(F_COS(angle), F_SIN(angle));
        }
        
        // N/2-point complex FFT
        _fft.do_fft_complex(y, fft_temp);
        
        // Post-twiddle by e^{-jπ(2k+0.5)/(2N)}
        for (int k = 0; k < N2; k++) {
            const T1 angle = -M_PI * T1(2*k + 0.5) / T1(2 * N);
            y[k] = y[k] * cmplxTT(F_COS(angle), F_SIN(angle));
        }
        
        // Unpack with scaling
        const T1 scale = F_SQRT(T1(2.0) / T1(N));
        for (int k = 0; k < N2; k++) {
            x[2*k]         =  y[k].re * scale;
            x[N - 1 - 2*k] = -y[k].im * scale;
        }
    }
    
    // Direct transforms for verification
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
};

