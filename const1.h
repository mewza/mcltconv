/*-----------------------------------------------------------------------------*\
**   MCLT Convolver v0.9 -- MCLT-Domain Convolution using MCLTReal             **
**   (c) 2021-2025 Dmitry Boldyrev. All rights reserved.                       **
**                                                                             **
**   const1.h - SimdBase<T>, Simd<T,size>, cmplxT<T>, etc                      **
**                                                                             **
**   FFT MCLT Bundle v2.0 -- PUBLIC EDITION --                                 **
**                                                                             **
**    @contact  E-mail: subband@gmail.com or subband@protonmail.com            **
**    @home https://github.com/mewza/mcltconv                                  **
**                                                                             **
**    This software relies on use of FFTReal class available at:               **
**    https://github.com/mewza/realfft/                                        **
\*-----------------------------------------------------------------------------*/


#pragma once

#include <math.h>
#include <fenv.h>
#include <sys/signal.h>
#include <malloc/malloc.h>
#include <simd/simd.h>
#include "mem.h"

#define DBOOST_USE_NEON

#if TARGET_OS_MACCATALYST
#define OVERFLOW_FIX overflow_fix
#else
#define OVERFLOW_FIX overflow_fix
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#include <simd/simd.h>
#else
#ifdef DBOOST_USE_NEON
#undef DBOOST_USE_NEON
#endif
#endif




typedef float mssFloat;
#define MSSFLOAT_IS_FLOAT (sizeof(mssFloat) == sizeof(float))

#define TWO_PI          (2 * M_PI)

#define MAX_FRAME_SIZE  2048

#ifdef __cplusplus

#include <type_traits>
#include <algorithm>
#include <map>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <functional>


template <typename> struct cmplxT;
// assume you already have IsVector<T> elsewhere; if not, define a placeholder:
template<typename T> struct IsVector : std::false_type {};
// e.g. specialize IsVector<simd_float8> : std::true_type {};

// detect complex-like wrapper by presence of .re member
template<typename, typename = void>
struct HasMemberRe : std::false_type {};
template<typename T>
struct HasMemberRe<T, std::void_t<decltype(std::declval<T&>().re)>> : std::true_type {};

// optional helper to detect "is cmplxT" if you prefer explicit check
template<typename> struct IsCmplxT : std::false_type {};
template<typename U> struct IsCmplxT<cmplxT<U>> : std::true_type {};

static inline constexpr int alignFor(int typeSize, int nelem) {
   // if (typeSize > 2)
     //   return 16;
    return typeSize * nelem;
}

// detect operator[] (indexable)
template<typename, typename = void>
struct HasIndex : std::false_type {};

template<typename T>
struct HasIndex<T, std::void_t<decltype(std::declval<T&>()[0])>> : std::true_type {};

// Helper that picks the right element type based on boolean flags.
// Primary template: fallback to T itself.
template<typename T, bool HasReFlag, bool HasIdxFlag>
struct ElementChoice {
    using type = T;
};

// Specialization when T has .re (highest priority)
template<typename T>
struct ElementChoice<T, true, false> {
    using type = std::decay_t<decltype(std::declval<T&>().re)>;
};

// If T has both .re and operator[], we still want .re to win:
template<typename T>
struct ElementChoice<T, true, true> {
    using type = std::decay_t<decltype(std::declval<T&>().re)>;
};

// Specialization when T has operator[] but no .re
template<typename T>
struct ElementChoice<T, false, true> {
    using type = std::decay_t<decltype(std::declval<T&>()[0])>;
};

// Public helper that evaluates the flags and selects the matching ElementChoice specialization.
template<typename T>
struct ElementOrSelf {
    static constexpr bool hasRe  = HasMemberRe<T>::value;
    static constexpr bool hasIdx = HasIndex<T>::value;
    using type = typename ElementChoice<T, hasRe, hasIdx>::type;
};

// SimdBase: keep arithmetic types as themselves, otherwise use ElementOrSelf
template<typename ZZ>
using SimdBase = std::conditional_t<
    std::is_arithmetic_v<ZZ>,
    ZZ,
    typename ElementOrSelf<ZZ>::type
>;

template<typename ZZ> struct SimdInfo {
    using Base = SimdBase<ZZ>;
    static constexpr int size = sizeof(ZZ) / sizeof(Base);
};

template <class ZZ>
inline constexpr int SimdSize = SimdInfo<ZZ>::size;

template<typename Z, int size>
using Simd = Z __attribute__((ext_vector_type(size),aligned(alignFor(sizeof(Z),size))));

template<typename ZZ, typename NewBase>
using SimdSame = Simd<NewBase, SimdInfo<ZZ>::size>;

template<typename ZZ>
using SimdSameHalf = Simd<SimdBase<ZZ>, SimdInfo<ZZ>::size/2>;

template<typename Z>
constexpr bool is_vector_v = IsVector<Z>::value;

#define NOT_VECTOR(Z) (std::is_same_v<Z, float> || std::is_same_v<Z, double>)
#define IS_VECTOR(Z)  (is_vector_v<Z>)

template <typename Z>
auto SimdMake = [](auto a, auto b) -> Z{
    using std::is_same_v;
    using base = SimdBase<Z>;
    switch (SimdSize<Z>) {
        case 8:
            if constexpr( std::is_same_v<base, float> )
                return simd_make_float8(a, b);
            else return simd_make_double8(a, b);
            break;
        case 4:
            if constexpr( std::is_same_v<base, float> )
                return simd_make_float8(a, b);
            else return simd_make_double8(a, b);
            break;
        case 2:
            if constexpr( std::is_same_v<base, float> )
                return simd_make_float2(a, b);
            else return simd_make_double2(a, b);
            break;
        default:
            return Z{0};
    }
};


// Template-based converter that works for scalars and all vector sizes
template<typename ToVec, typename FromVec>
static inline ToVec convertvector_safe(FromVec from) {
    using ToBase = SimdBase<ToVec>;
    using FromBase = SimdBase<FromVec>;
    
    // Check if ToVec is a scalar type
    constexpr bool ToIsScalar = std::is_same_v<ToVec, float> ||
                                std::is_same_v<ToVec, double> ||
                                std::is_same_v<ToVec, int> ||
                                std::is_same_v<ToVec, long>;
    
    constexpr bool FromIsScalar = std::is_same_v<FromVec, float> ||
                                  std::is_same_v<FromVec, double> ||
                                  std::is_same_v<FromVec, int> ||
                                  std::is_same_v<FromVec, long>;
    
    if constexpr (ToIsScalar && FromIsScalar) {
        // Scalar to scalar
        return static_cast<ToVec>(from);
    }
    else if constexpr (ToIsScalar && !FromIsScalar) {
        // Vector to scalar - take first element
        return static_cast<ToVec>(from[0]);
    }
    else if constexpr (!ToIsScalar && FromIsScalar) {
        // Scalar to vector - broadcast
        return (ToVec)from;
    }
    else {
        // Vector to vector
        constexpr int N = SimdSize<ToVec>;
        
        if constexpr (N >= 4) {
            return __builtin_convertvector(from, ToVec);
        } else {
            // Manual conversion for small vectors (size 1, 2, 3)
            ToVec result;
            for (int i = 0; i < N; i++) {
                result[i] = static_cast<ToBase>(from[i]);
            }
            return result;
        }
    }
}

template <typename T>
static inline T maskL() {
    static const int nch = SimdSize<T>;
    if constexpr(nch == 2)
        return T{1,0};
    else if constexpr(nch == 4)
        return T{1,0,1,0};
    else if constexpr(nch == 8)
        return T{1,0,1,0,1,0,1,0};
    else
        return T{1};
}

template <typename T>
static inline T maskR() {
    static const int nch = SimdSize<T>;
    if constexpr(nch == 2)
        return T{0,1};
    else if constexpr(nch == 4)
        return T{0,1,0,1};
    else if constexpr(nch == 8)
        return T{0,1,0,1,0,1,0,1};
    else
        return T{0};
}


typedef simd_float8 mssFloat8;
typedef simd_float4 mssFloat4;
typedef simd_float2 mssFloat2;

typedef simd_float2 float2v;
typedef simd_float4 float4v;
typedef simd_float8 float8v;

typedef simd_double2 double2v;
typedef simd_double4 double4v;
typedef simd_double8 double8v;

typedef simd_long8 long8v;
typedef simd_long4 long4v;
typedef simd_long2 long2v;

typedef simd_int2 int2v;
typedef simd_int4 int4v;
typedef simd_int8 int8v;

typedef simd_uint2 uint2v;
typedef simd_uint4 uint4v;
typedef simd_uint8 uint8v;

typedef Simd<int, 1> int1v;
typedef Simd<float, 1> float1v;
typedef Simd<double, 1> double1v;

// #define shufflevector(a, b, ...) shufflevector_safe<__VA_ARGS__>((a), (b))
// would require changing format: shufflevector(sum, int8v{0,0,2,2,4,4,6,6}, 1, 1, 3, 3, 5, 5, 7, 7)

#define shufflevector __builtin_shufflevector
#define convertvector(VecA, VecB) convertvector_safe<VecB>(VecA)

template<typename T>
auto stereo_diff_impl(const T& w) {
    if constexpr (SimdSize<T> == 8)
        return shufflevector(w, w, 0, 2, 4, 6) - shufflevector(w, w, 1, 3, 5, 7);
    else if constexpr (SimdSize<T> == 4)
        return shufflevector(w, w, 0, 2) - shufflevector(w, w, 1, 3);
}

// Template-based shufflevector that works for all vector sizes
template<int... Indices, typename Vec>
static inline auto shufflevector_safe(Vec a, Vec b) {
    constexpr int OutSize = sizeof...(Indices);
    using Base = SimdBase<Vec>;
    using OutVec = Simd<Base, OutSize>;
    constexpr int InSize = SimdSize<Vec>;
    
    // Check if input is scalar
    constexpr bool IsScalar = std::is_arithmetic_v<Vec>;
    if constexpr (IsScalar) {
        // Scalar input - broadcast to output
        OutVec result;
        for (int i = 0; i < OutSize; i++) {
            result[i] = a;
        }
        return result;
    }
    else if constexpr (InSize >= 4 && OutSize >= 4) {
        // Use builtin for larger vectors
        return __builtin_shufflevector(a, b, Indices...);
    }
    else {
        // Manual shuffle for small vectors
        constexpr int indices[] = {Indices...};
        OutVec result;
        for (int i = 0; i < OutSize; i++) {
            int idx = indices[i];
            if (idx < InSize) {
                result[i] = a[idx];
            } else {
                result[i] = b[idx - InSize];
            }
        }
        return result;
    }
}

// Usage examples:
// Swizzle elements:  auto result = shufflevector_safe<1, 0>(vec, vec);
// Select from two:   auto result = shufflevector_safe<0, 3>(vec1, vec2);
// Broadcast element: auto result = shufflevector_safe<0, 0, 0, 0>(vec, vec);

#ifndef D_ZFLOAT
#define D_ZFLOAT

#define DECL_ZFLOAT(TYPE) \
typedef TYPE zfloat; \
typedef simd_##TYPE##8 zfloat8; \
typedef simd_##TYPE##4 zfloat4; \
typedef simd_##TYPE##2 zfloat2; \
static inline zfloat8 make_zfloat8(zfloat4& a, zfloat4& b) { return simd_make_##TYPE##8(a,b); } \
static inline zfloat4 make_zfloat4(zfloat2& a, zfloat2& b) { return simd_make_##TYPE##4(a,b); } \
static inline zfloat2 make_zfloat2(zfloat& a, zfloat& b) { return simd_make_##TYPE##2(a,b); }

DECL_ZFLOAT(double)

#endif

#define ZFLOAT_IS_FLOAT constexpr(sizeof(zfloat) == sizeof(float))
#define FUNDEMENTAL(T) std::is_same_v<T, float> || std::is_same_v<T, double>

// Optimized scatter/gather for ARM NEON (Apple Silicon)


template<class T, size_t N, typename idx_t = int> struct ZArray
{
    using IT = SimdSame<T,idx_t>;
    static const int SIZE = SimdSize<T>;
public:
    MEM_ALIGN T dd[N];
    
    ZArray() {
        reset();
    }
    
    void reset() {
        memset(dd, 0, sizeof(dd));
    }
    
    class ZProxy
    {
    public:
        ZArray &a;
        IT ii;
        
        ZProxy(ZArray &a, const IT& i) : a(a), ii(i) {}
       
#define ACCESS_T_NEON \
        [[gnu::hot, gnu::always_inline]] \
        inline operator T() const \
        { \
            if constexpr (SIZE == 8 && std::is_same_v<T, uint8x8_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                __builtin_prefetch(&a.dd[ii[4]][4], 0, 3); \
                __builtin_prefetch(&a.dd[ii[5]][5], 0, 3); \
                __builtin_prefetch(&a.dd[ii[6]][6], 0, 3); \
                __builtin_prefetch(&a.dd[ii[7]][7], 0, 3); \
                uint8x8_t result = vdup_n_u8(0); \
                result = vset_lane_u8(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_u8(a.dd[ii[1]][1], result, 1); \
                result = vset_lane_u8(a.dd[ii[2]][2], result, 2); \
                result = vset_lane_u8(a.dd[ii[3]][3], result, 3); \
                result = vset_lane_u8(a.dd[ii[4]][4], result, 4); \
                result = vset_lane_u8(a.dd[ii[5]][5], result, 5); \
                result = vset_lane_u8(a.dd[ii[6]][6], result, 6); \
                result = vset_lane_u8(a.dd[ii[7]][7], result, 7); \
                return result; \
            } \
            else if constexpr (SIZE == 8 && std::is_same_v<T, int8x8_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                __builtin_prefetch(&a.dd[ii[4]][4], 0, 3); \
                __builtin_prefetch(&a.dd[ii[5]][5], 0, 3); \
                __builtin_prefetch(&a.dd[ii[6]][6], 0, 3); \
                __builtin_prefetch(&a.dd[ii[7]][7], 0, 3); \
                int8x8_t result = vdup_n_s8(0); \
                result = vset_lane_s8(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_s8(a.dd[ii[1]][1], result, 1); \
                result = vset_lane_s8(a.dd[ii[2]][2], result, 2); \
                result = vset_lane_s8(a.dd[ii[3]][3], result, 3); \
                result = vset_lane_s8(a.dd[ii[4]][4], result, 4); \
                result = vset_lane_s8(a.dd[ii[5]][5], result, 5); \
                result = vset_lane_s8(a.dd[ii[6]][6], result, 6); \
                result = vset_lane_s8(a.dd[ii[7]][7], result, 7); \
                return result; \
            } \
            else if constexpr (SIZE == 4 && std::is_same_v<T, float32x4_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                float32x4_t result = vdupq_n_f32(0); \
                result = vsetq_lane_f32(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_f32(a.dd[ii[1]][1], result, 1); \
                result = vsetq_lane_f32(a.dd[ii[2]][2], result, 2); \
                result = vsetq_lane_f32(a.dd[ii[3]][3], result, 3); \
                return result; \
            } \
            else if constexpr (SIZE == 4 && std::is_same_v<T, int32x4_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                int32x4_t result = vdupq_n_s32(0); \
                result = vsetq_lane_s32(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_s32(a.dd[ii[1]][1], result, 1); \
                result = vsetq_lane_s32(a.dd[ii[2]][2], result, 2); \
                result = vsetq_lane_s32(a.dd[ii[3]][3], result, 3); \
                return result; \
            } \
            else if constexpr (SIZE == 4 && std::is_same_v<T, uint32x4_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                uint32x4_t result = vdupq_n_u32(0); \
                result = vsetq_lane_u32(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_u32(a.dd[ii[1]][1], result, 1); \
                result = vsetq_lane_u32(a.dd[ii[2]][2], result, 2); \
                result = vsetq_lane_u32(a.dd[ii[3]][3], result, 3); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, float32x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                float32x2_t result = vdup_n_f32(0); \
                result = vset_lane_f32(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_f32(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, int32x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                int32x2_t result = vdup_n_s32(0); \
                result = vset_lane_s32(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_s32(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, uint32x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                uint32x2_t result = vdup_n_u32(0); \
                result = vset_lane_u32(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_u32(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, float64x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                float64x2_t result = vdupq_n_f64(0); \
                result = vsetq_lane_f64(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_f64(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else { \
                __builtin_prefetch(&a.dd[ii[0]], 0, 3); \
                return a.dd[ii[0]]; \
            } \
        }
        
#define ACCESS_T \
        [[gnu::hot, gnu::always_inline]] \
        inline operator T() const \
        { \
            if constexpr( SIZE == 8 ) { \
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] }; \
            } else if constexpr( SIZE == 4 ) { \
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] }; \
            } else if constexpr( SIZE == 2 ) { \
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1] }; \
            } else if constexpr( SIZE == 1 ) { \
                return a.dd[ii[0]]; \
            } \
        }
        
#if defined(__aarch64__) && defined(__ARM_NEON)
        ACCESS_T_NEON
        
        [[gnu::hot, gnu::always_inline]]
        T operator = (const T& v) {
            // SIZE == 8
            if constexpr (SIZE == 8 && std::is_same_v<T, uint8x8_t>) {
                // Prefetch for write (1 = write intent, 3 = high temporal locality)
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                __builtin_prefetch(&a.dd[ii[4]][4], 1, 3);
                __builtin_prefetch(&a.dd[ii[5]][5], 1, 3);
                __builtin_prefetch(&a.dd[ii[6]][6], 1, 3);
                __builtin_prefetch(&a.dd[ii[7]][7], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_u8(v, 0);
                a.dd[ii[1]][1] = vget_lane_u8(v, 1);
                a.dd[ii[2]][2] = vget_lane_u8(v, 2);
                a.dd[ii[3]][3] = vget_lane_u8(v, 3);
                a.dd[ii[4]][4] = vget_lane_u8(v, 4);
                a.dd[ii[5]][5] = vget_lane_u8(v, 5);
                a.dd[ii[6]][6] = vget_lane_u8(v, 6);
                a.dd[ii[7]][7] = vget_lane_u8(v, 7);
                return v;
            }
            else if constexpr (SIZE == 8 && std::is_same_v<T, int8x8_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                __builtin_prefetch(&a.dd[ii[4]][4], 1, 3);
                __builtin_prefetch(&a.dd[ii[5]][5], 1, 3);
                __builtin_prefetch(&a.dd[ii[6]][6], 1, 3);
                __builtin_prefetch(&a.dd[ii[7]][7], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_s8(v, 0);
                a.dd[ii[1]][1] = vget_lane_s8(v, 1);
                a.dd[ii[2]][2] = vget_lane_s8(v, 2);
                a.dd[ii[3]][3] = vget_lane_s8(v, 3);
                a.dd[ii[4]][4] = vget_lane_s8(v, 4);
                a.dd[ii[5]][5] = vget_lane_s8(v, 5);
                a.dd[ii[6]][6] = vget_lane_s8(v, 6);
                a.dd[ii[7]][7] = vget_lane_s8(v, 7);
                return v;
            } // SIZE == 4
            else if constexpr (SIZE == 4 && std::is_same_v<T, float32x4_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_f32(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_f32(v, 1);
                a.dd[ii[2]][2] = vgetq_lane_f32(v, 2);
                a.dd[ii[3]][3] = vgetq_lane_f32(v, 3);
                return v;
            }
            else if constexpr (SIZE == 4 && std::is_same_v<T, int32x4_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_s32(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_s32(v, 1);
                a.dd[ii[2]][2] = vgetq_lane_s32(v, 2);
                a.dd[ii[3]][3] = vgetq_lane_s32(v, 3);
                return v;
            }
            else if constexpr (SIZE == 4 && std::is_same_v<T, uint32x4_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_u32(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_u32(v, 1);
                a.dd[ii[2]][2] = vgetq_lane_u32(v, 2);
                a.dd[ii[3]][3] = vgetq_lane_u32(v, 3);
                return v;
            } // SIZE == 2
            else if constexpr (SIZE == 2 && std::is_same_v<T, float32x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_f32(v, 0);
                a.dd[ii[1]][1] = vget_lane_f32(v, 1);
                return v;
            }
            else if constexpr (SIZE == 2 && std::is_same_v<T, int32x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_s32(v, 0);
                a.dd[ii[1]][1] = vget_lane_s32(v, 1);
                return v;
            }
            else if constexpr (SIZE == 2 && std::is_same_v<T, uint32x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_u32(v, 0);
                a.dd[ii[1]][1] = vget_lane_u32(v, 1);
                return v;
            }
            else if constexpr (SIZE == 2 && std::is_same_v<T, float64x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_f64(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_f64(v, 1);
                return v;
            } // SIZE == 1
            else {
                __builtin_prefetch(&a.dd[ii[0]], 1, 3);
                return a.dd[ii[0]] = v;
            }
        }
#else
        [[gnu::hot, gnu::always_inline]]
        inline T operator = (const T& v) {
            if constexpr( SIZE == 8 ) {
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1]; a.dd[ii[2]][2] = v[2]; a.dd[ii[3]][3] = v[3];
                a.dd[ii[4]][4] = v[4]; a.dd[ii[5]][5] = v[5]; a.dd[ii[6]][6] = v[6]; a.dd[ii[7]][7] = v[7];
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] };
            }
            else if constexpr( SIZE == 4 ) {
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1]; a.dd[ii[2]][2] = v[2]; a.dd[ii[3]][3] = v[3];
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] };
            }
            else if constexpr( SIZE == 2 ) {
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1];
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1] };
            }
            else {
                return a.dd[ii[0]] = v;
            }
        }
        ACCESS_T
#endif
    };
    class ZProxyConst
      {
      public:
          const ZArray &a;
          IT ii;
      
          ZProxyConst(const ZArray &a, const IT& i) : a(a), ii(i) {}
#if defined(__aarch64__) && defined(__ARM_NEON)
          ACCESS_T_NEON
#else
          ACCESS_T
#endif
      };
    ZProxyConst operator[] (const IT& i) const { return ZProxyConst(*this, i); }
    ZProxy operator[] (const IT& i) { return ZProxy(*this, i); }
};

typedef std::conditional<sizeof(mssFloat) == sizeof(float), float2v, double2v>::type mssFloat2;
typedef std::conditional<sizeof(mssFloat) == sizeof(float), float4v, double4v>::type mssFloat4;
typedef std::conditional<sizeof(mssFloat) == sizeof(float), float8v, double8v>::type mssFloat8;

static __inline float8v make_mssFloat8(const float4v &a, const float4v &b) {
    return simd_make_float8(a,b);
}

static __inline double8v make_mssFloat8(const double4v &a, const double4v &b) {
    return simd_make_double8(a,b);
}

static __inline float4v make_mssFloat4(const float2v &a, const float2v &b) {
    return simd_make_float4(a,b);
}

static __inline double4v make_mssFloat4(const double2v &a, const double2v &b) {
    return simd_make_double4(a,b);
}

static __inline float2v make_mssFloat2(const float &a, const float &b) {
    return simd_make_float2(a,b);
}

static __inline double2v make_mssFloat2(const double &a, const double &b) {
    return simd_make_double2(a,b);
}

//typedef mssFloat cmplx[2];

#ifndef CMPLX_T_TYPE
#define CMPLX_T_TYPE

template <typename T> struct cmplxT;

#endif // CMPLX_T_TYPE

class const1 {
public:
    mssFloat v;
    constexpr const1 (mssFloat value) : v(value) {}
    inline operator mssFloat2() const {
        return mssFloat2{v, v};
    }
    inline operator mssFloat4() const {
        return mssFloat4{v, v, v, v};
    }
    inline operator mssFloat8() const {
        return mssFloat8{v, v, v, v, v, v, v, v};
    }
    inline operator float() const {
        return v;
    }
    inline constexpr const1 operator-() const { return const1(-v); }
};

class const1i {
public:
    int iv;
    constexpr const1i (int value) : iv(value) {}
    constexpr const1i (unsigned long long value) : iv((int)value) {}
    inline operator int2v() const {
        return int2v{iv, iv};
    }
    inline operator int4v() const {
        return int4v{iv, iv, iv, iv};
    }
    inline operator int8v() const {
        return int8v{iv, iv, iv, iv, iv, iv, iv, iv};
    }
    inline bool operator < (const const1i &x) const {
        return true;
    }
    inline operator unsigned long long() const {
        return iv;
    }
};

inline constexpr const1i operator""_v(unsigned long long d) {
    return const1i(d);
}

static __inline float fast_sqrt(float val)  {
        union {
            int32_t tmp;
            float val;
        } u;
    
        u.val = val;
        // Remove last bit so 1.0 gives 1.0
        u.tmp -= 1<<23;
        // tmp is now an approximation to logbase2(val)
        u.tmp >>= 1; // divide by 2
        u.tmp += 1<<29; // add 64 to exponent: (e+127)/2 =(e/2)+63,
        // that represents (e/2)-64 but we want e/2
        return u.val;
}


static __inline float fast_sqrt_2(const float x)
{
    const float xhalf = 0.5f*x;
    union {
        float x;
        int32_t i;
    } u;
    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);  // gives initial guess y0
    return x*u.x*(1.5f - xhalf*u.x*u.x);// Newton step, repeating increases accuracy
}

static __inline double F_ROUND(double v) { return std::round(v); }
static __inline float F_ROUND(float v) { return std::roundf(v); }
static __inline auto F_ROUND(auto v) { return simd::round(v); }


static __inline int F_ABS(int v) { return std::abs(v); }
static __inline double F_ABS(double v) { return std::fabs(v); }
static __inline float F_ABS(float v) { return std::fabs(v); }
static __inline auto F_ABS(auto v) { return simd::fabs(v); }

static __inline mssFloat8 F_SIGN(float mag, mssFloat8 sign) { return simd::copysign((mssFloat8)const1(mag), sign); }
static __inline mssFloat4 F_SIGN(float mag, mssFloat4 sign) { return simd::copysign((mssFloat4)const1(mag), sign); }
static __inline float2v F_SIGN(float2v mag, float2v sign) { return simd::copysign(mag, sign); }
static __inline float4v F_SIGN(float4v mag, float4v sign) { return simd::copysign(mag, sign); }
static __inline float8v F_SIGN(float8v mag, float8v sign) { return simd::copysign(mag, sign); }
static __inline double2v F_SIGN(double2v mag, double2v sign) { return simd::copysign(mag, sign); }
static __inline double4v F_SIGN(double4v mag, double4v sign) { return simd::copysign(mag, sign); }
static __inline double8v F_SIGN(double8v mag, double8v sign) { return simd::copysign(mag, sign); }
static __inline float F_SIGN(float mag, float sign) { return std::copysign(mag, sign); }

static __inline int8v F_MIN(int8v a, int8v b) {return simd::min(a,b); }
static __inline int4v F_MIN(int4v a, int4v b) { return simd::min(a,b); }
static __inline int2v F_MIN(int2v a, int2v b) { return simd::min(a,b); }
static __inline double8v F_MIN(double8v a, double8v b) {return simd::fmin(a,b); }
static __inline double4v F_MIN(double4v a, double4v b) { return simd::fmin(a,b); }
static __inline double2v F_MIN(double2v a, double2v b) { return simd::fmin(a,b); }
static __inline float8v F_MIN(float8v a, float8v b) { return simd::fmin(a,b); }
static __inline float4v F_MIN(float4v a, float4v b) { return simd::fmin(a,b); }
static __inline float2v F_MIN(float2v a, float2v b) { return simd::fmin(a,b); }

static __inline double F_MIN(float a, double b) { return std::fmin((double)a, b); }
static __inline float F_MIN(float a, float b) { return std::min(a, b); }
static __inline int F_MIN(int a, int b) { return std::min(a, b); }
static __inline size_t F_MIN(size_t a, size_t b) { return std::min(a, b); }

static __inline const1i F_MAX(const1i a, const1i b) { return std::max(a, b); }
static __inline size_t F_MAX(size_t a, size_t b) { return std::max(a, b); }
static __inline int F_MAX(int a, int b) {return std::max(a,b); }
static __inline double8v F_MAX(double8v a, double8v b) {return simd::fmax(a,b); }
static __inline double8v F_MAX(double8v a, double b) {return simd::fmax(a,(double8v)b); }
static __inline double4v F_MAX(double4v a, double4v b) { return simd::fmax(a,b); }
static __inline double2v F_MAX(double2v a, double2v b) { return simd::fmax(a,b); }
static __inline double2v F_MAX(double2v a, double b) { return simd::fmax(a,(double2v)b); }
static __inline float8v F_MAX(float8v a, float8v b) {return simd::fmax(a,b); }
static __inline float8v F_MAX(float8v a, float b) {return simd::fmax(a,(float8v)b); }
static __inline float4v F_MAX(float4v a, float4v b) { return simd::fmax(a,b); }
static __inline float2v F_MAX(float2v a, float2v b) { return simd::fmax(a,b); }
static __inline float2v F_MAX(float2v a, float b) { return simd::fmax(a,(float2v)b); }
static __inline int8v F_MAX(int8v a, int8v b) { return simd::max(a,b); }
static __inline int8v F_MAX(int8v a, int b) { return simd::max(a,(int8v)b); }
static __inline int4v F_MAX(int4v a, int4v b) { return simd::max(a,b); }
static __inline int4v F_MAX(int4v a, int b) { return simd::max(a,(int4v)b); }
static __inline int2v F_MAX(int2v a, int2v b) { return simd::max(a,b); }
static __inline int2v F_MAX(int2v a, int b) { return simd::max(a,(int2v)b); }
static __inline double F_MAX(double a, double b) { return std::fmax(a, b); }
static __inline double F_MAX(double a, float b) { return std::fmax(a, (double)b); }
static __inline double F_MAX(float a, double b) { return std::fmax((double)a, b); }
static __inline float F_MAX(float a, float b) { return std::fmax(a, b); }
static __inline int1v F_MAX(int1v a, int1v b) { return (int1v){std::max(a[0], b[0])}; }
static __inline int1v F_MAX(int1v a, int b) { return (int1v){std::max(a[0], b)}; }
static __inline float1v F_MAX(float1v a, float1v b) { return (float1v){std::fmax(a[0], b[0])}; }
static __inline float1v F_MAX(float1v a, float b) { return (float1v){std::fmax(a[0], b)}; }
static __inline double1v F_MAX(double1v a, double1v b) { return (double1v){std::fmax(a[0], b[0])}; }
static __inline double1v F_MAX(double1v a, double b) { return (double1v){std::fmax(a[0], b)}; }

template<typename T>
inline T F_SELECT(T mask, T a, T b) {
    // mask should be all-1s or all-0s per lane
    return (mask & a) | (~mask & b);  // requires bitwise ops on float
}
static __inline float8v F_CLAMP(float8v x, float8v a, float8v b) { return simd::clamp(x, a, b); }
static __inline float4v F_CLAMP(float4v x, float4v a, float4v b) { return simd::clamp(x, a, b); }
static __inline float2v F_CLAMP(float2v x, float2v a, float2v b) { return simd::clamp(x, a, b); }
static __inline double8v F_CLAMP(double8v x, double8v a, double8v b) { return simd::clamp(x, a, b); }
static __inline double4v F_CLAMP(double4v x, double4v a, double4v b) { return simd::clamp(x, a, b); }
static __inline double2v F_CLAMP(double2v x, double2v a, double2v b) { return simd::clamp(x, a, b); }
static __inline double F_CLAMP(double x, double a, double b) { return std::clamp(x, a, b); }
static __inline double F_CLAMP(double x, float a, float b) { return std::clamp(x, (double)a, (double)b); }
static __inline double F_CLAMP(double x, float a, double b) { return std::clamp(x, (double)a, b); }
static __inline double F_CLAMP(double x, double a, float b) { return std::clamp(x, a, (double)b); }
static __inline float F_CLAMP(float x, float a, float b) { return std::clamp(x, a, b); }
static __inline float F_CLAMP(float x, double a, float b) { return std::clamp(x, (float)a, b); }
static __inline float F_CLAMP(float x, float a, double b) { return std::clamp(x, a, (float)b); }

static __inline auto F_SIN(auto s) { return simd::sin(s); }
static __inline double F_SIN(double s) { return std::sin(s); }
static __inline float F_SIN(float s) { return std::sin(s); }

static __inline auto F_SINH(auto s) { return simd::sinh(s); }
static __inline double F_SINH(double s) { return std::sinh(s); }
static __inline float F_SINH(float s) { return std::sinh(s); }

static __inline auto F_ASIN(auto s) { return simd::asin(s); }
static __inline double F_ASIN(double s) { return std::asin(s); }
static __inline float F_ASIN(float s) { return std::asin(s); }

static __inline int __float_as_int(float in) {
     union fi { int i; float f; } conv;
     conv.f = in;
     return conv.i;
}

static __inline float __int_as_float(int a)
{
    union {int a; float b;} u;
    u.a = a;
    return u.b;
}

static inline constexpr double fast_logf(double a)
{
    double m, r, s, t, i, f;
    int32_t e;

 //   return __builtin_log(a);
    
    if ((a > 0.0) && (a <= 3.40e+38)) { // 0x1.fffffep+127
        m = frexpf(a, &e);
        if (m < 0.666666667) {
            m = m + m;
            e = e - 1;
        }
        i = (float)e;
        /* m in [2/3, 4/3] */
        f = m - 1.0f;
        s = f * f;
        /* Compute log1p(f) for f in [-1/3, 1/3] */
        r = fmaf(-0.130187988, f, 0.140889585); // -0x1.0aa000p-3, 0x1.208ab8p-3
        t = fmaf(-0.121489584, f, 0.139809534); // -0x1.f19f10p-4, 0x1.1e5476p-3
        r = fmaf(r, s, t);
        r = fmaf(r, f, -0.166845024); // -0x1.55b2d8p-3
        r = fmaf(r, f, 0.200121149); //  0x1.99d91ep-3
        r = fmaf(r, f, -0.249996364); // -0x1.fffe18p-3
        r = fmaf(r, f, 0.333331943); //  0x1.5554f8p-2
        r = fmaf(r, f, -0.500000000); // -0x1.000000p-1
        r = fmaf(r, s, f);
        r = fmaf(i, 0.693147182, r); //   0x1.62e430p-1 // log(2)
        return r;
    }
    return 0.0;
}

__inline static double fast_expf (double a)
{
  //  return __builtin_exp(a);
    
    double f, r, j, s, t;
    long i, ia;

    // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = fmaf (1.442695, a, 12582912.) - 12582912.; // 0x1.715476p0, 0x1.8p23
    f = fmaf (j, -6.93145752e-1, a); // -0x1.62e400p-1  // log_2_hi
    f = fmaf (j, -1.42860677e-6, f); // -0x1.7f7d1cp-20 // log_2_lo
    i = (int)j;
    // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
    r =             1.37805939e-3;  // 0x1.694000p-10
    r = fmaf (r, f, 8.37312452e-3); // 0x1.125edcp-7
    r = fmaf (r, f, 4.16695364e-2); // 0x1.555b5ap-5
    r = fmaf (r, f, 1.66664720e-1); // 0x1.555450p-3
    r = fmaf (r, f, 4.99999851e-1); // 0x1.fffff6p-2
    r = fmaf (r, f, 1.00000000e+0); // 0x1.000000p+0
    r = fmaf (r, f, 1.00000000e+0); // 0x1.000000p+0
    // exp(a) = 2**i * r
    ia = (i > 0) ?  0 : 0x83000000;
    s = __int_as_float (0x7f000000 + ia);
    t = __int_as_float ((i << 23) - ia);
    r = r * s;
    r = r * t;
    // handle special cases: severe overflow / underflow
    if (F_ABS (a) >= 104.0) r = s * s;
    return r;
}
static __inline double F_LOG(double g) { return std::log(g); }
static __inline float F_LOG(float g) { return std::logf(g); }
static __inline auto F_LOG(auto g) { return simd::log(g); }


static __inline double fast_powf(double a, double b) {
    return std::pow( a, b );
 //  return fast_expf(b * F_LOG(a));
}

static __inline double fast_atan( double x )
{
    double a, z, p, r, q, s, t;
    // argument reduction:
    //   arctan (-x) = -arctan(x);
    //   arctan (1/x) = 1/2 * pi - arctan (x), when x > 0
    
    z = F_ABS (x);
    a = (z > 1.0) ? (1.0 / z) : z;
    s = a * a;
    q = s * s;
    // core approximation: approximate atan(x) on [0,1]
    p =            -2.0258553044340116e-5;  // -0x1.53e1d2a258e3ap-16
    t =             2.2302240345710764e-4;  //  0x1.d3b63dbb6167ap-13
    p = fma (p, q, -1.1640717779912220e-3); // -0x1.312788ddde71dp-10
    t = fma (t, q,  3.8559749383656407e-3); //  0x1.f9690c824aaf1p-9
    p = fma (p, q, -9.1845592187222193e-3); // -0x1.2cf5aabc7dbd2p-7
    t = fma (t, q,  1.6978035834594660e-2); //  0x1.162b0b2a3bcdcp-6
    p = fma (p, q, -2.5826796814492296e-2); // -0x1.a7256feb6f841p-6
    t = fma (t, q,  3.4067811082715810e-2); //  0x1.171560ce4a4ecp-5
    p = fma (p, q, -4.0926382420509999e-2); // -0x1.4f44d841450e8p-5
    t = fma (t, q,  4.6739496199158334e-2); //  0x1.7ee3d3f36bbc6p-5
    p = fma (p, q, -5.2392330054601366e-2); // -0x1.ad32ae04a9fd8p-5
    t = fma (t, q,  5.8773077721790683e-2); //  0x1.e17813d669537p-5
    p = fma (p, q, -6.6658603633512892e-2); // -0x1.11089ca9a5be4p-4
    t = fma (t, q,  7.6922129305867892e-2); //  0x1.3b12b2db5173cp-4
    p = fma (p, s, t);
    p = fma (p, s, -9.0909012354005267e-2); // -0x1.745d022f8dc5fp-4
    p = fma (p, s,  1.1111110678749421e-1); //  0x1.c71c709dfe925p-4
    p = fma (p, s, -1.4285714271334810e-1); // -0x1.2492491fa1742p-3
    p = fma (p, s,  1.9999999999755005e-1); //  0x1.99999999840cdp-3
    p = fma (p, s, -3.3333333333331838e-1); // -0x1.5555555555448p-2
    p = fma (p * s, a, a);
    // back substitution in accordance with argument reduction //
    // double-precision factorization of PI/2 courtesy of Tor Myklebust //
    r = (z > 1.0) ? fma (0.93282184640716537, 1.6839188885261840, -p) : p;
    return copysign (r, x);
}

static __inline double F_SQRT(double s) { return simd::sqrt(s); }
static __inline float F_SQRT(float s) { return simd::sqrt(s); }
static __inline auto F_SQRT(auto s) { return simd::sqrt(s); }


static __inline auto F_SQR(const auto s) { return s * s; }

//static __inline zfloat8 F_ATAN(zfloat8 s) { return zfloat8{fast_atan(s[0]), fast_atan(s[1]), fast_atan(s[2]), fast_atan(s[3]), fast_atan(s[4]), fast_atan(s[5]), fast_atan(s[6]), fast_atan(s[7]) }; }
static __inline double F_ATAN(double s) { return simd::atan(s); }
static __inline float F_ATAN(float s) { return simd::atan(s); }
static __inline auto F_ATAN(auto s) { return simd::atan(s); }

static __inline double F_ACOS(double s) { return std::acos(s); }
static __inline auto F_ACOS(auto s) { return simd::acos(s); }
static __inline int F_CLAMP(int x, int a, int b) { return std::clamp(x, a, b); }

static __inline float __builtin_reduce_avg(const float8v &a) {
    float x = simd_reduce_add(a) * 0.5f;
    return (x != x) ? 0.0f : x;
}

static __inline float __builtin_reduce_avg(const float &a) {
    return a;
}

static __inline double __builtin_reduce_avg(const double &a) {
    return a;
}

static __inline float __builtin_reduce_avg(const float4v &a) {
    float x = simd_reduce_add(a) * 0.25f;
    return (x != x) ? 0.0 : x;
}

static __inline float __builtin_reduce_avg(const float2v &a) {
    float x = simd_reduce_add(a) * 0.125;
    return (x != x) ? 0.0 : x;
}

static __inline double __builtin_reduce_avg(const double8v &a) {
    double x = simd_reduce_add(a) * 0.5;
    return (x != x) ? 0.0 : x;
}
static __inline double __builtin_reduce_avg(const double4v &a) {
    double x = simd_reduce_add(a) * 0.25;
    return (x != x) ? 0.0 : x;
}
static __inline double __builtin_reduce_avg(const double2v &a) {
    double x = simd_reduce_add(a) * 0.125;
    return (x != x) ? 0.0 : x;
}

static __inline double fast_isqrt(double v)
{
    float x=(float)v, xhalf = 0.5f * x;
    int32_t i = *(int32_t*)&x;
    i = 0x5f3759df-(i>>1);
    x = *(float*)&i;
    x = x*(1.5f-(xhalf*x*x));
    return (double)x;
}

#if TARGET_OS_MACCATALYST
class DenormalsOff {
public:
    DenormalsOff() {
        // Save the current floating-point environment
        fegetenv(&oldEnv_);
        
        // Set the environment to default (usually flushes denormals to zero)
        fesetenv(FE_DFL_ENV);

        // Optional: Set rounding mode (for example, round to nearest)
        fesetround(FE_TONEAREST);
    }

    ~DenormalsOff() {
        // Restore the original floating-point environment
        fesetenv(&oldEnv_);
    }

private:
    fenv_t oldEnv_;  // Store the original floating-point environment
};
#else
class DenormalsOff {
   
public:
   DenormalsOff() {
       // Store current FPCR
       asm volatile("mrs %0, fpcr" : "=r" (fpcr_));
       // Set FZ (Flush-to-zero mode) bit to disable denormals
       asm volatile("orr %0, %0, (1 << 24)" : "+r" (fpcr_));
       // Write back the modified FPCR
       asm volatile("msr fpcr, %0" :: "r" (fpcr_));
   }

   ~DenormalsOff() {
       // Restore the original FPCR
       asm volatile("msr fpcr, %0" :: "r" (fpcr_));
   }
private:
   uint64_t fpcr_;
};
#endif

static __inline double fast_log2(double x) { return std::log2(x); }
static __inline double fast_log(double x) { return std::log(x); }

static __inline double F_POW(double a, double b) { return fast_powf(a,b); }
static __inline float F_POW(float a, float b) { return fast_powf(a, b); }
static __inline double8v F_POW(double8v a, double8v b) { return simd::pow(a, b); }
static __inline double4v F_POW(double4v a, double4v b) { return simd::pow(a, b); }
static __inline double2v F_POW(double2v a, double2v b) { return simd::pow(a, b); }
static __inline float8v F_POW(float8v a, float8v b) { return simd::pow(a, b); }
static __inline float4v F_POW(float4v a, float4v b) { return simd::pow(a, b); }
static __inline float2v F_POW(float2v a, float2v b) { return simd::pow(a, b); }

static __inline auto F_EXP(auto g) { return simd::exp(g); }
static __inline double F_EXP(double g) { return std::exp(g); }
static __inline float F_EXP(float g) { return std::expf(g); }

static __inline auto db2lin(auto db) { // dB to linear
   return F_POW(10.0, 0.05 * db);
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, double>
static inline T simd_fast_atan2(const T &y, const T &x) {
    using BaseType = T;
    constexpr auto c1 = 0.78539816339744830962;
    constexpr auto c2 = 2.35619449019234492885;

    if (y == 0.0 && x == 0.0)
        return 0.0;
    
    T abs_y = std::abs(y);
    T angle;
    if (x >= 0.0)
        angle = c1 - c1 * ((x - abs_y) / (x + abs_y));
    else
        angle = c2 - c1 * ((x + abs_y) / (abs_y - x));
    
    if (y < 0.0) return -angle;
    return angle;
}
#define SIMD_ATAN2_FAST(T) \
template<typename U> requires std::same_as<U, T> \
static inline U simd_fast_atan2(const U &y, const U& x) { \
    using IT = SimdSame<U, std::conditional_t<std::is_same_v<SimdBase<U>, double>, long, int>>; \
    using BaseType = SimdBase<U>; \
    constexpr auto c1 = 0.78539816339744830962; \
    constexpr auto c2 = 2.35619449019234492885; \
    U abs_y = F_ABS(y); \
    IT is_zero = convertvector(-((x == 0.0) && (y == 0.0)), IT); \
    U r1 = (x - abs_y) / (x + abs_y); \
    U angle1 = c1 - c1 * r1; \
    U r2 = (x + abs_y) / (abs_y - x); \
    U angle2 = c2 - c1 * r2; \
    IT x_positive = convertvector(-(x >= 0.0), IT); \
    U angle = simd::select(angle2, angle1, x_positive); \
    IT y_negative = convertvector(-(y < 0.0), IT); \
    angle = simd::select(angle, -angle, y_negative); \
    return simd::select(angle, 0.0, is_zero); \
}

SIMD_ATAN2_FAST(simd_float8);
SIMD_ATAN2_FAST(simd_double8);
SIMD_ATAN2_FAST(simd_float4);
SIMD_ATAN2_FAST(simd_double4);
SIMD_ATAN2_FAST(simd_float2);
SIMD_ATAN2_FAST(simd_double2);

static __inline auto F_ATAN2(float a, float b) { return simd::atan2(a, b); }
static __inline auto F_ATAN2(double a, double b) { return simd::atan2(a, b); }
static __inline auto F_ATAN2(auto a, auto b) { return simd::atan2(a, b); } // simd_fast_atan2

static __inline auto F_COS(auto g) { return simd::cos(g); }
static __inline double F_COS(double g) { return std::cos(g); }
static __inline float F_COS(float g) { return std::cosf(g); }

static __inline auto F_TANH(auto g) { return simd::tanh(g); }
static __inline double F_TANH(double g) { return std::tanh(g); }
static __inline float F_TANH(float g) { return std::tanh(g); }

static __inline auto F_FMOD(auto a, auto b) { return simd::fmod(a, b); }
static __inline float F_FMOD(float a, float b) { return std::fmodf(a, b); }
static __inline double F_FMOD(double a, double b) { return std::fmod(a, b); }

static __inline double F_SINF(double g) { return std::sinf(g); }
static __inline float F_SINF(float g) { return std::sinf(g); }

static __inline auto F_TAN(auto g) { return simd::tan(g); }
static __inline double F_TAN(double g) { return std::tan(g); }
static __inline float F_TAN(float g) { return std::tanf(g); }

static __inline double F_LOG10(double g) { return std::log10(g); }
static __inline float F_LOG10(float g) { return std::log10f(g); }
static __inline auto F_LOG10(auto g) { return simd::log10(g); }

static __inline double F_LOG2(double g) { return std::log2(g); }
static __inline float F_LOG2(float g) { return std::log2f(g); }
static __inline auto F_LOG2(auto g) { return simd::log2(g); }

static __inline zfloat2 from_dB(zfloat2 x) { return F_EXP( x * 0.1151292546497023 ); }
static __inline zfloat8 from_dB(zfloat8 x) { return F_EXP( x * 0.1151292546497023 ); }
static __inline mssFloat2 from_dB(mssFloat2 x) { return F_EXP( x * 0.1151292546497023 ); }
static __inline mssFloat8 from_dB(mssFloat8 x) { return F_EXP( x * 0.1151292546497023 ); }
static __inline double from_dB(double x) { return F_EXP( x * 0.1151292546497023 ); }


constexpr auto to_dB(auto g) {return (20.0f * F_LOG10(g)); }
constexpr float to_dB(float g) { return 20.0f * std::log10f(std::abs(g)); }
constexpr double to_dB(double g) { return 20.0 * std::log10(std::abs(g)); }

static inline auto F_RINT(auto x) { return simd::rint(x); }

#define DENORM_VAL  1.0e-15 // 1.0e-20

static inline float WRAP360(float x) {
    x = F_FMOD((float)x, 360.0f);
    if (x < 0.0f) x += 360.0f;
    return x;
}

inline float deg2rad(float degrees) {
    return degrees * (M_PI / 180.0f);
}

inline float rad2deg(float radians) {
    return radians * (180.0f / M_PI);
}

extern "C" int isnanf(float v);
extern "C" int isnand(double v);
extern "C" simd_int2 isnanf2(simd_float2 v);
extern "C" simd_int2 isnand2(simd_double2 v);

extern "C" int isinff(float v);
extern "C" int isinfd(double v);
extern "C" simd_int2 isinff2(simd_float2 v);
extern "C" simd_int2 isinfd2(simd_double2 v);

static inline bool ISNAN(float x) { return isnanf(x); }
static inline bool ISNAN(double x) { return isnand(x); }
static inline simd_int2 ISNAN(const simd_float2& v) {
    return isnanf2(v);
}
static inline simd_int4 ISNAN(const simd_float4& v) {
    return simd_make_int4(ISNAN(v.xy),ISNAN(v.zw));
}
static inline simd_int8 ISNAN(const simd_float8& v) {
    return simd_make_int8(ISNAN(v.lo),ISNAN(v.hi));
}

static inline bool ISINF(float x) { return isinff(x); }
static inline bool ISINF(double x) { return isinfd(x); }
static inline simd_int2 ISINF(const simd_float2& v) {
    return isinff2(v);
}
static inline simd_int4 ISINF(const simd_float4& v) {
    return simd_make_int4(ISINF(v.xy),ISINF(v.zw));
}
static inline simd_int8 ISINF(const simd_float8& v) {
    return simd_make_int8(ISINF(v.lo),ISINF(v.hi));
}
static inline simd_int2 ISNAN(const simd_double2& v) {
    return isnand2(v.xy);
}
static inline simd_int4 ISNAN(const simd_double4& v) {
    return simd_make_int4(ISNAN(v.xy),ISNAN(v.zw));
}
static inline simd_int8 ISNAN(const simd_double8& v) {
    return simd_make_int8(ISNAN(v.lo),ISNAN(v.hi));
}
static inline simd_int2 ISINF(const simd_double2& v) {
    return isinfd2(v.xy);
}
static inline simd_int4 ISINF(const simd_double4& v) {
    return simd_make_int4(ISINF(v.xy),ISINF(v.zw));
}
static inline simd_int8 ISINF(const simd_double8& v) {
    return simd_make_int8(ISINF(v.lo),ISINF(v.hi));
}

static inline const float _inf() {
    int inf = 0x7F800000;
    return *(float*)&inf;
}
static inline const float _nan() {
    int nan = 0x7F800001;
    return *(float*)&nan;
}

static inline bool ISNORM(float v) { return !ISNAN(v) && !ISINF(v); }
static inline bool ISNORM(double v) { return !ISNAN(v) && !ISINF(v); }
static inline bool ISNORM(auto v) { return simd_reduce_add(ISNAN(v) + ISINF(v)) == 0; }

static inline float FIXNORM(float v) { return ISNORM( v ) ? v : 0.0f; }
static inline double FIXNORM(double v) { return ISNORM( v ) ? v : 0.0; }
//static inline auto FIXNORM(auto v) { return ISNORM( v ) ? v : 0.0; }

static inline simd_float2 FIXNORM(const simd_float2& v) {
    simd_float2 result;
    result.x = ISNORM(v.x) ? v.x : 0.0f;
    result.y = ISNORM(v.y) ? v.y : 0.0f;
    return result;
}
static inline simd_float4 FIXNORM(const simd_float4& v) {
    return simd_make_float4(
        FIXNORM(v.xy),
        FIXNORM(v.zw)
    );
}
static inline simd_float8 FIXNORM(const simd_float8& v) {
    return simd_make_float8(
        FIXNORM(v.lo),
        FIXNORM(v.hi)
    );
}
static inline simd_double2 FIXNORM(const simd_double2& v) {
    simd_double2 result;
    result.x = ISNORM(v.x) ? v.x : 0.0f;
    result.y = ISNORM(v.y) ? v.y : 0.0f;
    return result;
}
static inline simd_double4 FIXNORM(const simd_double4& v) {
    return simd_make_double4(
        FIXNORM(v.xy),
        FIXNORM(v.zw)
    );
}
static inline simd_double8 FIXNORM(const simd_double8& v) {
    return simd_make_double8(
        FIXNORM(v.lo),
        FIXNORM(v.hi)
    );
}


#ifndef D_CMPLX_T
#define D_CMPLX_T

template <typename T>
struct cmplxT {
    using T1 = SimdBase<T>;
    // Data members
    T re, im;
    
    // Constructors
    cmplxT() : re(0.0), im(0.0) {}
    cmplxT(T r, T i) : re(r), im(i) {}
    cmplxT(const cmplxT& v) : re(v.re), im(v.im) {}
    cmplxT(long double v) : re(v), im(v) {}
    cmplxT(const T1& v) : re(v), im(v) {}
    
    static inline cmplxT<T> polar(T mag, T phase) {
        return cmplxT<T>(mag * F_COS(phase), mag * F_SIN(phase));
    }
    
    // Assignment operators - don't use const return type
    inline cmplxT<T>& operator = (const cmplxT<T>& v) {
        re = v.re;
        im = v.im;
        return *this;
    }
    
    inline cmplxT<T>& operator = (const T1& v) {
        re = v;
        im = v;
        return *this;
    }
    
    // Binary arithmetic operators - should not modify *this
    inline cmplxT<T> operator * (const T& x) const {
        return cmplxT<T>(re * x, im * x);
    }
    
    inline cmplxT<T> operator * (const long double x) const {
        return cmplxT<T>(re * x, im * x);
    }
    
    inline cmplxT<T> operator * (const cmplxT<T>& x) const {
        return cmplxT<T>(re * x.re - im * x.im, re * x.im + im * x.re);
    }
    
    inline cmplxT<T> operator + (const cmplxT<T>& x) const {
        return cmplxT<T>(re + x.re, im + x.im);
    }
    
    inline cmplxT<T> operator - () const {
        return cmplxT<T>(-re, -im);
    }
    
    inline cmplxT<T> operator - (const cmplxT<T>& d) const {
        return cmplxT<T>(re - d.re, im - d.im);
    }
    
    inline cmplxT<T> operator / (const cmplxT<T>& x) const {
        T denominator = (x.re * x.re + x.im * x.im);
        T safe_denom = F_MAX(denominator, T(1e-30)); // Prevent division by zero
        return cmplxT<T>(
            (re * x.re + im * x.im) / safe_denom,
            (im * x.re - re * x.im) / safe_denom
        );
    }
    
    inline cmplxT<T> operator & (const int8v& mask) const {
        return cmplxT<T>(shufflevector(re, mask), shufflevector(im, mask));
    }
    
    // Compound assignment operators - should not be const
    inline const cmplxT<T>& operator *= (const long double d) {
        return *this = *this * d;
    }
    
    inline const cmplxT<T>& operator *= (const T& d) {
        return *this = *this * d;
    }
    
    inline const cmplxT<T>& operator *= (const cmplxT<T>& x) {
        return *this = *this * x;
    }
    
    inline const cmplxT<T>& operator += (const cmplxT<T>& x) {
        return *this = *this + x;
    }
    
    inline const cmplxT<T>& operator -= (const cmplxT<T>& x) {
        return *this = *this - x;
    }
    
    inline cmplxT<T> conj() const { return cmplxT<T>(re, -im); }
    
    // Index operator - must return by value, not reference
    inline cmplxT<T1> operator [](const int i) const {
        return cmplxT<T1>(re[i], im[i]);
    }
    
    // Utility methods
    inline T length() const { return F_SQRT(re * re + im * im); }
    inline T mag() const { return F_SQRT(re * re + im * im); }
    inline T mag_squared() const { return re * re + im * im; }
    inline T phase() const { return F_ATAN2(im, re); }
    inline T real() const { return re; }
    inline T imag() const { return im; }
};

template <typename T>
inline cmplxT<T> operator * (const T& scalar, const cmplxT<T>& c) {
    return cmplxT<T>(scalar * c.re, scalar * c.im);
}

// Scalar complex - vector complex (broadcasts scalar)
template <typename S, typename V>
inline cmplxT<V> operator - (const cmplxT<S>& scalar, const cmplxT<V>& vec) {
    return cmplxT<V>(scalar.re - vec.re, scalar.im - vec.im);
}

// Scalar complex + vector complex (broadcasts scalar)
template <typename S, typename V>
inline cmplxT<V> operator + (const cmplxT<S>& scalar, const cmplxT<V>& vec) {
    return cmplxT<V>(scalar.re + vec.re, scalar.im + vec.im);
}

template <typename T, typename U>
inline cmplxT<T> operator * (const cmplxT<T>& a, const cmplxT<U>& b) {
    return cmplxT<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

template <typename T, typename U>
inline cmplxT<T>& operator *= (cmplxT<T>& a, const cmplxT<U>& b) {
    T new_re = a.re * b.re - a.im * b.im;
    T new_im = a.re * b.im + a.im * b.re;
    a.re = new_re;
    a.im = new_im;
    return a;
}

#endif //D_CMPLX_T



static inline float simd_reduce_max(float x) {
    return x;
}
static inline double simd_reduce_max(double x) {
    return x;
}

static inline double fix_normal(const double &val)
{
    return ISNORM(val) ? 0.0 : val;
}

static inline bool is_normal(const double &val)
{
    return ISNORM(val);
}

static inline bool is_normal(const auto &val)
{
    return ISNORM(val);
}

static inline bool is_denormal(const float& value) {
    return F_ABS(value) < DENORM_VAL;
}

static inline bool is_denormal(const double& value) {
    return F_ABS(value) < DENORM_VAL;
}

template <typename T>
static inline auto is_denormal(const auto& value) {
    return convertvector(-(F_ABS(value) < DENORM_VAL), T);
}

static inline auto fix_denormal(const auto& value) {
    return (1 - is_denormal(value)) * value;
}

static inline simd_int2 ISNORM_VEC(const simd_float2& v) {
    return !(ISNAN(v) | ISINF(v));
}

static inline simd_int4 ISNORM_VEC(const simd_float4& v) {
    return !(ISNAN(v) | ISINF(v));
}

// has bad values = true, no bad values = false
template <typename T>
static inline bool overflow_check(T& input) {
    bool hadBadValues = !ISNORM(input);
    input = FIXNORM(input);
    return hadBadValues;
}
// ------------------------------------------------------------------
// Generic overflow_fix for scalar or SIMD types
// ------------------------------------------------------------------
template<typename T>
inline T overflow_fix(const T& x) {
    return FIXNORM(x);
}

// ------------------------------------------------------------------
// Complex overload: works for scalar or SIMD T
// ------------------------------------------------------------------
template<typename T>
inline cmplxT<T> overflow_fix(const cmplxT<T>& x) {
    return cmplxT<T>(
        overflow_fix(x.re),
        overflow_fix(x.im)
    );
}

template<typename T>
static inline T OVERFLOW_FIX_CLIP(const T& x, SimdBase<T> CLIP_THRESH = 4.95, SimdBase<T> MAX_CAP = 5.0) {
    const auto clip_range = MAX_CAP - CLIP_THRESH;
    
    const T abs_x = F_ABS(x);
    const T cc = convertvector(-(abs_x > T(CLIP_THRESH)), T);
    const T excess = abs_x - CLIP_THRESH;
    
    const T clipped_magnitude = T(CLIP_THRESH) + clip_range * excess / (1 + excess);
    return cc * F_SIGN(clipped_magnitude, x) + (1-cc) * x;
}


static inline auto smooth_clamp(const auto &input, const auto &min_val, const auto &max_val) {
    auto above_max = F_MAX(input - max_val, 0.0);
    auto below_min = F_MAX(min_val - input, 0.0);
    
    auto above_factor = 1.0 / (1.0 + F_ABS(above_max));
    auto below_factor = 1.0 / (1.0 + F_ABS(below_min));
    
    auto result_above = max_val + above_max * above_factor;
    auto result_below = min_val - below_min * below_factor;
    
    auto result = F_MIN(input, result_above);
    return fix_denormal(F_MAX(result, result_below));
}

template <typename T>
__inline T sanitize_denormal(const T& x, bool is_bass = false) {
    // Use lower threshold for bass to preserve subtle details
    const SimdBase<T> BASS_DENORM_THRESHOLD = 1.0e-20;
    const SimdBase<T> NORMAL_DENORM_THRESHOLD = 1.0e-15;
    
    // Choose appropriate threshold
    SimdBase<T> threshold = is_bass ? BASS_DENORM_THRESHOLD : NORMAL_DENORM_THRESHOLD;
    
    // Eliminate denormal values to prevent performance issues
    T mask = convertvector(-(F_ABS(x) >= threshold), T);
    return x * mask;
}

template <typename T>
static inline int sanitize_normals( T *out, int count, const char *str = "" )
{
    int present = 0;
    
    while (count--) {
        T s = *out;
        if (is_normal(s))
            *out++ = s;
        else {
            *out++ = 0.0;
            present++;
        }
    }
 //   if (present > 0)
   //     LOGGER("%s: baddies found: %d", str, present);
    return present;
}

class PeakDistAnalyzer {
    static const int MAX_DIST = 30;
public:
    PeakDistAnalyzer() {
        peak_idx = idx = 0;
        memset(peaks, 0, sizeof(peaks));
    }
    
    void reset() {
        for (int i=0; i < peak_idx; i++)
            peaks[i].peak = 0.0;
    }
    
    void resetAll() {
        for (int i=0; i < peak_idx; i++) {
            peaks[i].peak = 0.0;
            peaks[i].sum_sq = 0.0;
            peaks[i].count = 0;
        }
    }
    
    void print(bool quiet = false) {
        if (quiet) return;
        for (int i=0; i < peak_idx; i++) {
            double rms = peaks[i].count > 0 ? F_SQRT(peaks[i].sum_sq / peaks[i].count) : 0.0;
            fprintf(stderr, "[%2d] %5s : peak=%f  rms=%f\n", i+1, peaks[i].name, (float)peaks[i].peak, (float)rms);
        }
    }
    
    template <typename T>
    void calcGain(T *out, int count, const char *str)
    {
        if (!str) return;
        
        if (!peaks[0].name[0] || !(strncasecmp(str, peaks[0].name, 255)))
            idx = 0;
        
        double peak = peaks[idx].peak;
        double sum_sq = peaks[idx].sum_sq;
        uint64_t n = count;
        
        while (count--) {
            T s = *out++;
            if constexpr (std::is_arithmetic_v<T>) {
                double abs_s = F_ABS(s);
                peak = F_MAX(abs_s, peak);
                sum_sq += s * s;
            } else {
                peak = F_MAX(simd_reduce_max(F_ABS(s)), peak);
                sum_sq += simd_reduce_add(s * s);
            }
        }
       
        peak_idx = F_MAX(peak_idx, idx + 1);
        peaks[idx].peak = peak;
        peaks[idx].sum_sq = sum_sq;
        peaks[idx].count += n;
        strncpy(peaks[idx].name, str, 255);
        
        idx++;
    }
    
protected:
    int idx, peak_idx;
    struct {
        double peak;
        double sum_sq;
        uint64_t count;
        char name[256];
    } peaks[MAX_DIST];
};

#define P_STR(x) #x
#define P_STRINGIFY(x) P_STR(x)

#define PRINT_D8(v) fprintf(stderr, "%s(D8): [%e %e %e %e %e %e %e %e]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
#define PRINT_D4(v) fprintf(stderr, "%s(D4): [%e %e %e %e]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3]);
#define PRINT_D2(v) fprintf(stderr, "%s(D2): [%e %e]\n", P_STRINGIFY(v), v[0], v[1];
#define PRINT_D1(v) fprintf(stderr, "%s(D1): [%e]\n", P_STRINGIFY(v), v);
#define PRINT_D1_CMPLX(v) fprintf(stderr, "%s(D1): [%e:%e]\n", P_STRINGIFY(v), v.re, v.im);

#define PRINT_F8(v) fprintf(stderr, "%s(F8): [%e %e %e %e %e %e %e %e]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
#define PRINT_F4(v) fprintf(stderr, "%s(F4): [%e %e %e %e]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3]);
#define PRINT_F2(v) fprintf(stderr, "%s(F2): [%e %e]\n", P_STRINGIFY(v), v[0], v[1]);
#define PRINT_D2(v) fprintf(stderr, "%s(D2): [%e %e]\n", P_STRINGIFY(v), (float)v[0], (float)v[1]);
#define PRINT_FC8(v) fprintf(stderr, "%s(D8): [%e %e %e %e %e %e %e %e]\n", P_STRINGIFY(v), v.re[0], v.re[1], v.re[2], v.re[3], v.re[4], v.re[5], v.re[6], v.re[7]);

#define PRINT_I8(v) fprintf(stderr, "%s(I8): [%d %d %d %d %d %d %d %d]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
#define PRINT_I4(v) fprintf(stderr, "%s(I4): [%d %d %d %d]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3]);
#define PRINT_I2(v) fprintf(stderr, "%s(I2): [%d %d]\n", P_STRINGIFY(v), v[0], v[1]);

#define getName(var)            #var
#define CHECK_NULL(x)           if (!(x)) { fprintf(stderr, "CHECK_NULL: %s\n",getName(x)); raise(SIGINT); }

static constexpr float SQ2_2 = 0.7071067811865475244;


#define CMPLX_MAG(X,Y)          F_SQRT( (X) * (X) + (Y) * (Y) )


#define THROW_MEM(cond)  if (cond) throw (Converror (Converror::MEM_ALLOC));

static inline double square(double x) {
    return x * x;
}

static inline float square(float x) {
    return x * x;
}

// Overloaded safe_exp function for auto type
__inline auto safe_exp(const auto& x) {
    // Limit the range to prevent overflow
    return F_EXP(F_CLAMP(x, -88.0, 88.0));
}

template <typename T>
class Queue {
public:
    Queue() : myFront(0), myBack(0) {}
    Queue(const T & q) {
        myFront = myBack = 0;
        if(!q.empty()) {
            myFront = myBack = new Node(q.front());
            NodePointer qPtr = q.myFront->next;
            while(qPtr != NULL) {
                myBack->next = new Node(qPtr->data);
                myBack = myBack->next;
                qPtr = qPtr->next;
            }
        }
    }
   
    ~Queue() {
        NodePointer prev = myFront, ptr;
        while(prev != NULL) {
            ptr = prev->next;
            delete prev;
            prev = ptr;
        }
    }
  
    bool empty() const {
        return (myFront == NULL);
    }
   
    void push(const T & value) {
        NodePointer newNodePtr = new Node(value);
        if(empty()) {
            myFront = myBack = newNodePtr;
            newNodePtr->next = NULL;
        } else {
            myBack->next = newNodePtr;
            myBack = newNodePtr;
            newNodePtr->next = NULL;
        }
    }
    
    T pop() {
        T data;
        if ( !empty() ) {
            data = myFront->data;
            NodePointer ptr = myFront;
            myFront = myFront->next;
            delete ptr;
            if (!myFront) myBack = NULL;
        }
        return data;
    }
    
    T *peek() const {
        return empty() ? nullptr : &myFront->data;
    }
   
    Queue<T>& operator=(const T &q) {
        if(this != &q) {
            this->~Queue();
            if(q.empty()) {
                myFront = myBack = NULL;
            } else {
                myFront = myBack = new Node(q.front());
                NodePointer qPtr = q.myFront->next;
                while(qPtr != NULL) {
                    myBack->next = new Node(qPtr->data);
                    myBack = myBack->next;
                    qPtr = qPtr->next;
                }
            }
        }
        return *this;
    }

private:
    class Node {
    public:
        T data;
        Node * next;
        Node(T value, Node * first = 0) : data(value),
                                          next(first) {}

    };
    typedef Node * NodePointer;
    NodePointer myFront, myBack, queueSize;
};

template <typename XT>
static void ConvolveDirect_FFT(cmplxT<XT> *buf, cmplxT<SimdBase<XT>> *res, int count) {
    for (int i = 0; i < count; i++) {
        buf[i] *= res[i];
    }
}

template <typename XT>
static void ConvolveDirect_FFT(cmplxT<XT> *buf, cmplxT<SimdBase<XT>> *res, int count, SimdBase<XT> koef) {
    for (int i = 0; i < count; i++) {
        buf[i] *= res[i] * koef;
    }
}

template <typename XT>
static void ConvolveDirect_MCLT(cmplxT<XT> *buf, cmplxT<SimdBase<XT>> *res, int numBins, SimdBase<XT> koef = 1.0) {
    // numBins is now the actual bin count, not time-domain size
    for (int i = 0; i < numBins; i++) {
        cmplxT<XT> temp = buf[i];
        buf[i].re = (temp.re * res[i].re - temp.im * res[i].im) * koef;
        buf[i].im = (temp.re * res[i].im + temp.im * res[i].re) * koef;
    }
}

// Aligned memory allocation
template<typename T>
inline T* callocT(size_t count) {
    void* ptr = nullptr;
    size_t size = count * sizeof(T);
    if (posix_memalign(&ptr, 64, size) == 0 && ptr) {
        std::memset(ptr, 0, size);
    }
    return static_cast<T*>(ptr);
}
inline void callocT_free(void* ptr) { if (ptr) free(ptr); }


#endif // __cplusplus

