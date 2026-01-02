/*-----------------------------------------------------------------------------*\
**   MCLT Convolver v0.9 -- MCLT-Domain Convolution using MCLTReal             **
**   (c) 2021-2025 Dmitry Boldyrev. All rights reserved.                       **
**                                                                             **
**   mem.h - callocT<T>, callocT_free, pthread_yield, misc                     **
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

#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread/pthread.h>

#define DISABLE_MALLOC_FREE_OVERRIDE
#define OVERRIDE_MALLOCS

// 128-bit alignment for SIMD vector processor cache optimal performance (can try 64)
#define MALLOC_ALIGN                128
#define MEM_ALIGN                   alignas(MALLOC_ALIGN)
#define MEM_ALIGNED                 __attribute__((aligned(MALLOC_ALIGN)))

// do align malloc/calloc/new/new[] to MALLOC_ALIGN settings, if not comment out below
//#define OVERRIDE_MALLOCS

#ifndef likely
#define likely(x)                   __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x)                 __builtin_expect(!!(x), 0)
#endif

// -------------------------------------------------------------------------
//  WARNING: Do not change anything here unless you know what you are doing
// -------------------------------------------------------------------------

// Optimized timing constants for 2048 frames at 48kHz

// Constants for timing control
#define DEFAULT_YIELD_TIMEOUT  0.05      // Default timeout 50.0 ms
#define DELTA_MIN              0.0001    // Threshold for shortest range (100μs)
#define DELTA_BREAK            0.4       // Fraction to sleep in shortest range (40%)
#define DELTA_CHUNK            0.00005   // Max wait in shortest range (50μs)

#ifndef DISABLE_MALLOC_FREE_OVERRIDE
#undef OVERRIDE_MALLOCS
#endif

static const uint64_t NANOS_PER_MSEC = 1000000ULL;

extern mach_timebase_info_data_t timebaseInfo;

void mss_init_mach_time_constants(void);

static inline double abs_to_seconds(uint64_t abs_time) {
    // Convert Mach absolute time units back to nanoseconds
    uint64_t nanos = abs_time * timebaseInfo.numer / timebaseInfo.denom;
    
    // Convert nanoseconds to seconds
    return (double)nanos * 1e-9;
}

static inline double to_sec(double nanos) {
    return nanos * 1e-6;
}

static inline uint64_t seconds_to_abs(double seconds) {
    // 1 second = 1e9 nanoseconds
    uint64_t nanos = (uint64_t)(seconds * 1e9);
    // Convert nanoseconds to mach absolute time units
    return nanos * timebaseInfo.denom / timebaseInfo.numer;
}

static inline uint64_t msec_to_abs(double msec) {
    return (uint64_t)(msec * NANOS_PER_MSEC * (double)timebaseInfo.denom / timebaseInfo.numer);
}

// must initialize timer with mss_init_mach_time_constants
static inline double current_timestamp() {
    return abs_to_seconds(mach_absolute_time());
}

static inline void USLEEP(useconds_t usec) {
    // Check if timebaseInfo has been initialized
    if (timebaseInfo.denom == 0) {
        // Fall back to standard usleep if not initialized
        usleep(usec);
        return;
    }
    
    // For very short waits, just yield
    if (usec < 50) {
        #if defined(__arm__) || defined(__aarch64__)
            asm volatile("yield");
        #else
            sched_yield();  // Standard POSIX function
        #endif
        return;
    }
    
    // For regular waits, use mach_wait_until
    uint64_t wait_time = usec * 1000ULL * timebaseInfo.denom / timebaseInfo.numer;
    mach_wait_until(mach_absolute_time() + wait_time);
}
#ifdef __cplusplus

#include <cstdlib>

static inline void pthread_yield(double timeout = DEFAULT_YIELD_TIMEOUT, uint64_t *_Nullable inTimer = NULL)
{
    static uint64_t globalTimer = mach_absolute_time();
    
    // Get current time in absolute units
    uint64_t current_time = mach_absolute_time();
    uint64_t timeout_abs = seconds_to_abs(timeout);
    uint64_t time_diff, target_time;

    // Calculate target time in absolute units (with safety check)
    
    if (inTimer) {
        // Use provided timer value as base
        target_time = *inTimer + timeout_abs;
        
        // Protection against overflow
        if (target_time < *inTimer)
            target_time = UINT64_MAX;
    } else {
        // Use global timer as base
        target_time = globalTimer + timeout_abs;
        
        // Protection against overflow
        if (target_time < globalTimer)
            target_time = UINT64_MAX;
    }
    
    double delta_seconds;
    
    // Early exit if already past target time
    if (current_time >= target_time)
        goto exit;
    
    // Calculate time difference (safely)
    time_diff = target_time - current_time;
    delta_seconds = abs_to_seconds(time_diff);
    
    // Initial bulk wait for longer timeouts
    if (delta_seconds > 0.001) { // > 1ms
        // Use mach_wait for ~70% of the longer wait time (using floating point)
        uint64_t bulk_wait = (uint64_t)(time_diff * 0.7);
        
        // Safety check - make sure bulk_wait is reasonable
        const uint64_t MAX_WAIT = seconds_to_abs(0.5); // Never wait more than 0.5 second at once
        if (bulk_wait > MAX_WAIT) {
            bulk_wait = MAX_WAIT;
        }
        
        // Only wait if bulk_wait is positive
        if (bulk_wait > 0) {
            mach_wait_until(current_time + bulk_wait);
            current_time = mach_absolute_time();
        }
    }
    // Main adaptive wait loop
    uint64_t remaining;
    while (current_time < target_time && (remaining = target_time - current_time) > 0) {
        // Convert remaining to seconds for comparison
        delta_seconds = abs_to_seconds(remaining);
        
        if (delta_seconds > 0.0002) { // > 200μs
            // Medium range - use OS scheduler yield
            sched_yield();
        }
        else if (delta_seconds > DELTA_MIN) { // Between DELTA_MIN and 200μs
            // Short range - architecture-specific yield
#if defined(__arm__) || defined(__aarch64__)
            asm volatile("yield"); // ARM - efficient CPU hint
#else
            asm volatile("pause"); // x86 - pause instruction
#endif
        }
        else { // Very short range (< DELTA_MIN)
            // Controlled short sleep for final precision
            double chunk_seconds = std::min(delta_seconds * DELTA_BREAK, DELTA_CHUNK);
            uint64_t chunk_abs = seconds_to_abs(chunk_seconds);
            
            // For extremely short waits, just CPU hint instead of mach_wait
            if (chunk_seconds < 0.00001) { // < 10μs
#if defined(__arm__) || defined(__aarch64__)
                for (int i = 0; i < 3; i++) { asm volatile("yield"); }
#else
                for (int i = 0; i < 8; i++) { asm volatile("pause"); }
#endif
            } else if (chunk_abs > 0) { // Only wait if positive
                mach_wait_until(current_time + chunk_abs);
            }
        }
        
        current_time = mach_absolute_time();
    }
    
exit:
    // Update timer reference with current absolute time
    if (inTimer) {
        *inTimer = mach_absolute_time();
    } else {
        globalTimer = mach_absolute_time();
    }
}

#endif

#ifdef __cplusplus

#ifndef DISABLE_MALLOC_FREE_OVERRIDE

#define mss_aligned_malloc(a,b) bigpool.malloc(a)
#define mss_aligned_free bigpool.free
#define mss_aligned_realloc bigpool.realloc
#define mss_alaigned_malloc_size bigpool.malloc_size

#else // defined(DISABLE_MALLOC_FREE_OVERRIDE)

#ifdef OVERRIDE_MALLOCS

#include <c++/v1/__memory/aligned_alloc.h>

static inline void*_Nullable aligned_realloc(size_t alignment, void*_Nullable ptr, size_t new_size) {
    if (ptr == nullptr) {
        return std::__libcpp_aligned_alloc(alignment, new_size);
    }
    
    if (new_size == 0) {
        // If size is 0, free the memory and return NULL
        std::__libcpp_aligned_free(ptr);
        return nullptr;
    }
    
    // Get the current size using malloc_size
    size_t current_size = malloc_size(ptr);
    
    // If the new size fits in the current allocation and we don't need to shrink
    // significantly, just return the same pointer
    if (new_size <= current_size && new_size > (current_size / 2)) {
        return ptr;
    }
    
    // Allocate new block with proper alignment
    void* new_ptr = std::__libcpp_aligned_alloc(alignment, new_size);
    if (!new_ptr) {
        return nullptr; // Allocation failed
    }
    
    // Copy the data from the old block to the new block
    size_t copy_size = (current_size < new_size) ? current_size : new_size;
    memcpy(new_ptr, ptr, copy_size);
    
    // Free the old block
    std::__libcpp_aligned_free(ptr);
    
    return new_ptr;
}

#define mss_aligned_malloc(a,b) std::__libcpp_aligned_alloc(MALLOC_ALIGN, a)
#define mss_aligned_calloc(a,b) std::__libcpp_aligned_alloc(MALLOC_ALIGN, a * b)
#define mss_aligned_realloc(a,b) aligned_realloc(MALLOC_ALIGN, a, b)
#define mss_aligned_free std::__libcpp_aligned_free

#else // !OVERRIDE_MALLOCS

#define mss_aligned_malloc(a,b) malloc(a)
#define mss_aligned_calloc calloc
#define mss_aligned_realloc realloc
#define mss_aligned_free free

#endif // !OVERRIDE_MALLOCS
#endif // DISABLE_MALLOC_FREE_OVERRIDE

#if !defined(mss_aligned_malloc)
static inline void*_Nullable mss_aligned_malloc(size_t size, size_t alignment)
{

    // Allocate extra space to store the original pointer
    void* original_memory = real_malloc(size + alignment - 1 + sizeof(void*));
    if (!original_memory) return NULL;
    
    // Calculate aligned pointer
    void* aligned_ptr = (void*)(((uintptr_t)original_memory + alignment - 1 + sizeof(void*)) & ~(alignment - 1));
    
    // Store original pointer right before the aligned pointer
    memcpy((char*)aligned_ptr - sizeof(void*), &original_memory, sizeof(void*));
    
    return aligned_ptr;
}
#endif // mss_aligned_malloc

#if !defined(mss_aligned_free)
static inline void mss_aligned_free(void*_Nullable ptr)
{
    if (ptr) {
        // Retrieve the original malloc'd pointer stored right before the aligned pointer
        void* original_ptr;
        memcpy(&original_ptr, (char*)ptr - sizeof(void*), sizeof(void*));
        real_free(original_ptr);
    }
}
#endif // mss_aligned_free

#if !defined(mss_aligned_calloc)
static inline void *_Nullable mss_aligned_calloc(size_t count, size_t size)
{
     return mss_aligned_malloc(count * size, MALLOC_ALIGN);
}
#endif

#if !defined(mss_aligned_malloc_size)
static inline size_t mss_aligned_malloc_size(const void *_Nonnull p)
{
   
    void *p_ptr;
    
    if (p) {
        memcpy(&p_ptr, (char*)p - sizeof(void*), sizeof(void*));
        return malloc_size(p_ptr);
    } else
        return 0;
}
#endif // mss_aligned_malloc_size

#if !defined(mss_aligned_realloc)
static inline void *_Nullable mss_aligned_realloc(void *_Nullable p, size_t new_size)
{
    void* new_p;
    
    if (!p) {
        return mss_aligned_malloc(new_size);
    }
    if (!(new_p = mss_aligned_malloc(new_size)))
        return nullptr;
    
    memcpy(new_p, p, mss_aligned_malloc_size(p));
    mss_aligned_free(p);
    
    return new_p;
}
#endif // mss_aligned_realloc

#if !defined(mss_aligned_malloc_usable_size)
static inline size_t mss_aligned_malloc_usable_size(const void *_Nullable p)
{
    return mss_aligned_malloc_size(p);
}
#endif // mss_aligned_malloc_usable_size

#endif // __cplusplus 

