# Makefile for MCLT Convolver
# Requires: mclt.h, realfft.h, const1.h, mem.h

CXX := g++
CXXFLAGS := -std=c++23 -O3 -Wall -Wextra
LDFLAGS :=

# ============================================================================
# Platform Detection
# ============================================================================

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    # macOS
    CXX := clang++
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS += -mcpu=apple-m1
    else
        CXXFLAGS += -march=native
    endif
    LDFLAGS += -framework Accelerate
else ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -march=native
    ifeq ($(UNAME_M),aarch64)
        CXXFLAGS += -DTARGET_CPU_ARM64=1
    endif
endif

# ============================================================================
# Paths - EDIT THESE
# ============================================================================

# Path to your mclt.h, realfft.h, const1.h
INCLUDES := -I. -I./include -I/path/to/mss

# ============================================================================
# Build Variants
# ============================================================================

ifdef DEBUG
    CXXFLAGS := -std=c++17 -O0 -g -Wall -Wextra -DDEBUG
endif

ifdef FAST
    CXXFLAGS += -ffast-math -flto
    LDFLAGS += -flto
endif

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean test bench help

all: help

help:
	@echo "MCLT Convolver Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make test          - Build and run test (requires your headers)"
	@echo "  make bench         - Build with benchmarking"
	@echo "  make DEBUG=1 test  - Debug build"
	@echo "  make FAST=1 test   - Optimized build with fast-math"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
	@echo "Before building, edit INCLUDES in Makefile to point to your headers."

# Main test target
test: mclt_conv_test
	./mclt_conv_test

mclt_conv_test: mclt_convolver_test.cpp mclt_convolver.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# Benchmark
bench: CXXFLAGS += -DBENCHMARK
bench: mclt_conv_test
	./mclt_conv_test

# Clean
clean:
	rm -f mclt_conv_test *.o

# ============================================================================
# Example compile commands (for reference)
# ============================================================================

# Single file compile:
#   g++ -std=c++17 -O3 -march=native -I. -I./include your_app.cpp -o your_app
#
# Apple Silicon:
#   clang++ -std=c++17 -O3 -mcpu=apple-m1 -framework Accelerate -I. your_app.cpp -o your_app
#
# ARM64 Linux with NEON:
#   g++ -std=c++17 -O3 -march=native -DTARGET_CPU_ARM64=1 -I. your_app.cpp -o your_app
#
# Maximum optimization:
#   g++ -std=c++17 -O3 -march=native -flto -ffast-math -DNDEBUG -I. your_app.cpp -o your_app
