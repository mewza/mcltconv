/*-----------------------------------------------------------------------------*\
**   MCLT Convolver v0.9 -- MCLT-Domain Convolution using MCLTReal             **
**   (c) 2021-2025 Dmitry Boldyrev. All rights reserved.                       **
**                                                                             **
**   mclt_convolver.h                                                          **
**                                                                             **
**   FFT MCLT Bundle v2.0 -- PUBLIC EDITION --                                 **
**                                                                             **
**    @contact  E-mail: subband@gmail.com or subband@protonmail.com            **
**    @home https://github.com/mewza/mcltconv                                  **
**                                                                             **
**    This software relies on use of FFTReal class available at:               **
**    https://github.com/mewza/realfft/                                        **
\*-----------------------------------------------------------------------------*/


#ifndef MCLT_CONVOLVER_H
#define MCLT_CONVOLVER_H

#include <vector>
#include <cstring>
#include <cmath>

// Assumes mclt.h is included first, bringing in:
//   - cmplxT<T>
//   - SimdBase<T>
//   - MCLTReal<T>

//=============================================================================
// Subband Convolver (Fast, Approximate)
// 
// O(M) per block. Treats each bin independently.
// Good for: EQ, smooth filters, short IRs
//=============================================================================

template<typename T>
class MCLTConvolverSubband {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    int _M;
    int _ir_blocks;
    
    // Per-bin complex gains (simple mode)
    std::vector<cmplxTT> _bin_gains;
    
    // Per-bin FIR filters (for longer IRs)
    std::vector<std::vector<cmplxTT>> _bin_filters;
    std::vector<std::vector<cmplxTT>> _bin_delay_lines;
    
    bool _use_fir;
    
public:
    MCLTConvolverSubband(MCLTReal<T>& mclt, const T1* ir, int ir_length, bool force_fir = false)
        : _M(mclt.get_M())
        , _ir_blocks((ir_length + _M - 1) / _M)
        , _use_fir(force_fir || _ir_blocks > 2)
    {
        _bin_gains.resize(_M);
        
        if (_use_fir) {
            init_fir(mclt, ir, ir_length);
        } else {
            init_gains(ir, ir_length);
        }
    }
    
    void process(const cmplxTT* __restrict__ in, cmplxTT* __restrict__ out) {
        if (_use_fir) {
            process_fir(in, out);
        } else {
            process_simple(in, out);
        }
    }
    
    void reset() {
        for (auto& dl : _bin_delay_lines) {
            for (auto& x : dl) x = cmplxTT(T1(0), T1(0));
        }
    }
    
private:
    void init_gains(const T1* ir, int ir_length) {
        // Compute H(e^jω) at MCLT bin centers: ω_k = π(k + 0.5)/M
        for (int k = 0; k < _M; k++) {
            T1 omega = M_PI * (k + T1(0.5)) / T1(_M);
            T1 re = 0, im = 0;
            
            for (int n = 0; n < ir_length; n++) {
                T1 phase = -omega * n;
                re += ir[n] * F_COS(phase);
                im += ir[n] * F_SIN(phase);
            }
            _bin_gains[k] = cmplxTT(re, im);
        }
    }
    
    void init_fir(MCLTReal<T>& mclt, const T1* ir, int ir_length) {
        _bin_filters.resize(_M);
        _bin_delay_lines.resize(_M);
        
        for (int k = 0; k < _M; k++) {
            _bin_filters[k].resize(_ir_blocks);
            _bin_delay_lines[k].resize(_ir_blocks, cmplxTT(T1(0), T1(0)));
        }
        
        int L = mclt.get_length();
        std::vector<T1> block(L, T1(0));
        std::vector<cmplxTT> mclt_block(_M);
        
        for (int b = 0; b < _ir_blocks; b++) {
            std::fill(block.begin(), block.end(), T1(0));
            
            int start = b * _M;
            int len = std::min(_M, ir_length - start);
            if (len > 0) {
                for (int i = 0; i < len; i++) {
                    block[_M/2 + i] = ir[start + i];
                }
            }
            
            mclt.analyze(block.data(), mclt_block.data());
            
            for (int k = 0; k < _M; k++) {
                _bin_filters[k][b] = mclt_block[k];
            }
        }
    }
    
    void process_simple(const cmplxTT* __restrict__ in, cmplxTT* __restrict__ out) {
        for (int k = 0; k < _M; k++) {
            out[k] = in[k] * _bin_gains[k];
        }
    }
    
    void process_fir(const cmplxTT* __restrict__ in, cmplxTT* __restrict__ out) {
        for (int k = 0; k < _M; k++) {
            // Shift delay line
            for (int d = _ir_blocks - 1; d > 0; d--) {
                _bin_delay_lines[k][d] = _bin_delay_lines[k][d - 1];
            }
            _bin_delay_lines[k][0] = in[k];
            
            // Convolve
            cmplxTT sum(T1(0), T1(0));
            for (int t = 0; t < _ir_blocks; t++) {
                sum += _bin_delay_lines[k][t] * _bin_filters[k][t];
            }
            out[k] = sum;
        }
    }
};

//=============================================================================
// Matrix Convolver (Accurate)
//
// O(M²) per block. Full cross-bin coupling.
// Good for: reference quality, any IR
//=============================================================================

template<typename T>
class MCLTConvolverMatrix {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    int _M;
    int _ir_blocks;
    int _hop;
    
    // Transfer matrices: T[b][out * M + in]
    std::vector<std::vector<cmplxTT>> _T;
    
    // Input history circular buffer
    std::vector<std::vector<cmplxTT>> _history;
    int _hist_idx;
    
public:
    MCLTConvolverMatrix(MCLTReal<T>& mclt, const T1* ir, int ir_length)
        : _M(mclt.get_M())
        , _ir_blocks((ir_length + _M - 1) / _M + 1)
        , _hop(mclt.get_hop())
        , _hist_idx(0)
    {
        build_matrices(mclt, ir, ir_length);
        
        _history.resize(_ir_blocks);
        for (auto& h : _history) {
            h.resize(_M, cmplxTT(T1(0), T1(0)));
        }
    }
    
    void process(const cmplxTT* __restrict__ in, cmplxTT* __restrict__ out) {
        // Store input
        for (int k = 0; k < _M; k++) {
            _history[_hist_idx][k] = in[k];
        }
        
        // Clear output
        for (int k = 0; k < _M; k++) {
            out[k] = cmplxTT(T1(0), T1(0));
        }
        
        // Y = Σ_b T[b] · X[m-b]
        for (int b = 0; b < _ir_blocks; b++) {
            int hi = (_hist_idx + _ir_blocks - b) % _ir_blocks;
            const cmplxTT* __restrict__ X = _history[hi].data();
            const cmplxTT* __restrict__ Tb = _T[b].data();
            
            for (int out_k = 0; out_k < _M; out_k++) {
                cmplxTT sum(T1(0), T1(0));
                for (int in_k = 0; in_k < _M; in_k++) {
                    sum += Tb[out_k * _M + in_k] * X[in_k];
                }
                out[out_k] += sum;
            }
        }
        
        _hist_idx = (_hist_idx + 1) % _ir_blocks;
    }
    
    void reset() {
        for (auto& h : _history) {
            for (auto& x : h) x = cmplxTT(T1(0), T1(0));
        }
        _hist_idx = 0;
    }
    
    int get_latency_blocks() const { return _ir_blocks; }
    
private:
    void build_matrices(MCLTReal<T>& mclt, const T1* ir, int ir_length) {
        _T.resize(_ir_blocks);
        for (auto& mat : _T) {
            mat.resize(_M * _M, cmplxTT(T1(0), T1(0)));
        }
        
        int L = mclt.get_length();
        std::vector<cmplxTT> impulse(_M);
        std::vector<T1> time_sig(L);
        std::vector<T1> conv(L + ir_length);
        std::vector<T1> block(L);
        std::vector<cmplxTT> mclt_out(_M);
        
        for (int in_k = 0; in_k < _M; in_k++) {
            // Unit impulse at bin in_k
            for (int k = 0; k < _M; k++) {
                impulse[k] = cmplxTT(k == in_k ? T1(1) : T1(0), T1(0));
            }
            
            // To time domain
            mclt.synthesize_no_tdac(impulse.data(), time_sig.data());
            
            // Convolve with IR
            std::fill(conv.begin(), conv.end(), T1(0));
            for (int n = 0; n < L; n++) {
                for (int m = 0; m < ir_length; m++) {
                    conv[n + m] += time_sig[n] * ir[m];
                }
            }
            
            // Analyze each output block
            for (int b = 0; b < _ir_blocks; b++) {
                int start = b * _hop;
                
                std::fill(block.begin(), block.end(), T1(0));
                for (int i = 0; i < L && (start + i) < (int)conv.size(); i++) {
                    block[i] = conv[start + i];
                }
                
                mclt.analyze(block.data(), mclt_out.data());
                
                for (int out_k = 0; out_k < _M; out_k++) {
                    _T[b][out_k * _M + in_k] = mclt_out[out_k];
                }
            }
        }
    }
};

//=============================================================================
// Hybrid Convolver (Best Tradeoff)
//
// O(M + k) where k = significant cross-terms
// Diagonal terms always kept, prunes small cross-bin coupling
//=============================================================================

template<typename T>
class MCLTConvolverHybrid {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    int _M;
    int _ir_blocks;
    int _hop;
    T1 _threshold;
    
    // Diagonal: _diag[k][b] = response of bin k at block offset b
    std::vector<std::vector<cmplxTT>> _diag;
    
    // Sparse cross-terms
    struct Cross {
        int from, to, block;
        cmplxTT gain;
    };
    std::vector<Cross> _cross;
    
    // History
    std::vector<std::vector<cmplxTT>> _history;
    int _hist_idx;
    
public:
    MCLTConvolverHybrid(MCLTReal<T>& mclt, const T1* ir, int ir_length, T1 thresh = T1(0.01))
        : _M(mclt.get_M())
        , _ir_blocks((ir_length + _M - 1) / _M + 1)
        , _hop(mclt.get_hop())
        , _threshold(thresh)
        , _hist_idx(0)
    {
        build_hybrid(mclt, ir, ir_length);
        
        _history.resize(_ir_blocks);
        for (auto& h : _history) {
            h.resize(_M, cmplxTT(T1(0), T1(0)));
        }
    }
    
    void process(const cmplxTT* __restrict__ in, cmplxTT* __restrict__ out) {
        // Store input
        for (int k = 0; k < _M; k++) {
            _history[_hist_idx][k] = in[k];
        }
        
        // Diagonal contributions
        for (int k = 0; k < _M; k++) {
            cmplxTT sum(T1(0), T1(0));
            for (int b = 0; b < _ir_blocks; b++) {
                int hi = (_hist_idx + _ir_blocks - b) % _ir_blocks;
                sum += _history[hi][k] * _diag[k][b];
            }
            out[k] = sum;
        }
        
        // Cross-term contributions
        for (const auto& c : _cross) {
            int hi = (_hist_idx + _ir_blocks - c.block) % _ir_blocks;
            out[c.to] += _history[hi][c.from] * c.gain;
        }
        
        _hist_idx = (_hist_idx + 1) % _ir_blocks;
    }
    
    void reset() {
        for (auto& h : _history) {
            for (auto& x : h) x = cmplxTT(T1(0), T1(0));
        }
        _hist_idx = 0;
    }
    
    size_t cross_count() const { return _cross.size(); }
    size_t full_size() const { return _M * _M * _ir_blocks; }
    T1 sparsity() const { 
        return T1(1) - T1(_cross.size() + _M * _ir_blocks) / T1(full_size()); 
    }
    
private:
    void build_hybrid(MCLTReal<T>& mclt, const T1* ir, int ir_length) {
        // Build full matrices first, then extract diagonal + significant cross
        int L = mclt.get_length();
        
        _diag.resize(_M);
        for (int k = 0; k < _M; k++) {
            _diag[k].resize(_ir_blocks, cmplxTT(T1(0), T1(0)));
        }
        
        std::vector<cmplxTT> impulse(_M);
        std::vector<T1> time_sig(L);
        std::vector<T1> conv(L + ir_length);
        std::vector<T1> block(L);
        std::vector<cmplxTT> mclt_out(_M);
        
        for (int in_k = 0; in_k < _M; in_k++) {
            for (int k = 0; k < _M; k++) {
                impulse[k] = cmplxTT(k == in_k ? T1(1) : T1(0), T1(0));
            }
            
            mclt.synthesize_no_tdac(impulse.data(), time_sig.data());
            
            std::fill(conv.begin(), conv.end(), T1(0));
            for (int n = 0; n < L; n++) {
                for (int m = 0; m < ir_length; m++) {
                    conv[n + m] += time_sig[n] * ir[m];
                }
            }
            
            for (int b = 0; b < _ir_blocks; b++) {
                int start = b * _hop;
                
                std::fill(block.begin(), block.end(), T1(0));
                for (int i = 0; i < L && (start + i) < (int)conv.size(); i++) {
                    block[i] = conv[start + i];
                }
                
                mclt.analyze(block.data(), mclt_out.data());
                
                for (int out_k = 0; out_k < _M; out_k++) {
                    T1 mag = mclt_out[out_k].mag();
                    
                    if (in_k == out_k) {
                        _diag[out_k][b] = mclt_out[out_k];
                    } else if (mag > _threshold) {
                        _cross.push_back({in_k, out_k, b, mclt_out[out_k]});
                    }
                }
            }
        }
    }
};

//=============================================================================
// Full Processor Wrapper
//
// Handles analyze -> convolve -> synthesize chain
//=============================================================================

template<typename T>
class MCLTConvolver {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
    enum Mode { SUBBAND, MATRIX, HYBRID };
    
private:
    MCLTReal<T>* _mclt;      // Non-owning pointer
    Mode _mode;
    
    // Only one is active
    std::unique_ptr<MCLTConvolverSubband<T>> _sub;
    std::unique_ptr<MCLTConvolverMatrix<T>> _mat;
    std::unique_ptr<MCLTConvolverHybrid<T>> _hyb;
    
    std::vector<cmplxTT> _buf_in;
    std::vector<cmplxTT> _buf_out;
    
public:
    MCLTConvolver(MCLTReal<T>& mclt, const T1* ir, int ir_len, Mode mode = HYBRID)
        : _mclt(&mclt)
        , _mode(mode)
        , _buf_in(mclt.get_M())
        , _buf_out(mclt.get_M())
    {
        switch (mode) {
            case SUBBAND:
                _sub = std::make_unique<MCLTConvolverSubband<T>>(mclt, ir, ir_len);
                break;
            case MATRIX:
                _mat = std::make_unique<MCLTConvolverMatrix<T>>(mclt, ir, ir_len);
                break;
            case HYBRID:
            default:
                _hyb = std::make_unique<MCLTConvolverHybrid<T>>(mclt, ir, ir_len);
                break;
        }
    }
    
    // Process time-domain frame
    void process(const T1* __restrict__ in, T1* __restrict__ out) {
        _mclt->analyze(in, _buf_in.data());
        process_mclt(_buf_in.data(), _buf_out.data());
        _mclt->synthesize(_buf_out.data());
        _mclt->overlap_add_to_buffer(out);
    }
    
    // Process MCLT-domain directly
    void process_mclt(const cmplxTT* __restrict__ in, cmplxTT* __restrict__ out) {
        switch (_mode) {
            case SUBBAND: _sub->process(in, out); break;
            case MATRIX:  _mat->process(in, out); break;
            case HYBRID:  _hyb->process(in, out); break;
        }
    }
    
    void reset() {
        if (_sub) _sub->reset();
        if (_mat) _mat->reset();
        if (_hyb) _hyb->reset();
    }
    
    int get_M() const { return _mclt->get_M(); }
    int get_hop() const { return _mclt->get_hop(); }
};

#endif // MCLT_CONVOLVER_H
