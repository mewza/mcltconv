MCLT Convolver Test Suite
=========================

To build on OS X (tested on Sequioa and Tahoe using XCode CLI):

bash> make test
bash> ./mclt_convolver_test

=== Test: MCLT Round-trip ===
  First 4 output: 1.60006e-10 1.65263e-05 9.00423e-05 0.000256193 
  PASS

=== Test: Subband Identity ===
  MSE: 0
  PASS

=== Test: Hybrid Sparsity ===
Smooth IR:
  Cross-terms: 2061 / 131072
  Sparsity: 98.0%
  PASS

=== Test: Performance ===
  Subband: 0.17 µs/frame
  Hybrid:  7.17 µs/frame
  RT budget @ 48kHz: 10666.67 µs
  PASS

=== Test: Full Chain ===
  Peak output: 0.97
  PASS

=== Done ===
