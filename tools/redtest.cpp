// Compile with the production flags (clang++-17 -O3 -march=core-avx2
// -ffp-model=fast -std=c++17) and run to answer: does clang's fast-math
// v8f32 reduction (the vshufpd/vmovshdup/vaddss tail) compute the true
// sum? Prints the compiler's reduced result vs a fixed-order sequential
// FMA sum over the same N=657-length data.
#include <cstdio>
#include <random>
#include <immintrin.h>

int main() {
  const int N = 657;
  static float w[N], in[N];
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (int j = 0; j < N; ++j) {
    w[j] = d(rng);
    in[j] = d(rng);
  }
  volatile float sink = 0;
  float ref = 0;  // -ffp-model=fast: clang's own reduction tree
  for (int j = 0; j < N; ++j) ref += w[j] * in[j];
  float seq = 0;
  {
#pragma clang fp reassociate(off) contract(off)
    for (int j = 0; j < N; ++j) seq = __builtin_fmaf(w[j], in[j], seq);
  }
  // Replica of the exact instruction sequence the LSTM matvec emits:
  // 4 x 8-lane FMA accumulators over 32-element blocks, then the emitted
  // reduce tail (vaddps pairs, vextractf128+vaddps, vshufpd(copy)+vaddps,
  // vmovshdup+vaddss), then a scalar FMA tail. Intrinsics are explicit, so
  // fast-math cannot reassociate this sequence.
  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
  int M = (N / 32) * 32;
  for (int b = 0; b < M; b += 32) {
    __m256 w0 = _mm256_loadu_ps(w + b), w1 = _mm256_loadu_ps(w + b + 8);
    __m256 w2 = _mm256_loadu_ps(w + b + 16), w3 = _mm256_loadu_ps(w + b + 24);
    __m256 i0 = _mm256_loadu_ps(in + b), i1 = _mm256_loadu_ps(in + b + 8);
    __m256 i2 = _mm256_loadu_ps(in + b + 16), i3 = _mm256_loadu_ps(in + b + 24);
    y0 = _mm256_fmadd_ps(w0, i0, y0);
    y1 = _mm256_fmadd_ps(w1, i1, y1);
    y2 = _mm256_fmadd_ps(w2, i2, y2);
    y3 = _mm256_fmadd_ps(w3, i3, y3);
  }
  __m256 t = _mm256_add_ps(y1, y0);
  __m256 u = _mm256_add_ps(y3, y2);
  t = _mm256_add_ps(u, t);
  __m128 x = _mm256_castps256_ps128(t);
  __m128 hi = _mm256_extractf128_ps(t, 1);
  x = _mm_add_ps(x, hi);
  __m128 s = _mm_shuffle_pd(x, x, 0x1);  // both operands identical -> copy
  x = _mm_add_ps(x, s);
  s = _mm_movehdup_ps(x);
  x = _mm_add_ss(x, s);
  float rep = _mm_cvtss_f32(x);
  for (int j = M; j < N; ++j) rep = __builtin_fmaf(w[j], in[j], rep);
  printf("ref(fast-math tree)= %.9g\n", ref);
  printf("seq(fixed-order fma)= %.9g\n", seq);
  printf("rep(emitted tail)=   %.9g\n", rep);
  printf("rep == ref? %s   rep == seq? %s\n", rep == ref ? "YES" : "NO",
         rep == seq ? "YES" : "NO");
  sink = ref + seq + rep;
  return sink == 0;
}
