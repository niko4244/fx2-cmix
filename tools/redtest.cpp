// Compile with the production flags (clang++-17 -O3 -march=core-avx2
// -ffp-model=fast -std=c++17) and run to answer: does clang's fast-math
// v8f32 reduction (the vshufpd/vmovshdup/vaddss tail) compute the true
// sum? Prints the compiler's reduced result vs a fixed-order sequential
// FMA sum over the same N=657-length data.
#include <cstdio>
#include <random>

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
#pragma clang fp reassociate(off) contract(off)
  for (int j = 0; j < N; ++j) seq = __builtin_fmaf(w[j], in[j], seq);
  printf("ref(fast-math tree)= %.9g\n", ref);
  printf("seq(fixed-order fma)= %.9g\n", seq);
  printf("equal: %s\n", ref == seq ? "YES" : "NO");
  sink = ref + seq;
  return sink == 0;
}
