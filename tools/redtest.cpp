// Compile with the production flags (clang++-17 -O3 -march=core-avx2
// -ffp-model=fast -std=c++17) and run. Answers two questions:
//   1) Does clang's fast-math v8f32 reduction compute the true sum?
//   2) Is the LSTM matvec reconstruction (explicit intrinsics) bit-identical
//      to the original valarray loop compiled the same way?
#include <cstdio>
#include <cstring>
#include <random>
#include <valarray>
#include <immintrin.h>

static const int o = 256;  // output_size_ (offset into the weights row)
static const int N = 657;  // input.size() for the LSTM layer-0 matvec

// ---- the original loop, exactly as upstream writes it ----
static float old_matvec(const float* w, const float* in, int is) {
  float f = w[is];
  for (int j = 0; j < N; ++j) f += in[j] * w[o + j];
  return f;
}

// ---- the reconstructed loop: 4x8 FMA accs, seed in lane 0, half-swap
//      reduce (s0+s2)+(s1+s3), scalar FMA tail ----
static float new_matvec(const float* w, const float* in, int is) {
  const int M = (N / 32) * 32;
  float f = w[is];
  __m256 y0 = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, f);
  __m256 y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps();
  __m256 y3 = _mm256_setzero_ps();
  for (int b = 0; b < M; b += 32) {
    y0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + o + b), _mm256_loadu_ps(in + b), y0);
    y1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + o + b + 8), _mm256_loadu_ps(in + b + 8), y1);
    y2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + o + b + 16), _mm256_loadu_ps(in + b + 16), y2);
    y3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + o + b + 24), _mm256_loadu_ps(in + b + 24), y3);
  }
  __m256 t0 = _mm256_add_ps(y1, y0);
  __m256 t1 = _mm256_add_ps(y3, y2);
  __m256 t2 = _mm256_add_ps(t1, t0);
  __m128 x = _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));
  x = _mm_add_ps(x, _mm_shuffle_pd(x, x, 0x1));
  x = _mm_add_ss(x, _mm_movehdup_ps(x));
  f = _mm_cvtss_f32(x);
  {
#pragma clang fp reassociate(off) contract(off)
    for (int j = M; j < N; ++j) f = __builtin_fmaf(w[o + j], in[j], f);
  }
  return f;
}

static void test_case(const char* tag, const float* w, const float* in,
                      int is) {
  float a = old_matvec(w, in, is);
  float b = new_matvec(w, in, is);
  printf("%-14s old=%a new=%a %s\n", tag, (double)a, (double)b,
         memcmp(&a, &b, 4) == 0 ? "BIT-EQUAL" : "DIFFER");
}

int main() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  static float w[N + o], in[N];
  for (int j = 0; j < N + o; ++j) w[j] = d(rng);
  for (int j = 0; j < N; ++j) in[j] = d(rng);
  for (int is = 0; is < 4; ++is) test_case("random", w, in, is);
  // all-ones: stresses rounding boundaries
  for (int j = 0; j < N + o; ++j) w[j] = 1.0f;
  for (int j = 0; j < N; ++j) in[j] = 1.0f;
  test_case("all-ones", w, in, 3);
  // powers of two + tiny values
  for (int j = 0; j < N + o; ++j) w[j] = ldexpf(1.0f, (j % 60) - 30);
  for (int j = 0; j < N; ++j) in[j] = ldexpf(1.0f, ((j * 7) % 60) - 30);
  test_case("powers2", w, in, 1);
  return 0;
}
