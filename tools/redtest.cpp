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
#include <math.h>

#ifndef UPDATE_LIMIT
#define UPDATE_LIMIT 3000
#endif

static const float beta1 = 0.025f, beta2 = 0.9999f, eps = 1e-6f;

// N and o are RUNTIME values (like input.size()/output_size_ in the real
// binary): constant bounds would let clang unroll differently and make the
// harness an unfaithful proxy for the valarray loop in the production build.
static int o = 256;
static int N = 657;

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
  // Same pinned reduce as the real binary: clang re-pairs the partial-sum
  // tree under fast-math (broke byte-identity). The pragma alone was NOT
  // enough in the real binary's context (pairing is encoded in register
  // allocation, invisible to instruction-level comparison), so pin the two
  // partials through a volatile round-trip — no pass can regroup them.
  {
#pragma clang fp reassociate(off) contract(off)
    __m256 t0 = _mm256_add_ps(y1, y0);
    __m256 t1 = _mm256_add_ps(y3, y2);
    volatile __m256 v0 = t0, v1 = t1;
    __m256 t2 = _mm256_add_ps(v1, v0);
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));
    x = _mm_add_ps(x, _mm_shuffle_pd(x, x, 0x1));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    f = _mm_cvtss_f32(x);
  }
  {
#pragma clang fp reassociate(off) contract(off)
    for (int j = M; j < N; ++j) f = __builtin_fmaf(w[o + j], in[j], f);
  }
  return f;
}

// ---- the LSTM output projection: same 4-acc dot but NO seed and a
//      forward tail (hidden_.size()=201 -> M=192, tail 9) ----
static float old_proj(const float* w, const float* in) {
  float sum = 0;
  for (int j = 0; j < N; ++j) sum += in[j] * w[j];
  return sum;
}

static float new_proj(const float* w, const float* in) {
  const int M = (N / 32) * 32;
  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
  for (int b = 0; b < M; b += 32) {
    y0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b), _mm256_loadu_ps(in + b), y0);
    y1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b + 8), _mm256_loadu_ps(in + b + 8), y1);
    y2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b + 16), _mm256_loadu_ps(in + b + 16), y2);
    y3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b + 24), _mm256_loadu_ps(in + b + 24), y3);
  }
  float sum;
  {
#pragma clang fp reassociate(off) contract(off)
    __m256 t0 = _mm256_add_ps(y1, y0);
    __m256 t1 = _mm256_add_ps(y3, y2);
    volatile __m256 v0 = t0, v1 = t1;
    __m256 t2 = _mm256_add_ps(v1, v0);
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));
    x = _mm_add_ps(x, _mm_shuffle_pd(x, x, 0x1));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    sum = _mm_cvtss_f32(x);
  }
  for (int j = M; j < N; ++j) sum = __builtin_fmaf(w[j], in[j], sum);
  return sum;
}

static void test_case(const char* tag, const float* w, const float* in,
                      int is) {
  float a = old_matvec(w, in, is);
  float b = new_matvec(w, in, is);
  printf("%-14s old=%a new=%a %s\n", tag, (double)a, (double)b,
         memcmp(&a, &b, 4) == 0 ? "BIT-EQUAL" : "DIFFER");
}

// ---- the LayerNorm sum: (norm_*norm_).sum(), the reversed-accumulate
//      tree with backward tail (norm_ size = num_cells_ = 200) ----
// old uses the real valarray product+sum (a plain loop is an UNFAITHFUL
// proxy: clang vectorizes it with a forward tree, not the reversed one).
static float old_sqsum(const float* x) {
  std::valarray<float> xv(x, N);
  return (xv * xv).sum();
}

static float new_sqsum(const float* x) {
  const int M = (N / 32) * 32;
  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
  float sum = 0.0f;
  if (N > 0) {
    y0 = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, x[N - 1] * x[N - 1]);
    for (int blk = N - 33; blk >= N - M - 1; blk -= 32) {
      __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(x + blk),
                                _mm256_loadu_ps(x + blk));
      __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(x + blk + 8),
                                _mm256_loadu_ps(x + blk + 8));
      __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(x + blk + 16),
                                _mm256_loadu_ps(x + blk + 16));
      __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(x + blk + 24),
                                _mm256_loadu_ps(x + blk + 24));
      g0 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g0, g0, 0x1b)), 0x4e));
      g1 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g1, g1, 0x1b)), 0x4e));
      g2 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g2, g2, 0x1b)), 0x4e));
      g3 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g3, g3, 0x1b)), 0x4e));
      // Mirror the production SumRevProduct accumulation pins: the four
      // chains are independent, so fast-math permutes which group lands in
      // which slot unless each add is routed through a volatile.
      {
#pragma clang fp reassociate(off) contract(off)
        volatile __m256 v0 = _mm256_add_ps(g3, y0);
        y0 = v0;
        volatile __m256 v1 = _mm256_add_ps(g2, y1);
        y1 = v1;
        volatile __m256 v2 = _mm256_add_ps(g1, y2);
        y2 = v2;
        volatile __m256 v3 = _mm256_add_ps(g0, y3);
        y3 = v3;
      }
    }
    {
#pragma clang fp reassociate(off) contract(off)
      // Mirror the production SumRevProduct: partial sums pinned through
      // volatile round-trips (fast-math re-pairs the combine in the big
      // function even when the pragma holds standalone).
      volatile __m256 v = _mm256_add_ps(y1, y0);
      __m256 a0 = v;
      v = _mm256_add_ps(y2, a0);
      a0 = v;
      v = _mm256_add_ps(y3, a0);
      a0 = v;
      __m128 xv = _mm_add_ps(_mm256_castps256_ps128(a0),
                             _mm256_extractf128_ps(a0, 1));
      xv = _mm_add_ps(xv, _mm_shuffle_pd(xv, xv, 0x1));
      xv = _mm_add_ss(xv, _mm_movehdup_ps(xv));
      sum = _mm_cvtss_f32(xv);
    }
    {
#pragma clang fp reassociate(off) contract(off)
      for (int i = (int)((N - 1) & 31) - 1; i >= 0; --i)
        sum = __builtin_fmaf(x[i], x[i], sum);
    }
  }
  return sum;
}

static void proj_case(const char* tag, const float* w, const float* in) {
  float a = old_proj(w, in);
  float b = new_proj(w, in);
  printf("%-14s old=%a new=%a %s\n", tag, (double)a, (double)b,
         memcmp(&a, &b, 4) == 0 ? "BIT-EQUAL" : "DIFFER");
}

// ---- softmax sum: seed o[0] in lane 0, vector over elements 1..M where
//      M = (N-1)&~31, sequential scalar tail, then * (1/sum) ----
// old uses the real valarray .sum() (a plain loop is an UNFAITHFUL proxy:
// clang vectorizes it differently than the valarray method).
static float old_sftsum(const float* x) {
  std::valarray<float> xv(x, N);
  return xv.sum();
}

static float new_sftsum(const float* x) {
  const int M = ((N - 1) & ~31);
  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
  if (N > 1) {
    y0 = _mm256_blend_ps(y0, _mm256_set1_ps(x[0]), 0x1);
    for (int b = 0; b < M; b += 32) {
      y0 = _mm256_add_ps(y0, _mm256_loadu_ps(x + b + 1));
      y1 = _mm256_add_ps(y1, _mm256_loadu_ps(x + b + 9));
      y2 = _mm256_add_ps(y2, _mm256_loadu_ps(x + b + 17));
      y3 = _mm256_add_ps(y3, _mm256_loadu_ps(x + b + 25));
    }
    float sum;
    {
#pragma clang fp reassociate(off) contract(off)
      __m256 t0 = _mm256_add_ps(y1, y0);
      __m256 t1 = _mm256_add_ps(y3, y2);
      volatile __m256 v0 = t0, v1 = t1;
      __m256 t2 = _mm256_add_ps(v1, v0);
      __m128 xv = _mm_add_ps(_mm256_castps256_ps128(t2),
                             _mm256_extractf128_ps(t2, 1));
      xv = _mm_add_ps(xv, _mm_shuffle_pd(xv, xv, 0x1));
      xv = _mm_add_ss(xv, _mm_movehdup_ps(xv));
      sum = _mm_cvtss_f32(xv);
    }
    {
#pragma clang fp reassociate(off) contract(off)
      for (int j = M + 1; j < N; ++j) sum += x[j];
    }
    return sum;
  }
  return N == 1 ? x[0] : 0.0f;
}

// ---- the four gate-error elementwise chains in BackwardPass ----
// Chain 1: out = tanh*stored*state*(1-state)   [uses p1,p2,p3]
static void old_chain1(float* o, const float* a, const float* b,
                       const float* c, const float*, const float*) {
  for (int j = 0; j < N; ++j) o[j] = a[j] * b[j] * c[j] * (1.0f - c[j]);
}
static void new_chain1(float* o, const float* a, const float* b,
                       const float* c, const float*, const float*) {
#pragma clang fp reassociate(off) contract(off)
  for (int j = 0; j < N; ++j) {
    float t = a[j] * b[j];
    t = t * c[j];
    o[j] = __builtin_fmaf(-t, c[j], t);
  }
}
// Chain 2: out += stored*state*(1-tanh^2)   [uses p1,p2,p3]
static void old_chain2(float* o, const float* s, const float* t,
                       const float* st, const float*, const float*) {
  for (int j = 0; j < N; ++j) o[j] += s[j] * st[j] * (1.0f - t[j] * t[j]);
}
static void new_chain2(float* o, const float* s, const float* t,
                       const float* st, const float*, const float*) {
#pragma clang fp reassociate(off) contract(off)
  for (int j = 0; j < N; ++j) {
    const float t1 = s[j] * st[j];
    const float t2 = __builtin_fmaf(t[j], t[j], -1.0f);
    o[j] = __builtin_fmaf(-t2, t1, o[j]);
  }
}
// Chain 3: out = state_err*ig_state*(1-in^2)   [uses p1,p2,p3]
static void old_chain3(float* o, const float* e, const float* g,
                       const float* x, const float*, const float*) {
  for (int j = 0; j < N; ++j) o[j] = e[j] * g[j] * (1.0f - x[j] * x[j]);
}
static void new_chain3(float* o, const float* e, const float* g,
                       const float* x, const float*, const float*) {
#pragma clang fp reassociate(off) contract(off)
  for (int j = 0; j < N; ++j) {
    const float t1 = e[j] * g[j];
    const float t2 = x[j] * x[j];
    o[j] = __builtin_fmaf(-t2, t1, t1);
  }
}
// Chain 4: out = (last-in)*state_err*forget_state*ig_state [uses all]
static void old_chain4(float* o, const float* l, const float* in,
                       const float* e, const float* f, const float* g) {
  for (int j = 0; j < N; ++j)
    o[j] = (l[j] - in[j]) * e[j] * f[j] * g[j];
}
static void new_chain4(float* o, const float* l, const float* in,
                       const float* e, const float* f, const float* g) {
#pragma clang fp reassociate(off) contract(off)
  // Mirror the production chain 4: intermediates pinned through volatiles
  // so fast-math cannot re-pair ((l-in)*e)*f)*ig into ((l-in)*e)*(f*ig).
  for (int j = 0; j < N; ++j) {
    float t = l[j] - in[j];
    volatile float v1 = t * e[j];
    t = v1 * f[j];
    volatile float v2 = t;
    o[j] = v2 * g[j];
  }
}

typedef void (*chainfn)(float*, const float*, const float*, const float*,
    const float*, const float*);

static void chain_case(const char* tag, chainfn oldf, chainfn newf,
    float* oa, float* ob, float* p1, float* p2, float* p3, float* p4,
    float* p5) {
  oldf(oa, p1, p2, p3, p4, p5);
  newf(ob, p1, p2, p3, p4, p5);
  int diff = 0;
  for (int j = 0; j < N; ++j)
    if (memcmp(&oa[j], &ob[j], 4) != 0) { diff = j; break; }
  printf("%-14s %s (first diff @%d: %a vs %a)\n", tag,
         diff == 0 ? "BIT-EQUAL" : "DIFFER", diff, (double)oa[diff],
         (double)ob[diff]);
}

// ---- Adam: alpha (t<LIMIT: rsqrt+Newton; else: folded) ----
// The disasm job builds this harness with -fno-inline, so every old_*/new_*
// function compiles standalone (a context-independent instruction-level
// oracle). The explicit noinline attributes below keep that guarantee even
// for local builds that omit the flag.
__attribute__((noinline)) static float old_alpha(float t, float lr) {
  float alpha;
  if (t < UPDATE_LIMIT) {
    alpha = lr * 0.1f / sqrt(5e-5f * t + 1.0f);
  } else {
    alpha = lr * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
  }
  return alpha;
}
__attribute__((noinline)) static float new_alpha(float t, float lr) {
  float alpha;
  if (t < UPDATE_LIMIT) {
#pragma clang fp reassociate(off) contract(off)
    {
      const float x = __builtin_fmaf(5e-5f, t, 1.0f);
      const float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
      alpha = ((lr * 0.1f) * (r * -0.5f)) *
          __builtin_fmaf(r, x * r, -3.0f);
    }
  } else {
    alpha = lr * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
  }
  return alpha;
}

// ---- full Adam elementwise updates (m/v/w) at a fixed t ----
__attribute__((noinline)) static void old_adam(float* g, float* m, float* v,
                                               float* w, float lr, float t) {
  float alpha = old_alpha(t, lr);
  for (int j = 0; j < N; ++j) {
    m[j] *= beta1;
    m[j] += (1.0f - beta1) * g[j];
    v[j] *= beta2;
    v[j] += (1.0f - beta2) * g[j] * g[j];
    if (t < UPDATE_LIMIT) {
      w[j] -= alpha * (((m[j]) / (float)(1.0f - pow(beta1, t))) /
          (sqrt((v[j]) / (float)(1.0f - pow(beta2, t)) + eps)));
    } else {
      w[j] -= alpha * (((m[j]) / (float)(1.0f - pow(beta1, UPDATE_LIMIT))) /
          (sqrt((v[j]) / (float)(1.0f - pow(beta2, UPDATE_LIMIT)) + eps)));
    }
  }
}
__attribute__((noinline)) static void new_adam(float* g, float* m,
                                                  float* v, float* w,
                                                  float lr, float t) {
  float alpha = new_alpha(t, lr);
  float den1, inv_den2;
  if (t < UPDATE_LIMIT) {
    den1 = 1.0f - exp2f(t * __builtin_log2f(beta1));
    inv_den2 = 1.0f / (1.0f - exp2f(t * __builtin_log2f(beta2)));
  } else {
    den1 = (float)(1.0f - pow(beta1, UPDATE_LIMIT));
    inv_den2 = 1.0f / (float)(1.0f - pow(beta2, UPDATE_LIMIT));
  }
  for (int j = 0; j < N; ++j) {
    m[j] *= beta1;
    m[j] = __builtin_fmaf((1.0f - beta1), g[j], m[j]);
    v[j] *= beta2;
    v[j] = __builtin_fmaf(g[j] * g[j], (1.0f - beta2), v[j]);
    const float xv = __builtin_fmaf(v[j], inv_den2, eps);
    const float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(xv)));
    // +0.5: the w-path Newton constant (rodata 39220) is the negative of
    // the alpha path's (39204), giving rr ~ -1/sqrt(xv) so the emitted
    // w + alpha*m*rr equals the source's subtraction.
    const float rr = (r * 0.5f) * __builtin_fmaf(r, xv * r, -3.0f);
    if (t < UPDATE_LIMIT) {
      const float rcp = _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(den1)));
      const float t1 = rr * rcp;
      const float t2 = __builtin_fmaf(rr, den1, -t1);
      const float q = __builtin_fmaf(-t2, rcp, t1);
      w[j] = __builtin_fmaf(alpha * m[j], q, w[j]);
    } else {
      w[j] = __builtin_fmaf(alpha * m[j], rr, w[j]);
    }
  }
}

static void adam_case(const char* tag, float t, float lr) {
  static float g[5120], m1[5120], m2[5120], v1[5120], v2[5120], w1[5120],
      w2[5120];
  std::mt19937 rng(999);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (int j = 0; j < N; ++j) g[j] = d(rng);
  for (int j = 0; j < N; ++j) m1[j] = m2[j] = d(rng);
  for (int j = 0; j < N; ++j) v1[j] = v2[j] = d(rng);
  for (int j = 0; j < N; ++j) w1[j] = w2[j] = d(rng);
  old_adam(g, m1, v1, w1, lr, t);
  new_adam(g, m2, v2, w2, lr, t);
  int diff = 0;
  for (int j = 0; j < N; ++j)
    if (memcmp(&w1[j], &w2[j], 4) != 0) { diff = j; break; }
  printf("%-14s %s (first w diff @%d: %a vs %a)\n", tag,
         diff == 0 ? "BIT-EQUAL" : "DIFFER", diff, (double)w1[diff],
         (double)w2[diff]);
  diff = 0;
  for (int j = 0; j < N; ++j)
    if (memcmp(&m1[j], &m2[j], 4) != 0) { diff = j; break; }
  printf("%-14s m %s%s\n", tag, diff == 0 ? "BIT-EQUAL" : "DIFFER",
         diff ? "" : "");
}

int main(int argc, char** argv) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  static float w[5120], in[5120];
  // 457 is the real LSTM gate-matvec length (1 + num_cells + input_size =
  // 1 + 200 + 256); 657/201 probe tail lengths 17/9 (tail-9 is the real
  // binary's tail and the case the standalone harness originally missed).
  const int cases_n[] = {657, 457, 201, 128, 40, 33, 32, 31, 17};
  for (int c = 0; c < 8; ++c) {
    N = cases_n[c];
    o = 256;
    for (int j = 0; j < N + o; ++j) w[j] = d(rng);
    for (int j = 0; j < N; ++j) in[j] = d(rng);
    char tag[32];
    snprintf(tag, sizeof tag, "N=%d", N);
    test_case(tag, w, in, 1 + (c % 3));
  }
  // all-ones: exact arithmetic, any reassociation must agree
  N = 657;
  for (int j = 0; j < N + o; ++j) w[j] = 1.0f;
  for (int j = 0; j < N; ++j) in[j] = 1.0f;
  test_case("all-ones", w, in, 3);
  // Output projection (no seed): the real length is hidden_.size()=201.
  for (int c = 0; c < 3; ++c) {
    N = cases_n[c == 0 ? 1 : c == 1 ? 4 : 0];  // 457, 128, 657
    for (int j = 0; j < N; ++j) w[j] = d(rng);
    for (int j = 0; j < N; ++j) in[j] = d(rng);
    char tag[32];
    snprintf(tag, sizeof tag, "proj N=%d", N);
    proj_case(tag, w, in);
  }
  N = 201;
  for (int j = 0; j < N; ++j) w[j] = d(rng);
  for (int j = 0; j < N; ++j) in[j] = d(rng);
  proj_case("proj N=201", w, in);
  // LayerNorm reversed sum: real length num_cells_=200 (also probe 201/33).
  for (int c = 0; c < 3; ++c) {
    N = c == 0 ? 200 : c == 1 ? 201 : 33;
    for (int j = 0; j < N; ++j) in[j] = d(rng);
    char tag[32];
    snprintf(tag, sizeof tag, "sqsum N=%d", N);
    float a = old_sqsum(in);
    float b = new_sqsum(in);
    printf("%-14s old=%a new=%a %s\n", tag, (double)a, (double)b,
           memcmp(&a, &b, 4) == 0 ? "BIT-EQUAL" : "DIFFER");
  }
  // Softmax sum (seed-first): real length output_size_=256.
  for (int c = 0; c < 3; ++c) {
    N = c == 0 ? 256 : c == 1 ? 201 : 33;
    for (int j = 0; j < N; ++j) in[j] = d(rng);
    char tag[32];
    snprintf(tag, sizeof tag, "sft N=%d", N);
    float a = old_sftsum(in);
    float b = new_sftsum(in);
    printf("%-14s old=%a new=%a %s\n", tag, (double)a, (double)b,
           memcmp(&a, &b, 4) == 0 ? "BIT-EQUAL" : "DIFFER");
  }
  // Transpose matvec forward pattern at the real length num_cells_=200.
  N = 200;
  for (int j = 0; j < N; ++j) w[j] = d(rng);
  for (int j = 0; j < N; ++j) in[j] = d(rng);
  proj_case("proj N=200", w, in);
  // The four gate-error elementwise chains (N=200 real length).
  {
    static float ca[5120], cb[5120], p1[5120], p2[5120], p3[5120], p4[5120],
        p5[5120];
    for (int j = 0; j < N; ++j) {
      p1[j] = d(rng); p2[j] = d(rng); p3[j] = d(rng); p4[j] = d(rng);
      p5[j] = d(rng);
    }
    chain_case("chain1", old_chain1, new_chain1, ca, cb, p1, p2, p3, p4, p5);
    chain_case("chain2", old_chain2, new_chain2, ca, cb, p1, p2, p3, p4, p5);
    chain_case("chain3", old_chain3, new_chain3, ca, cb, p1, p2, p3, p4, p5);
    chain_case("chain4", old_chain4, new_chain4, ca, cb, p1, p2, p3, p4, p5);
  }
  // Adam alpha + full updates at t < LIMIT and t >= LIMIT.
  for (int c = 0; c < 4; ++c) {
    float t = c == 0 ? 1.0f : c == 1 ? 100.0f : c == 2 ? 3000.0f : 5000.0f;
    char tag[32];
    snprintf(tag, sizeof tag, "alpha t=%.0f", (double)t);
    float a = old_alpha(t, 0.1f);
    float b = new_alpha(t, 0.1f);
    printf("%-14s old=%a new=%a %s\n", tag, (double)a, (double)b,
           memcmp(&a, &b, 4) == 0 ? "BIT-EQUAL" : "DIFFER");
  }
  for (int c = 0; c < 4; ++c) {
    float t = c == 0 ? 1.0f : c == 1 ? 100.0f : c == 2 ? 3000.0f : 5000.0f;
    char tag[32];
    snprintf(tag, sizeof tag, "adam t=%.0f", (double)t);
    adam_case(tag, t, 0.1f);
  }
  // Frozen constants for the reconstruction literals.
  printf("FOLD log2f(beta1)=%a\n", (double)__builtin_log2f(beta1));
  printf("FOLD log2f(beta2)=%a\n", (double)__builtin_log2f(beta2));
  printf("FOLD alpha_const=%a\n",
         (double)(0.1f / sqrtf(5e-5f * UPDATE_LIMIT + 1.0f)));
  printf("FOLD den1=%a\n", (double)(1.0f - pow(beta1, (float)UPDATE_LIMIT)));
  printf("FOLD inv_den2=%a\n",
         (double)(1.0f / (1.0f - pow(beta2, (float)UPDATE_LIMIT))));
  return 0;
}
