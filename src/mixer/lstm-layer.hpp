#include "lstm-layer.h"

#include "sigmoid.h"

#include <math.h>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

#define FAST_TANH tanh //fast_tanh
#define FAST_TANH_VEC tanh //fast_tanh_vec
namespace {
// inline float fast_tanh(const float x)
// {
//     const float ax = fabs(x);
//     const float x2 = x * x;

//     return(x * (2.45550750702956f + 2.45550750702956f * ax +
//         (0.893229853513558f + 0.821226666969744f * ax) * x2) /
//         (2.44506634652299f + (2.44506634652299f + x2) *
//             fabs(x + 0.814642734961073f * x * ax)));
// }

// float fast_tanh(float x){
//   float x2 = x * x;
//   float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
//   float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
//   return a / b;
// }
// 
// template <class _Tp>
// inline std::valarray<_Tp> fast_tanh_vec(const std::valarray<_Tp>& __x) {
//   std::valarray<_Tp> __tmp(__x.size());
//   for (size_t __i = 0; __i < __x.size(); ++__i)
//     __tmp[__i] = fast_tanh(__x[__i]);
//   return __tmp;
// }

// Reversed-accumulate dot-product sum, reproducing clang-17's exact codegen
// for the valarray `.sum()` reductions in the LSTM (norm_*norm_ and
// error_*norm_): the LAST element's product seeds accumulator lane 0, the
// remaining elements accumulate 8-wide from the top down in REVERSED lane
// order (vshufps 0x1b + vpermpd 0x4e), the four accumulators combine
// sequentially ((y0+y1)+y2)+y3, and the scalar tail runs BACKWARD. Every
// chain is dependent (no pairing freedom), so the FP tree is fixed by
// construction regardless of compiler version or flags.
inline float SumRevProduct(const float* a, const float* b, unsigned int n) {
  const int M = (int)((n / 32) * 32);
  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
  float sum = 0.0f;
  if (n > 0) {
    y0 = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                       a[n - 1] * b[n - 1]);
    for (int blk = (int)n - 33; blk >= (int)n - M - 1; blk -= 32) {
      __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(a + blk),
                                _mm256_loadu_ps(b + blk));
      __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(a + blk + 8),
                                _mm256_loadu_ps(b + blk + 8));
      __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(a + blk + 16),
                                _mm256_loadu_ps(b + blk + 16));
      __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(a + blk + 24),
                                _mm256_loadu_ps(b + blk + 24));
      g0 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g0, g0, 0x1b)), 0x4e));
      g1 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g1, g1, 0x1b)), 0x4e));
      g2 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g2, g2, 0x1b)), 0x4e));
      g3 = _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(_mm256_shuffle_ps(g3, g3, 0x1b)), 0x4e));
      // The four accumulator chains are independent, so fast-math can
      // permute WHICH group lands in WHICH slot (observed in-context:
      // y1/y2/y3 held the wrong groups, changing the partial-sum tree and
      // the final rounded value). Route each add through a volatile
      // round-trip so the y0+=g3/y1+=g2/y2+=g1/y3+=g0 mapping is fixed by
      // construction (same class of bug as the combine re-pairing).
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
      // The combine (((y1+y0)+y2)+y3) is a dependent chain, but fast-math
      // re-pairs it in the big-function context (the pragma alone does not
      // hold there — the identity gate caught the same class of bug in the
      // ForwardPass matvec reduce). Route each partial sum through a
      // volatile round-trip so no pass can regroup the tree.
      volatile __m256 v = _mm256_add_ps(y1, y0);
      __m256 a0 = v;
      v = _mm256_add_ps(y2, a0);
      a0 = v;
      v = _mm256_add_ps(y3, a0);
      a0 = v;
      __m128 x = _mm_add_ps(_mm256_castps256_ps128(a0),
                            _mm256_extractf128_ps(a0, 1));
      x = _mm_add_ps(x, _mm_shuffle_pd(x, x, 0x1));
      x = _mm_add_ss(x, _mm_movehdup_ps(x));
      sum = _mm_cvtss_f32(x);
    }
    {
#pragma clang fp reassociate(off) contract(off)
      for (int i = (int)((n - 1) & 31) - 1; i >= 0; --i)
        sum = __builtin_fmaf(a[i], b[i], sum);
    }
  }
  return sum;
}

// Forward-accumulate dot product, reproducing clang-17's codegen for the
// transpose matvecs in BackwardPass (`f += error_[j] * transpose_[r][j]`,
// num_cells_=200 -> M=192, tail 8): ZERO seed, 4 x 8-lane FMA accumulators
// over 32-element blocks, reduce (y1+y0)+(y3+y2) pinned through a volatile
// round-trip (fast-math re-pairs the partial-sum tree in register
// allocation), forward scalar FMA tail. Every chain is fixed by
// construction.
inline float DotForwardFma(const float* a, const float* b, unsigned int n) {
  const int M = (int)((n / 32) * 32);
  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
  __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
  for (int blk = 0; blk < M; blk += 32) {
    y0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + blk),
                         _mm256_loadu_ps(b + blk), y0);
    y1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + blk + 8),
                         _mm256_loadu_ps(b + blk + 8), y1);
    y2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + blk + 16),
                         _mm256_loadu_ps(b + blk + 16), y2);
    y3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + blk + 24),
                         _mm256_loadu_ps(b + blk + 24), y3);
  }
  float sum;
  {
#pragma clang fp reassociate(off) contract(off)
    __m256 t0 = _mm256_add_ps(y1, y0);
    __m256 t1 = _mm256_add_ps(y3, y2);
    volatile __m256 v0 = t0, v1 = t1;
    __m256 t2 = _mm256_add_ps(v1, v0);
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(t2),
                          _mm256_extractf128_ps(t2, 1));
    x = _mm_add_ps(x, _mm_shuffle_pd(x, x, 0x1));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    sum = _mm_cvtss_f32(x);
  }
  {
#pragma clang fp reassociate(off) contract(off)
    for (int j = M; j < (int)n; ++j)
      sum = __builtin_fmaf(a[j], b[j], sum);
  }
  return sum;
}

inline void Adam(std::valarray<float>* g, std::valarray<float>* m,
    std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
    float t) {
  const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f; 
  float alpha;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f); 
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f); 
  }
  (*m) *= beta1;
  (*m) += (1.0f - beta1) * (*g);
  (*v) *= beta2;
  (*v) += (1.0f - beta2) * (*g) * (*g);
  if (t < UPDATE_LIMIT) {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))) /
        (sqrt((*v) / (float)(1.0f - pow(beta2, t)) + eps)));
  } else {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, UPDATE_LIMIT))) /
        (sqrt((*v) / (float)(1.0f - pow(beta2, UPDATE_LIMIT)) + eps)));
  }
}

}

inline LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float gradient_clip, float learning_rate) :
    state_(num_cells), state_error_(num_cells), stored_error_(num_cells),
    tanh_state_(std::valarray<float>(num_cells), horizon),
    input_gate_state_(std::valarray<float>(num_cells), horizon),
    last_state_(std::valarray<float>(num_cells), horizon),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    input_node_(input_size, num_cells, horizon, output_size_ + input_size_),
    output_gate_(input_size, num_cells, horizon, output_size_ + input_size_) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_.weights_[i].size(); ++j) {
      forget_gate_.weights_[i][j] = low + Rand() * range;
      input_node_.weights_[i][j] = low + Rand() * range;
      output_gate_.weights_[i][j] = low + Rand() * range;
    }
    forget_gate_.weights_[i][forget_gate_.weights_[i].size() - 1] = 1;
  }
}

inline void LstmLayer::ForwardPass(const std::valarray<float>& input, int input_symbol,
    std::valarray<float>* hidden, int hidden_start) {
  last_state_[epoch_] = state_;
  ForwardPass(forget_gate_, input, input_symbol);
  ForwardPass(input_node_, input, input_symbol);
  ForwardPass(output_gate_, input, input_symbol);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    forget_gate_.state_[epoch_][i] = Sigmoid::Logistic(
        forget_gate_.state_[epoch_][i]);
    input_node_.state_[epoch_][i] = FAST_TANH(input_node_.state_[epoch_][i]);
    output_gate_.state_[epoch_][i] = Sigmoid::Logistic(
        output_gate_.state_[epoch_][i]);
  }
  input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
  state_ *= forget_gate_.state_[epoch_];
  state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
  tanh_state_[epoch_] = FAST_TANH_VEC(state_);
  std::slice slice = std::slice(hidden_start, num_cells_, 1);
  (*hidden)[slice] = output_gate_.state_[epoch_] * tanh_state_[epoch_];
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void LstmLayer::ForwardPass(NeuronLayer& neurons,
    const std::valarray<float>& input, int input_symbol) {
  // The matvec below is reconstructed BY CONSTRUCTION from the exact
  // instruction sequence clang-17 emits for the original valarray loop at
  // -O3 -march=core-avx2 -ffp-model=fast (see tools/redtest.cpp and the
  // disasm job): 4 x 8-lane FMA accumulators over 32-element blocks, the
  // seed (weights_[i][input_symbol]) fused into accumulator lane 0, then
  // the horizontal reduce (pair adds, extract128+add, 64-bit-half swap
  // vshufpd+add, vmovshdup+vaddss), then a scalar FMA tail. Intrinsics are
  // explicit, so no compiler can reassociate or re-contract them: the FP
  // operation sequence — and therefore the compressed bytes — is fixed
  // regardless of compiler version or vectorizer behavior.
  const int N = (int)input.size();
  const int M = (N / 32) * 32;
  const float* in = N > 0 ? &input[0] : nullptr;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    const float* w = &neurons.weights_[i][0];
    float f = w[input_symbol];
    if (N >= 32) {
      __m256 y0 = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, f);
      __m256 y1 = _mm256_setzero_ps();
      __m256 y2 = _mm256_setzero_ps();
      __m256 y3 = _mm256_setzero_ps();
      for (int b = 0; b < M; b += 32) {
        y0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + output_size_ + b),
                             _mm256_loadu_ps(in + b), y0);
        y1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + output_size_ + b + 8),
                             _mm256_loadu_ps(in + b + 8), y1);
        y2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + output_size_ + b + 16),
                             _mm256_loadu_ps(in + b + 16), y2);
        y3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + output_size_ + b + 24),
                             _mm256_loadu_ps(in + b + 24), y3);
      }
      // The horizontal reduce must NOT be re-paired. The individual
      // __m256 adds are commutative (a+b == b+a bit-for-bit), so operand
      // order within each add is free — but the PARTIAL-SUM TREE is not:
      // under -ffp-model=fast, clang's backend reassociated the tree here,
      // pairing (y2+y1),(y3+y0) instead of the emitted loop's
      // (y1+y0),(y3+y2) — same operations, different rounding, different
      // bytes. #pragma clang fp reassociate(off) was NOT sufficient in this
      // context (it held in the standalone harness but the larger function
      // re-paired anyway — that pairing is encoded in register allocation,
      // invisible to instruction-level comparisons). The volatile round-trip
      // makes the two partial sums distinct, memory-pinned values that no
      // pass can regroup: the tree (y1+y0)+(y3+y2) is fixed by construction.
      {
#pragma clang fp reassociate(off) contract(off)
        __m256 t0 = _mm256_add_ps(y1, y0);
        __m256 t1 = _mm256_add_ps(y3, y2);
        volatile __m256 v0 = t0, v1 = t1;
        __m256 t2 = _mm256_add_ps(v1, v0);
        __m128 x = _mm_add_ps(_mm256_castps256_ps128(t2),
                              _mm256_extractf128_ps(t2, 1));
        x = _mm_add_ps(x, _mm_shuffle_pd(x, x, 0x1));
        x = _mm_add_ss(x, _mm_movehdup_ps(x));
        f = _mm_cvtss_f32(x);
      }
      // Scalar FMA tail; reassociate(off) keeps it sequential like the
      // emitted loop (fast-math would otherwise re-tree it).
      {
#pragma clang fp reassociate(off) contract(off)
        for (int j = M; j < N; ++j)
          f = __builtin_fmaf(w[output_size_ + j], in[j], f);
      }
    } else {
      // Small-input path: the emitted code skips the vector loop entirely.
      {
#pragma clang fp reassociate(off) contract(off)
        for (int j = 0; j < N; ++j)
          f = __builtin_fmaf(w[output_size_ + j], in[j], f);
      }
    }
    neurons.norm_[epoch_][i] = f;
  }
  // LayerNorm ivar reconstructed BY CONSTRUCTION. The sum uses the same
  // reversed-accumulate tree as the emitted (norm_*norm_).sum(); then
  // 1/sqrt(x) becomes the exact emitted vrsqrtss + one-Newton sequence:
  // r = rsqrt(x); ivar = (r*-0.5) * fma(r, x*r, -3.0) — the Newton
  // constants follow uniquely from the target 1/sqrt(x) (C2=-1/2, C1=-3).
  {
    const float* nx = &neurons.norm_[epoch_][0];
    const float xv = SumRevProduct(nx, nx, num_cells_) / (float)num_cells_ +
        1e-5f;
    const float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(xv)));
    {
#pragma clang fp reassociate(off) contract(off)
      neurons.ivar_[epoch_] =
          (r * -0.5f) * __builtin_fmaf(r, xv * r, -3.0f);
    }
  }
  neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
  // state = norm*gamma + beta — elementwise mul+add; pinned as an FMA
  // (the fast-math contraction, one rounding).
  for (unsigned int j = 0; j < num_cells_; ++j) {
    neurons.state_[epoch_][j] = __builtin_fmaf(neurons.norm_[epoch_][j],
        neurons.gamma_[j], neurons.beta_[j]);
  }
}

inline void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
  }
}

inline void LstmLayer::BackwardPass(const std::valarray<float>&input, int epoch,
    int layer, int input_symbol, std::valarray<float>* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
  } else {
    stored_error_ += *hidden_error;
  }

  // The four gate-error elementwise chains, reconstructed BY CONSTRUCTION
  // from the emitted clang-17 codegen (tools/redtest.cpp validates each
  // form). The emitted associations are: (1) t = (tanh*stored)*state then
  // out = t - t*state (vfnmadd); (2) t1 = stored*state,
  // t2 = tanh*tanh - 1 (fmsub), out = fma(-t2, t1, out);
  // (3) t1 = state_err*ig_state, t2 = in*in, out = fma(-t2, t1, t1);
  // (4) (((last-in)*state_err)*forget_state)*ig_state. The pure-mul
  // chains are fully dependent (no pairing freedom), so plain muls under
  // reassociate(off)+contract(off) reproduce the emitted vmulss exactly
  // (fmaf(x,y,0) would flip -0 to +0).
  {
    const float* t = &tanh_state_[epoch][0];
    const float* s = &stored_error_[0];
    const float* gs = &output_gate_.state_[epoch][0];
    float* o = &output_gate_.error_[0];
    {
#pragma clang fp reassociate(off) contract(off)
      for (unsigned int j = 0; j < num_cells_; ++j) {
        float p = t[j] * s[j];
        p = p * gs[j];
        o[j] = __builtin_fmaf(-p, gs[j], p);
      }
    }
  }
  {
    const float* s = &stored_error_[0];
    const float* gs = &output_gate_.state_[epoch][0];
    const float* t = &tanh_state_[epoch][0];
    float* o = &state_error_[0];
    {
#pragma clang fp reassociate(off) contract(off)
      for (unsigned int j = 0; j < num_cells_; ++j) {
        const float t1 = s[j] * gs[j];
        const float t2 = __builtin_fmaf(t[j], t[j], -1.0f);
        o[j] = __builtin_fmaf(-t2, t1, o[j]);
      }
    }
  }
  {
    const float* e = &state_error_[0];
    const float* ig = &input_gate_state_[epoch][0];
    const float* x = &input_node_.state_[epoch][0];
    float* o = &input_node_.error_[0];
    {
#pragma clang fp reassociate(off) contract(off)
      for (unsigned int j = 0; j < num_cells_; ++j) {
        const float t1 = e[j] * ig[j];
        const float t2 = x[j] * x[j];
        o[j] = __builtin_fmaf(-t2, t1, t1);
      }
    }
  }
  {
    const float* l = &last_state_[epoch][0];
    const float* x = &input_node_.state_[epoch][0];
    const float* e = &state_error_[0];
    const float* f = &forget_gate_.state_[epoch][0];
    const float* ig = &input_gate_state_[epoch][0];
    float* o = &forget_gate_.error_[0];
    {
#pragma clang fp reassociate(off) contract(off)
      // Dependent mul chain pinned through volatile round-trips: the
      // emitted sequence is (((l-x)*e)*f)*ig and fast-math must not be
      // able to re-pair it into ((l-x)*e)*(f*ig) (which the pragma alone
      // does not reliably prevent in the big-function context).
      for (unsigned int j = 0; j < num_cells_; ++j) {
        float p = l[j] - x[j];
        volatile float v1 = p * e[j];
        p = v1 * f[j];
        volatile float v2 = p;
        o[j] = v2 * ig[j];
      }
    }
  }

  *hidden_error = 0;
  if (epoch > 0) {
    state_error_ *= forget_gate_.state_[epoch];
    stored_error_ = 0;
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void LstmLayer::BackwardPass(NeuronLayer& neurons,
    const std::valarray<float>&input, int epoch, int layer, int input_symbol,
    std::valarray<float>* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_ = 0;
    neurons.beta_u_ = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.update_[i] = 0;
      int offset = output_size_ + input_size_;
      for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
        neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
      }
    }
  }
  // All elementwise chains reconstructed BY CONSTRUCTION from the emitted
  // clang-17 codegen (tools/redtest.cpp validates each form): beta_u_ stays
  // a plain add; gamma_u_ is an FMA; error_ *= gamma_*ivar_ is the emitted
  // (ivar*gamma)*error two-mul chain — fully dependent, plain muls pinned
  // with reassociate(off)+contract(off) reproduce the emitted vmulss
  // exactly); the .sum() is the reversed tree (SumRevProduct) and
  // error_ -= (sum/N)*norm is the emitted fnmadd. The transpose matvecs
  // are the forward zero-seed pattern (DotForwardFma), N=200 -> M=192,
  // tail 8. update_ += error*input is an elementwise FMA.
  for (unsigned int j = 0; j < num_cells_; ++j) {
    neurons.beta_u_[j] += neurons.error_[j];
    neurons.gamma_u_[j] = __builtin_fmaf(
        neurons.error_[j], neurons.norm_[epoch][j], neurons.gamma_u_[j]);
  }
  {
    const float ivar_ep = neurons.ivar_[epoch];
    // (ivar*gamma)*error — fully dependent, no pairing freedom; plain muls
    // pinned with reassociate(off) (fmaf(x,y,0) would flip -0 to +0).
    {
#pragma clang fp reassociate(off) contract(off)
      for (unsigned int j = 0; j < num_cells_; ++j) {
        const float t = ivar_ep * neurons.gamma_[j];
        neurons.error_[j] = t * neurons.error_[j];
      }
    }
  }
  {
    const float ssum = SumRevProduct(
        &neurons.error_[0], &neurons.norm_[epoch][0], num_cells_);
    const float inv_n = 1.0f / (float)num_cells_;
    for (unsigned int j = 0; j < num_cells_; ++j) {
      const float t = ssum * neurons.norm_[epoch][j];
      neurons.error_[j] = __builtin_fmaf(-t, inv_n, neurons.error_[j]);
    }
  }
  if (layer > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const float f = DotForwardFma(&neurons.error_[0],
          &neurons.transpose_[num_cells_ + i][0], num_cells_);
      (*hidden_error)[i] += f;
    }
  }
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const float f = DotForwardFma(&neurons.error_[0],
          &neurons.transpose_[i][0], num_cells_);
      stored_error_[i] += f;
    }
  }
  std::slice slice = std::slice(output_size_, input.size(), 1);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    const float ei = neurons.error_[i];
    for (unsigned int j = 0; j < input.size(); ++j) {
      neurons.update_[i][output_size_ + j] = __builtin_fmaf(
          ei, input[j], neurons.update_[i][output_size_ + j]);
    }
    neurons.update_[i][input_symbol] += ei;
  }
  if (epoch == 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i],
          &neurons.weights_[i], learning_rate_, update_steps_);
    }
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
        &neurons.gamma_, learning_rate_, update_steps_);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
        &neurons.beta_, learning_rate_, update_steps_);
  }
}

inline std::vector<std::valarray<std::valarray<float>>*> LstmLayer::Weights() {
  std::vector<std::valarray<std::valarray<float>>*> weights;
  weights.push_back(&forget_gate_.weights_);
  weights.push_back(&input_node_.weights_);
  weights.push_back(&output_gate_.weights_);
  return weights;
}

