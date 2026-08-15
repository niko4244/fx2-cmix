#include "lstm-layer.h"

#include "sigmoid.h"

#include <math.h>
#include <algorithm>
#include <numeric>

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

// The scalar-loop rewrites below are byte-identical to the original valarray
// expressions ONLY with FP contraction disabled: the valarray operators
// rounded every intermediate into a heap temporary (two roundings per fused
// op), whereas a merged scalar expression would let clang contract a*b+c
// into one FMA (one rounding). #pragma clang fp contract(off) reproduces the
// valarray's per-operation rounding exactly. GCC ignores the pragma.

// Scalar-loop Adam: identical per-element arithmetic to the valarray version
// (same operations, same order), but allocates no temporaries. Called once per
// cell per backprop epoch, so the valarray temp churn here was measurable.
inline void Adam(std::valarray<float>* g, std::valarray<float>* m,
    std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
    float t) {
#pragma clang fp contract(off)
  const float beta1 = 0.025f, beta2 = 0.9999f, eps = 1e-6f;
  float alpha, denom1, denom2;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    denom1 = 1.0f - pow(beta1, t);
    denom2 = 1.0f - pow(beta2, t);
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
    denom1 = 1.0f - pow(beta1, (float)UPDATE_LIMIT);
    denom2 = 1.0f - pow(beta2, (float)UPDATE_LIMIT);
  }
  const float b1 = 1.0f - beta1, b2 = 1.0f - beta2;
  for (size_t i = 0; i < g->size(); ++i) {
    (*m)[i] *= beta1;
    (*m)[i] += b1 * (*g)[i];
    (*v)[i] *= beta2;
    (*v)[i] += b2 * (*g)[i] * (*g)[i];
    (*w)[i] -= alpha * (((*m)[i] / denom1) /
        (sqrt((*v)[i] / denom2 + eps)));
  }
}

// LayerNorm update for one forward pass. Was `(norm*norm).sum()`, `norm*=
// ivar`, `state = norm*gamma + beta` as valarray expressions (each op rounded
// into a heap temp); contraction-off reproduces that rounding exactly.
static inline void lstm_forward_normalize(NeuronLayer& neurons, int epoch,
    unsigned int num_cells) {
#pragma clang fp contract(off)
  float sum = 0;
  for (unsigned int i = 0; i < num_cells; ++i) {
    float n = neurons.norm_[epoch][i];
    sum += n * n;
  }
  neurons.ivar_[epoch] = 1.0f / sqrt((sum / num_cells) + 1e-5f);
  for (unsigned int i = 0; i < num_cells; ++i) {
    neurons.norm_[epoch][i] *= neurons.ivar_[epoch];
  }
  for (unsigned int i = 0; i < num_cells; ++i) {
    neurons.state_[epoch][i] = neurons.norm_[epoch][i] * neurons.gamma_[i] +
        neurons.beta_[i];
  }
}

// Per-cell gradient updates for one backward pass: the error_/gamma_u_
// adjustments and the weight-gradient accumulation. Was valarray expressions
// (per-op heap temps); contraction-off reproduces that rounding exactly.
// `input_symbol` and the input slice update are split into a separate helper
// (lstm_backward_weight_grads) so the untouched transpose reductions in
// BackwardPass keep their original FMA behavior.
static inline void lstm_backward_errors(NeuronLayer& neurons, int epoch,
    unsigned int num_cells) {
#pragma clang fp contract(off)
  for (unsigned int i = 0; i < num_cells; ++i) {
    neurons.gamma_u_[i] += neurons.error_[i] * neurons.norm_[epoch][i];
  }
  for (unsigned int i = 0; i < num_cells; ++i) {
    neurons.error_[i] *= neurons.gamma_[i] * neurons.ivar_[epoch];
  }
  float sum = 0;
  for (unsigned int i = 0; i < num_cells; ++i) {
    sum += neurons.error_[i] * neurons.norm_[epoch][i];
  }
  const float mean = sum / num_cells;
  for (unsigned int i = 0; i < num_cells; ++i) {
    neurons.error_[i] -= mean * neurons.norm_[epoch][i];
  }
}

static inline void lstm_backward_weight_grads(NeuronLayer& neurons,
    const std::valarray<float>& input, unsigned int num_cells,
    unsigned int output_size, int input_symbol) {
#pragma clang fp contract(off)
  const unsigned int input_size = input.size();
  for (unsigned int i = 0; i < num_cells; ++i) {
    const float e = neurons.error_[i];
    for (unsigned int j = 0; j < input_size; ++j) {
      neurons.update_[i][output_size + j] += e * input[j];
    }
    neurons.update_[i][input_symbol] += e;
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
#pragma clang fp contract(off)
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
  // Scalar loops replace valarray temporaries; arithmetic is element-identical.
  for (unsigned int i = 0; i < num_cells_; ++i) {
    input_gate_state_[epoch_][i] = 1.0f - forget_gate_.state_[epoch_][i];
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    state_[i] *= forget_gate_.state_[epoch_][i];
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    state_[i] += input_node_.state_[epoch_][i] * input_gate_state_[epoch_][i];
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    tanh_state_[epoch_][i] = FAST_TANH(state_[i]);
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    (*hidden)[hidden_start + i] =
        output_gate_.state_[epoch_][i] * tanh_state_[epoch_][i];
  }
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void LstmLayer::ForwardPass(NeuronLayer& neurons,
    const std::valarray<float>& input, int input_symbol) {
  for (unsigned int i = 0; i < num_cells_; ++i) {
    float f = neurons.weights_[i][input_symbol];
    for (unsigned int j = 0; j < input.size(); ++j) {
      f += input[j] * neurons.weights_[i][output_size_ + j];
    }
    neurons.norm_[epoch_][i] = f;
  }
  // Scalar loop with identical rounding to the original valarray expressions
  // (see lstm_forward_normalize: contraction-off, no heap temporaries). The
  // matvec loop above keeps its original FMA contraction.
  lstm_forward_normalize(neurons, epoch_, num_cells_);
}

inline void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
  }
}

inline void LstmLayer::BackwardPass(const std::valarray<float>&input, int epoch,
    int layer, int input_symbol, std::valarray<float>* hidden_error) {
#pragma clang fp contract(off)
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
  } else {
    stored_error_ += *hidden_error;
  }

  // Scalar loops replace valarray temporaries; per-element operations and
  // their order are identical to the valarray expressions.
  for (unsigned int i = 0; i < num_cells_; ++i) {
    output_gate_.error_[i] = tanh_state_[epoch][i] * stored_error_[i] *
        output_gate_.state_[epoch][i] *
        (1.0f - output_gate_.state_[epoch][i]);
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    state_error_[i] += stored_error_[i] * output_gate_.state_[epoch][i] *
        (1.0f - (tanh_state_[epoch][i] * tanh_state_[epoch][i]));
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    input_node_.error_[i] = state_error_[i] * input_gate_state_[epoch][i] *
        (1.0f - (input_node_.state_[epoch][i] * input_node_.state_[epoch][i]));
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    forget_gate_.error_[i] = (last_state_[epoch][i] - input_node_.state_[epoch][i]) *
        state_error_[i] * forget_gate_.state_[epoch][i] *
        input_gate_state_[epoch][i];
  }

  for (unsigned int i = 0; i < num_cells_; ++i) {
    (*hidden_error)[i] = 0;
  }
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      state_error_[i] *= forget_gate_.state_[epoch][i];
    }
    for (unsigned int i = 0; i < num_cells_; ++i) {
      stored_error_[i] = 0;
    }
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
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.gamma_u_[i] = 0;
      neurons.beta_u_[i] = 0;
    }
    for (unsigned int i = 0; i < num_cells_; ++i) {
      for (unsigned int j = 0; j < neurons.update_[i].size(); ++j) {
        neurons.update_[i][j] = 0;
      }
      int offset = output_size_ + input_size_;
      for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
        neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
      }
    }
  }
  // Pure adds (beta_u_ += error_) can't contract, so it stays inline. The
  // error_/gamma_u_ updates move to lstm_backward_errors (contraction-off to
  // match the original valarray rounding); the transpose reductions below
  // keep their original FMA behavior.
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.beta_u_[i] += neurons.error_[i];
  }
  lstm_backward_errors(neurons, epoch, num_cells_);
  if (layer > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += neurons.error_[j] * neurons.transpose_[num_cells_ + i][j];
      }
      (*hidden_error)[i] += f;
    }
  }
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += neurons.error_[j] * neurons.transpose_[i][j];
      }
      stored_error_[i] += f;
    }
  }
  lstm_backward_weight_grads(neurons, input, num_cells_, output_size_,
      input_symbol);
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

