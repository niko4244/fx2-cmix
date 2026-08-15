#include "lstm.h"

#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <iostream>

inline Lstm::Lstm(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip) : input_history_(horizon),
    hidden_(num_cells * num_layers + 1), hidden_error_(num_cells),
    layer_input_(std::valarray<std::valarray<float>>(std::valarray<float>
    (input_size + 1 + num_cells * 2), num_layers), horizon),
    output_layer_(std::valarray<std::valarray<float>>(std::valarray<float>
   (num_cells * num_layers + 1), output_size), horizon),
    output_(std::valarray<float>(1.0 / output_size, output_size), horizon),
    learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
    horizon_(horizon), input_size_(input_size), output_size_(output_size) {
  hidden_[hidden_.size() - 1] = 1;
  for (int epoch = 0; epoch < horizon; ++epoch) {
    layer_input_[epoch][0].resize(1 + num_cells + input_size);
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
  }
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.emplace_back(layer_input_[0][i].size() + output_size, input_size_, output_size_,
        num_cells, horizon, gradient_clip, learning_rate);
  }
}

inline Lstm::~Lstm() {
  //SaveToDisk("lstm.dat");
}
/*
inline void Lstm::SaveToDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ofstream os(path, std::ios::binary | std::ios::out);
  if (!os.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    os.write(reinterpret_cast<const char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i].Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        os.write(reinterpret_cast<const char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  os.close();
}

inline void Lstm::LoadFromDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ifstream is(path, std::ios::binary | std::ios::in);
  if (!is.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    is.read(reinterpret_cast<char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i].Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        is.read(reinterpret_cast<char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  is.close();
}
*/
inline void Lstm::SetInput(const std::valarray<float>& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::copy(begin(input), begin(input) + input_size_,
        begin(layer_input_[epoch_][i]));
  }
}

inline std::valarray<float>& Lstm::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        for (unsigned int i = 0; i < output_size_; ++i) {
//          float error = 0;
//          if (i == input_history_[epoch]) error = output_[epoch][i] - 1;
//          else error = output_[epoch][i];
          float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
          for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
            hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
          }
        }
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer,
            input_symbol, &hidden_error_);
      }
    }
  }

  for (unsigned int i = 0; i < output_size_; ++i) {
//    float error = 0;
//    if (i == input) error = output_[last_epoch][i] - 1;
//    else error = output_[last_epoch][i];
    float error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
    output_layer_[epoch_][i] = output_layer_[last_epoch][i];
    output_layer_[epoch_][i] -= learning_rate_ * error * hidden_;
  }
  return Predict(input);
}

inline std::valarray<float>& Lstm::Predict(unsigned int input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_, begin(layer_input_[epoch_][i]) +
        input_size_);
    layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
        num_cells_);
    if (i < layers_.size() - 1) {
      auto start2 = begin(layer_input_[epoch_][i + 1]) + num_cells_ +
          input_size_;
      std::copy(start, start + num_cells_, start2);
    }
  }
  // Output projection reconstructed BY CONSTRUCTION, matching clang-17's
  // codegen for the original loop (see tools/redtest.cpp and the disasm
  // job): 4 x 8-lane FMA accumulators over 32-element blocks (all-zero
  // seed), the reduce (y1+y0)+(y3+y2), then a forward scalar FMA tail,
  // then expf. The reduce's partial sums are pinned through a volatile
  // round-trip (fast-math re-pairs the tree otherwise — the pairing lives
  // in register allocation, invisible to instruction-level diffs); the
  // tail runs under reassociate(off). expf stays a libm call (compiler-
  // independent).
  {
    const int N = (int)hidden_.size();
    const int M = (N / 32) * 32;
    const float* h = N > 0 ? &hidden_[0] : nullptr;
    for (unsigned int i = 0; i < output_size_; ++i) {
      const float* w = &output_layer_[epoch_][i][0];
      __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps();
      __m256 y2 = _mm256_setzero_ps(), y3 = _mm256_setzero_ps();
      for (int b = 0; b < M; b += 32) {
        y0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b),
                             _mm256_loadu_ps(h + b), y0);
        y1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b + 8),
                             _mm256_loadu_ps(h + b + 8), y1);
        y2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b + 16),
                             _mm256_loadu_ps(h + b + 16), y2);
        y3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + b + 24),
                             _mm256_loadu_ps(h + b + 24), y3);
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
      for (int j = M; j < N; ++j)
        sum = __builtin_fmaf(w[j], h[j], sum);
      output_[epoch_][i] = expf(sum);
    }
  }
  output_[epoch_] /= output_[epoch_].sum();
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  last_input_ = input;
  return output_[epoch];
}

