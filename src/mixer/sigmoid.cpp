#include "sigmoid.h"

#include <math.h>

Sigmoid::Sigmoid(int logit_size) : logit_size_(logit_size),
    logit_table_(logit_size, 0) {
  for (int i = 0; i < logit_size_; ++i) {
    logit_table_[i] = SlowLogit((i + 0.5f) / logit_size_);
  }
}

float Sigmoid::Logistic(float p) {
  return 1 / (1 + exp(-p));
}

float Sigmoid::FastLogistic(float p) {
  return (0.5f * (p / (1.0f + abs(p)) + 1.0f));
}

float Sigmoid::SlowLogit(float p) {
  return log(p / (1 - p));
}
