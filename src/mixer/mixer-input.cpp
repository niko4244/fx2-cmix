#include "mixer-input.h"

MixerInput::MixerInput(const Sigmoid& sigmoid, float eps) :
    inputs_(0.5, 1), sigmoid_(sigmoid), min_(eps), max_(1 - eps),
    stretched_min_(sigmoid.Logit(0)), stretched_max_(sigmoid.Logit(1)) {}

void MixerInput::SetNumModels(int num_models) {
  inputs_.resize(num_models, 0.5);
}

