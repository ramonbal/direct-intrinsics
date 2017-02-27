#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// sxe sigmoid cross entropy
template <typename Dtype>
__global__ void diff_rank_sxe_kernel(
    const int n, const Dtype*sigmoid, const Dtype* target, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[index] = (target[index] >= 0) * (sigmoid[index] - 1) +
          + (target[index] <= 0) * sigmoid[index];
  }
}
template <typename Dtype>
void gpu_diff_rank_sxe(const int n, const Dtype*sigmoid, const Dtype* target,
  Dtype *bottom_diff) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  diff_rank_sxe_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, sigmoid, target, bottom_diff);
}

template <typename Dtype>
void RankSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    const Dtype loss_const =
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    const Dtype ploss = input_data[i] * (1 - (input_data[i] >= 0)) - loss_const;
    const Dtype nloss = input_data[i] * (- (input_data[i] >= 0)) - loss_const;
    // if t=0, both losses are added.
    loss -= (target[i] >= 0) * ploss + (target[i] <= 0) * nloss;
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void RankSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    gpu_diff_rank_sxe<Dtype>(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RankSigmoidCrossEntropyLossLayer);


}  // namespace caffe
