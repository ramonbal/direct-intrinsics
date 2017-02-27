#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void slice_map_forward_gpu_kernel(
    const Dtype* bottom_data, const int height_b, const int width_b,
    const int y_begin, const int x_begin,
    const int channels, const int height, const int width,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, channels * height * width) {
    int x = index % width;
    int y_ind = index / width;
    int y = y_ind % height;
    int c = y_ind / height;
    top_data[index] = bottom_data[
        (c * height_b + (y + y_begin)) * width_b + x + x_begin];
  }
}

template <typename Dtype>
void slice_map_forward_gpu(
    const Dtype* bottom_data, const int height_b, const int width_b,
    const int y_begin, const int x_begin,
    const int channels, const int height, const int width,
    Dtype* top_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  slice_map_forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(
    channels * height * width), CAFFE_CUDA_NUM_THREADS>>>(
    bottom_data, height_b, width_b,
    y_begin, x_begin,
    channels, height, width,
    top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void slice_map_backward_gpu_kernel(
    const Dtype* top_diff,
    const int channels, const int height, const int width,
    const int height_b, const int width_b,
    const int y_begin, const int x_begin,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, channels * height * width) {
    int x = index % width;
    int y_ind = index / width;
    int y = y_ind % height;
    int c = y_ind / height;
    bottom_diff[(c * height_b + (y + y_begin)) * width_b + x + x_begin] = \
        top_diff[index];
  }
}

template <typename Dtype>
void slice_map_backward_gpu(
    const Dtype* top_diff,
    const int channels, const int height, const int width,
    const int height_b, const int width_b,
    const int y_begin, const int x_begin,
    Dtype* bottom_diff) {
  caffe_gpu_set(channels  * height_b * width_b, Dtype(0), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  slice_map_backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(
    channels * height * width), CAFFE_CUDA_NUM_THREADS>>>(
    top_diff,
    channels, height, width,
    height_b, width_b,
    y_begin, x_begin,
    bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void SliceMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < num_; n++) {
    slice_map_forward_gpu(
        bottom_data + bottom[0]->offset(n),
        bottom[0]->height(), bottom[0]->width(),
        y_begin_, x_begin_,
        channels_, height_, width_,
        top_data + top[0]->offset(n));
  }
}

template <typename Dtype>
void SliceMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int n = 0; n < num_; n++) {
    slice_map_backward_gpu(
        top_diff + top[0]->offset(n),
        channels_, height_, width_,
        bottom[0]->height(), bottom[0]->width(),
        y_begin_, x_begin_,
        bottom_diff + bottom[0]->offset(n));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SliceMapLayer);

}  // namespace caffe
