#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SliceMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SliceMapParameter& slice_map_param = \
      this->layer_param_.slice_map_param();
  y_begin_ = slice_map_param.y_begin();
  y_end_ = slice_map_param.y_end();
  x_begin_ = slice_map_param.x_begin();
  x_end_ = slice_map_param.x_end();
}

template <typename Dtype>
void SliceMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom shape
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  // Check slicing
  y_end_ = y_end_ < 0 ? height + y_end_ + 1: y_end_;
  x_end_ = x_end_ < 0 ? width + x_end_ + 1: x_end_;
  CHECK_GE(y_begin_, 0);
  CHECK_GE(x_begin_, 0);
  CHECK_LT(y_begin_, y_end_);
  CHECK_LT(x_begin_, x_end_);
  CHECK_LE(y_end_, height);
  CHECK_LE(x_end_, width);

  // top shape
  num_ = num;
  channels_ = channels;
  height_ = y_end_ - y_begin_;
  width_ = x_end_ - x_begin_;
  count_ = num_ * channels_ * height_ * width_;

  // Reshape top
  CHECK_EQ(top.size(), 1);
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void SliceMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num_; n++) {
    for (int c = 0; c < channels_; c++) {
      for (int y = 0; y < height_; y++) {
        caffe_copy(
            width_,
            bottom_data + bottom[0]->offset(n, c, y + y_begin_, x_begin_),
            top_data + top[0]->offset(n, c, y));
      }
    }
  }
}

template <typename Dtype>
void SliceMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  for (int n = 0; n < num_; n++) {
    for (int c = 0; c < channels_; c++) {
      for (int y = 0; y < height_; y++) {
        caffe_copy(
            width_,
            top_diff + top[0]->offset(n, c, y),
            bottom_diff + bottom[0]->offset(n, c, y + y_begin_, x_begin_));
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SliceMapLayer);
#endif

INSTANTIATE_CLASS(SliceMapLayer);
REGISTER_LAYER_CLASS(SliceMap);
}  // namespace caffe
