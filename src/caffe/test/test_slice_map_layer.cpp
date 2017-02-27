#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SliceMapLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SliceMapLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 8, 15, 20)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual void ReduceBottomBlobSize() {
    blob_bottom_->Reshape(2, 2, 4, 5);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }
  virtual ~SliceMapLayerTest() {
    delete blob_top_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(SliceMapLayerTest, TestDtypesAndDevices);

TYPED_TEST(SliceMapLayerTest, TestSetup1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceMapParameter* slice_map_param = layer_param.mutable_slice_map_param();
  slice_map_param->set_y_begin(0);
  slice_map_param->set_y_end(-1);
  slice_map_param->set_x_begin(0);
  slice_map_param->set_x_end(-1);
  SliceMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_->width());
}

TYPED_TEST(SliceMapLayerTest, TestSetup2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceMapParameter* slice_map_param = layer_param.mutable_slice_map_param();
  slice_map_param->set_y_begin(0);
  slice_map_param->set_y_end(-2);
  slice_map_param->set_x_begin(0);
  slice_map_param->set_x_end(-2);
  SliceMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_->channels());
  EXPECT_EQ(this->blob_bottom_->height()-1, this->blob_top_->height());
  EXPECT_EQ(this->blob_bottom_->width()-1, this->blob_top_->width());
}

TYPED_TEST(SliceMapLayerTest, TestSetup3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceMapParameter* slice_map_param = layer_param.mutable_slice_map_param();
  slice_map_param->set_y_begin(3);
  slice_map_param->set_y_end(5);
  slice_map_param->set_x_begin(6);
  slice_map_param->set_x_end(12);
  SliceMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_->channels());
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 6);
}

TYPED_TEST(SliceMapLayerTest, TestSlice) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceMapParameter* slice_map_param = layer_param.mutable_slice_map_param();
  slice_map_param->set_y_begin(3);
  slice_map_param->set_y_end(5);
  slice_map_param->set_x_begin(6);
  slice_map_param->set_x_end(12);
  SliceMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(
              this->blob_bottom_->data_at(
                  n, c,
                  h + slice_map_param->y_begin(),
                  w + slice_map_param->x_begin()),
              this->blob_top_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceMapLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  SliceMapParameter* slice_map_param = layer_param.mutable_slice_map_param();
  slice_map_param->set_y_begin(1);
  slice_map_param->set_y_end(2);
  slice_map_param->set_x_begin(2);
  slice_map_param->set_x_end(-1);
  SliceMapLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_);
}

}  // namespace caffe
