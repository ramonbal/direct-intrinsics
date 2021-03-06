#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <google/protobuf/text_format.h>

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/caffe.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/python_layer.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace bp = boost::python;

namespace caffe {

// For Python, for now, we'll just always use float as the type.
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;

// Selecting mode.
void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
// Checking current mode.
bool check_mode_cpu() { return Caffe::mode() == Caffe::CPU; }
bool check_mode_gpu() { return Caffe::mode() == Caffe::GPU; }
#ifndef CPU_ONLY
// Cuda num threads
int get_cuda_num_threads() { return CAFFE_CUDA_NUM_THREADS; }
bp::object cublas_handle() {
  return bp::object((size_t)Caffe::cublas_handle());
}
#endif

// For convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases).
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

void CheckContiguousArray(PyArrayObject* arr, string name,
    int channels, int height, int width) {
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error(name + " must be C contiguous");
  }
  if (PyArray_NDIM(arr) != 4) {
    throw std::runtime_error(name + " must be 4-d");
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error(name + " must be float32");
  }
  if (PyArray_DIMS(arr)[1] != channels) {
    throw std::runtime_error(name + " has wrong number of channels");
  }
  if (PyArray_DIMS(arr)[2] != height) {
    throw std::runtime_error(name + " has wrong height");
  }
  if (PyArray_DIMS(arr)[3] != width) {
    throw std::runtime_error(name + " has wrong width");
  }
}

// Net constructor for passing phase as int
shared_ptr<Net<Dtype> > Net_Init(
    string param_file, int phase) {
  CheckFile(param_file);

  shared_ptr<Net<Dtype> > net(new Net<Dtype>(param_file,
      static_cast<Phase>(phase)));
  return net;
}

// Net construct-and-load convenience constructor
shared_ptr<Net<Dtype> > Net_Init_Load(
    string param_file, string pretrained_param_file, int phase) {
  CheckFile(param_file);
  CheckFile(pretrained_param_file);

  shared_ptr<Net<Dtype> > net(new Net<Dtype>(param_file,
      static_cast<Phase>(phase)));
  net->CopyTrainedLayersFrom(pretrained_param_file);
  return net;
}

void Net_Save(const Net<Dtype>& net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, filename.c_str());
}

void Net_SetInputArrays(Net<Dtype>* net, bp::object data_obj,
    bp::object labels_obj) {
  // check that this network has an input MemoryDataLayer
  shared_ptr<MemoryDataLayer<Dtype> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<Dtype> >(net->layers()[0]);
  if (!md_layer) {
    throw std::runtime_error("set_input_arrays may only be called if the"
        " first layer is a MemoryDataLayer");
  }

  // check that we were passed appropriately-sized contiguous memory
  PyArrayObject* data_arr =
      reinterpret_cast<PyArrayObject*>(data_obj.ptr());
  PyArrayObject* labels_arr =
      reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
  CheckContiguousArray(data_arr, "data array", md_layer->channels(),
      md_layer->height(), md_layer->width());
  CheckContiguousArray(labels_arr, "labels array", 1, 1, 1);
  if (PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
    throw std::runtime_error("data and labels must have the same first"
        " dimension");
  }
  if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
    throw std::runtime_error("first dimensions of input arrays must be a"
        " multiple of batch size");
  }

  md_layer->Reset(static_cast<Dtype*>(PyArray_DATA(data_arr)),
      static_cast<Dtype*>(PyArray_DATA(labels_arr)),
      PyArray_DIMS(data_arr)[0]);
}

// Scoped GIL releasing
class ReleaseGIL {
 public:
  ReleaseGIL() {
    state_ = PyEval_SaveThread();
  }
  ~ReleaseGIL() {
    PyEval_RestoreThread(state_);
  }
 private:
  PyThreadState* state_;
};

// Run forward prop without GIL
Dtype Net_ForwardFromToNoGIL(Net<Dtype>* net, int start, int end) {
  ReleaseGIL gil;
  return net->ForwardFromTo(start, end);
}

// Run backward prop without GIL
void Net_BackwardFromToNoGIL(Net<Dtype>* net, int start, int end) {
  ReleaseGIL gil;
  net->BackwardFromTo(start, end);
}

Solver<Dtype>* GetSolverFromFile(const string& filename) {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(filename, &param);
  return GetSolver<Dtype>(param);
}

struct NdarrayConverterGenerator {
  template <typename T> struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<Dtype*> {
  struct type {
    PyObject* operator() (Dtype* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
    }
    const PyTypeObject* get_pytype() {
      return &PyArray_Type;
    }
  };
};

struct NdarrayCallPolicies : public bp::default_call_policies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    bp::object pyblob = bp::extract<bp::tuple>(pyargs)()[0];
    shared_ptr<Blob<Dtype> > blob =
      bp::extract<shared_ptr<Blob<Dtype> > >(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->num_axes();
    vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
    PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(),
                                                  NPY_FLOAT32, data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj),
        pyblob.ptr());
    return arr_obj;
  }
};

// Blob constructor with shape iterable
shared_ptr<Blob<Dtype> > Blob_Init(bp::object shape_object) {
  size_t ndim;
  try {
    ndim = bp::len(shape_object);
  } catch(...) {
    throw std::runtime_error("1st arg must be iterable.");
  }
  vector<int> shape(ndim);
  try {
    for (int i = 0; i < ndim; ++i) {
      shape[i] = bp::extract<int>(shape_object[i]);
    }
  } catch(...) {
    throw std::runtime_error("All element in shape iterable must be integer.");
  }
  return shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape));
}

bp::tuple Blob_Shape(const Blob<Dtype>* self) {
  const vector<int> &shape = self->shape();
  bp::list shape_list;
  BOOST_FOREACH(int s, shape) {
    shape_list.append(s);
  }
  return bp::tuple(shape_list);
}

bp::object Blob_Reshape(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }
  Blob<Dtype>* self = bp::extract<Blob<Dtype>*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->Reshape(shape);
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

// Layer
template <class T>
vector<T> py_to_vector(bp::object pyiter) {
  vector<T> vec;
  for (int i = 0; i < bp::len(pyiter); ++i) {
    vec.push_back(bp::extract<T>(pyiter[i]));
  }
  return vec;
}
void Layer_SetUp(Layer<Dtype> *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob<Dtype>*> bottom = py_to_vector<Blob<Dtype>*>(py_bottom);
  vector<Blob<Dtype>*> top = py_to_vector<Blob<Dtype>*>(py_top);
  {
    ReleaseGIL gil;
    layer->SetUp(bottom, top);
  }
}
void Layer_Reshape(
    Layer<Dtype> *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob<Dtype>*> bottom = py_to_vector<Blob<Dtype>*>(py_bottom);
  vector<Blob<Dtype>*> top = py_to_vector<Blob<Dtype>*>(py_top);
  {
    ReleaseGIL gil;
    layer->Reshape(bottom, top);
  }
}
Dtype Layer_Forward(
    Layer<Dtype> *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob<Dtype>*> bottom = py_to_vector<Blob<Dtype>*>(py_bottom);
  vector<Blob<Dtype>*> top = py_to_vector<Blob<Dtype>*>(py_top);
  Dtype loss;
  {
    ReleaseGIL gil;
    loss = layer->Forward(bottom, top);
  }
  return loss;
}
void Layer_Backward(
    Layer<Dtype> *layer, bp::object py_top, bp::object py_propagate_down,
    bp::object py_bottom) {
  vector<Blob<Dtype>*> top = py_to_vector<Blob<Dtype>*>(py_top);
  vector<bool> propagate_down = py_to_vector<bool>(py_propagate_down);
  vector<Blob<Dtype>*> bottom = py_to_vector<Blob<Dtype>*>(py_bottom);
  {
    ReleaseGIL gil;
    layer->Backward(top, propagate_down, bottom);
  }
}

// LayerParameter
shared_ptr<LayerParameter> LayerParameter_Init(bp::object py_layer_param) {
  shared_ptr<LayerParameter> layer_param(new LayerParameter);
  if (PyObject_HasAttrString(py_layer_param.ptr(), "SerializeToString")) {
    string dump = bp::extract<string>(
        py_layer_param.attr("SerializeToString")());
    layer_param->ParseFromString(dump);
  } else {
    try {
      string dump = bp::extract<string>(py_layer_param);
      google::protobuf::TextFormat::ParseFromString(dump, layer_param.get());
    } catch(...) {
      throw std::runtime_error("1st arg must be LayerPrameter or string.");
    }
  }
  if (!layer_param->IsInitialized()) {
    throw std::runtime_error(
      "LayerParameter not initialized: Missing required fields.");
  }
  return layer_param;
}
void LayerParameter_FromPython(
    LayerParameter *layer_param, bp::object py_layer_param) {
  shared_ptr<LayerParameter> copy = \
      LayerParameter_Init(py_layer_param);
  layer_param->Clear();
  layer_param->CopyFrom(*copy);
}
bp::object LayerParameter_ToPython(
    const LayerParameter *layer_param, bp::object py_layer_param) {
  string dump;
  layer_param->SerializeToString(&dump);
  py_layer_param.attr("ParseFromString")(bp::object(dump));
  return py_layer_param;
}

// Create layer from caffe_pb2.LayerParameter in Python
shared_ptr<Layer<Dtype> > create_layer(bp::object py_layer_param) {
  shared_ptr<LayerParameter> layer_param(LayerParameter_Init(py_layer_param));
  return LayerRegistry<Dtype>::CreateLayer(*layer_param.get());
}

#ifndef CPU_ONLY
size_t Blob_GpuDataPtr(Blob<Dtype>* self) {
  return (size_t)(self->mutable_gpu_data());
}

size_t Blob_GpuDiffPtr(Blob<Dtype>* self) {
  return (size_t)(self->mutable_gpu_diff());
}
#endif

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolveOverloads, Solve, 0, 1);

BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python
  // Caffe utility functions
  bp::def("set_mode_cpu", &set_mode_cpu);
  bp::def("set_mode_gpu", &set_mode_gpu);
  bp::def("check_mode_cpu", &check_mode_cpu);
  bp::def("check_mode_gpu", &check_mode_gpu);
  bp::def("set_device", &Caffe::SetDevice);
  bp::def("get_device", &Caffe::GetDevice);
  bp::def("set_random_seed", &Caffe::set_random_seed);
#ifndef CPU_ONLY
  bp::def("get_cuda_num_threads", &get_cuda_num_threads);
  bp::def("get_blocks", &CAFFE_GET_BLOCKS);
  bp::def("cublas_handle", &cublas_handle);
#endif

  bp::class_<Net<Dtype>, shared_ptr<Net<Dtype> >, boost::noncopyable >("Net",
    bp::no_init)
    .def("__init__", bp::make_constructor(&Net_Init))
    .def("__init__", bp::make_constructor(&Net_Init_Load))
    .def("_forward", &Net_ForwardFromToNoGIL)
    .def("_backward", &Net_BackwardFromToNoGIL)
    .def("reshape", &Net<Dtype>::Reshape)
    // The cast is to select a particular overload.
    .def("copy_from", static_cast<void (Net<Dtype>::*)(const string)>(
        &Net<Dtype>::CopyTrainedLayersFrom))
    .def("share_with", &Net<Dtype>::ShareTrainedLayersWith)
    .add_property("_blobs", bp::make_function(&Net<Dtype>::blobs,
        bp::return_internal_reference<>()))
    .add_property("layers", bp::make_function(&Net<Dtype>::layers,
        bp::return_internal_reference<>()))
    .add_property("_blob_names", bp::make_function(&Net<Dtype>::blob_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_layer_names", bp::make_function(&Net<Dtype>::layer_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_inputs", bp::make_function(&Net<Dtype>::input_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_outputs",
        bp::make_function(&Net<Dtype>::output_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_set_input_arrays", &Net_SetInputArrays,
        bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >())
    .def("save", &Net_Save);

  bp::class_<Blob<Dtype>, shared_ptr<Blob<Dtype> >, boost::noncopyable>(
      "Blob", bp::no_init)
    .def("__init__", bp::make_constructor(&Blob_Init))
    .add_property("num",      &Blob<Dtype>::num)
    .add_property("channels", &Blob<Dtype>::channels)
    .add_property("height",   &Blob<Dtype>::height)
    .add_property("width",    &Blob<Dtype>::width)
    .add_property("count",    static_cast<int (Blob<Dtype>::*)() const>(
        &Blob<Dtype>::count))
    .add_property("shape", &Blob_Shape)
    .def("reshape",           bp::raw_function(&Blob_Reshape))
#ifndef CPU_ONLY
    .add_property("gpu_data_ptr", &Blob_GpuDataPtr)
    .add_property("gpu_diff_ptr", &Blob_GpuDiffPtr)
#endif
    .add_property("data",     bp::make_function(&Blob<Dtype>::mutable_cpu_data,
          NdarrayCallPolicies()))
    .add_property("diff",     bp::make_function(&Blob<Dtype>::mutable_cpu_diff,
          NdarrayCallPolicies()));

  bp::class_<Layer<Dtype>, shared_ptr<PythonLayer<Dtype> >,
      boost::noncopyable>(
      "Layer", bp::init<const LayerParameter&>())
    .add_property("blobs", bp::make_function(&Layer<Dtype>::blobs,
          bp::return_internal_reference<>()))
    .def("setup", &Layer<Dtype>::LayerSetUp)
    .def("SetUp", &Layer_SetUp)
    .def("reshape", &Layer<Dtype>::Reshape)
    .def("Reshape", &Layer_Reshape)
    .def("Forward", &Layer_Forward)
    .def("Backward", &Layer_Backward)
    .add_property("type", bp::make_function(&Layer<Dtype>::type));
  bp::register_ptr_to_python<shared_ptr<Layer<Dtype> > >();

  bp::class_<LayerParameter, shared_ptr<LayerParameter> >(
      "LayerParameter", bp::no_init)
    .def("__init__", bp::make_constructor(&LayerParameter_Init))
    .def("from_python", &LayerParameter_FromPython)
    .def("_to_python", &LayerParameter_ToPython);

  bp::def("create_layer", &create_layer);

  bp::class_<Solver<Dtype>, shared_ptr<Solver<Dtype> >, boost::noncopyable>(
    "Solver", bp::no_init)
    .add_property("net", &Solver<Dtype>::net)
    .add_property("test_nets", bp::make_function(&Solver<Dtype>::test_nets,
          bp::return_internal_reference<>()))
    .add_property("iter", &Solver<Dtype>::iter)
    .def("solve", static_cast<void (Solver<Dtype>::*)(const char*)>(
          &Solver<Dtype>::Solve), SolveOverloads())
    .def("step", &Solver<Dtype>::Step)
    .def("restore", &Solver<Dtype>::Restore);

  bp::class_<SGDSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<SGDSolver<Dtype> >, boost::noncopyable>(
        "SGDSolver", bp::init<string>())
    .add_property("learning_rate",
          &SGDSolver<Dtype>::GetLearningRate,
          &SGDSolver<Dtype>::SetLearningRate);

  bp::class_<NesterovSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<NesterovSolver<Dtype> >, boost::noncopyable>(
        "NesterovSolver", bp::init<string>());
  bp::class_<AdaGradSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<AdaGradSolver<Dtype> >, boost::noncopyable>(
        "AdaGradSolver", bp::init<string>());

  bp::def("get_solver", &GetSolverFromFile,
      bp::return_value_policy<bp::manage_new_object>());

  // vector wrappers for all the vector types we use
  bp::class_<vector<shared_ptr<Blob<Dtype> > > >("BlobVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Blob<Dtype> > >, true>());
  bp::class_<vector<Blob<Dtype>*> >("RawBlobVec")
    .def(bp::vector_indexing_suite<vector<Blob<Dtype>*>, true>());
  bp::class_<vector<shared_ptr<Layer<Dtype> > > >("LayerVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Layer<Dtype> > >, true>());
  bp::class_<vector<string> >("StringVec")
    .def(bp::vector_indexing_suite<vector<string> >());
  bp::class_<vector<int> >("IntVec")
    .def(bp::vector_indexing_suite<vector<int> >());
  bp::class_<vector<shared_ptr<Net<Dtype> > > >("NetVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Net<Dtype> > >, true>());
  bp::class_<vector<bool> >("BoolVec")
    .def(bp::vector_indexing_suite<vector<bool> >());
  bp::class_<vector<float> >("FloatVec")
      .def(bp::vector_indexing_suite<vector<float> >());

  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}

}  // namespace caffe
