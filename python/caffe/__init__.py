from .pycaffe import Net, SGDSolver, LayerParameter
from ._caffe import (
    set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver,
    get_device,
    check_mode_cpu, check_mode_gpu,
    set_random_seed,
    Blob,
    create_layer)
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
import io
try:
	from ._caffe import get_cuda_num_threads, get_blocks, cublas_handle
except ImportError:
	pass
