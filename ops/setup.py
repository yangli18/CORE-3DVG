from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": []}
define_macros = []

define_macros += [("WITH_CUDA", None)]
extra_compile_args["nvcc"] = [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]
sources = [
    'csrc/vision.cpp', 
    'csrc/cuda/ml_nms.cu'
]

setup(
    name='ops',
    ext_modules=[
        CUDAExtension(
            '_C', 
            sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)