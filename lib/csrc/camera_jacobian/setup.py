from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='camera_jacobian',
      ext_modules=[
          CUDAExtension(
              'camera_jacobian',
              ['./src/camera_jacobian.cpp', './src/camera_jacobian_kernel.cu'],
              extra_compile_args={
                  'cxx': ['-std=c++14'],
                  'nvcc': ['--use_fast_math']
              })
      ],
      cmdclass={'build_ext': BuildExtension},
      version='1.0.0')
