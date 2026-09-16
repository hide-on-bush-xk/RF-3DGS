#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    # MSVC 14.4x+ (VS 2022 17.10 and VS 2026) ships a C++17-only STL, and CCCL
    # (CUDA 12.9+) refuses to build under the traditional preprocessor.
    nvcc_compiler_flags += ["-std=c++17", "-Xcompiler", "/Zc:preprocessor"]

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
