from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def build_extensions(self):
        # 延迟到 build_extensions 时才导入 torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
        import torch

        extra_compile_args = {
            'cxx': ['/std:c++17']  # Windows 上的 C++17 标志
        }

        self.extensions = [
            CUDAExtension(
                "custom_rasterizer_kernel_for_windows",
                [
                    "lib/custom_rasterizer_kernel_for_windows/rasterizer.cpp",
                    "lib/custom_rasterizer_kernel_for_windows/grid_neighbor.cpp",
                    "lib/custom_rasterizer_kernel_for_windows/rasterizer_gpu.cu",
                ],
                extra_compile_args=extra_compile_args
            )
        ]
        super().build_extensions()


setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[],  # 留空，由 CustomBuildExt 动态填充
    cmdclass={"build_ext": CustomBuildExt},
)