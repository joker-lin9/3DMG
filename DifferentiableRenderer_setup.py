# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mesh_inpaint_processor",
        ["mesh_inpaint_processor.cpp"],
        cxx_std=17,
        define_macros=[("NOMINMAX", None)],
    ),
]

setup(
    name="mesh_inpaint_processor",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
