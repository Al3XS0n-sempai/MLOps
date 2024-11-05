from setuptools import setup, find_packages
from glob import glob

so_files = glob("relu_module/module/relu_binding*.so")

setup(
    name="relu_module",
    version="0.1",
    description="My ReLU with Python bindings",
    packages=find_packages(),
    package_data={
        "relu_module": ["module/*.so"],
    },
)
