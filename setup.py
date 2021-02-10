from setuptools import setup, find_packages

setup(
    name="visplot",
    install_requires=[
        "vispy",
        "numpy",
    ],
    packages=find_packages(),
    py_modules=["visplot"],
    version=1.0,
    author="yhql",
    description="Side-channel traces visualizer",
)