import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = '4D Post-stack seismic inversion with Proximal solvers'

setup(
    name="prox4d", # Choose your package name
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'time-lapse',
              'deep learning',
              'seismic'],
    author='Juan Romero, Nick Luiken, Matteo Ravasi',
    author_email='juan.romeromurcia@kaust.edu.sa, nicolaas.luiken@kaust.edu.sa, matteo.ravasi@kaust.edu.sa',
    packages=find_packages(),
)
