![](https://github.com/DIG-Kaust/4DProximal/blob/main/4DJIS_gif.gif)

# 4DProximal

Joint inversion-segmentation of 4D post-stack seismic data using proximal solvers as introduced in the paper:

> **[Seeing through the CO2 plume: joint inversion-segmentation of the Sleipner 4D Seismic Dataset](https://arxiv.org/abs/2303.11662)** \
> [Juan Romero](https://www.juanromerom.com/) <sup>1</sup>, [Nick Luiken](https://github.com/NickLuiken) <sup>1</sup>
> , [Matteo Ravasi](https://mrava87.github.io/) <sup>1</sup>\
> <sup>1</sup> King Abdullah University of Science and Technology (KAUST)

This repository contains the 4D joint inversion-segmentation (4D JIS) algorithm, which introduces a novel strategy to
jointly regularize a 4D seismic inversion problem and segment the 4D difference volume into percentages of acoustic
impedance changes. We validate our algorithm with the VTI Hess synthetic seismic and 4D Sleipner seismic datasets.
Furthermore, this repository comprehensively explains the data preparation workflow for 4D seismic inversion, including
time-shift nonlinear inversion and well-to-seismic tie. We provide a CPU and GPU-based implementations of the 4D JIS.

## Project structure

This repository is organized as follows:

- :open_file_folder: **data**: Folder containing the data. For the Sleipner seismic dataset please download
  the `94p07ful.sgy` and `01p07ful.sgy` seismic files from [https://co2datashare.org/dataset] and put them
  in `data/Sleipner_pysubsurface/Seismic/Post/`
- :open_file_folder: **prox4d**: Python library containing the 4D JIS, among other functions.
- :open_file_folder: **notebooks**: Set of Jupyter notebooks reproducing the experiments of the paper mentioned above.

## Notebooks

The following notebooks are provided:

- :orange_book: ``4DPoststackInversion_Hess.ipynb``: Notebook that contains the implementation of 4D JIS in the Hess VTI
  synthetic dataset and the benchmark 4D seismic inversion methods we compare to.

- :orange_book: ``4DPoststackInversion_Sleipner_94_01_IL120.ipynb``: Notebook that contains the implementation of 4D JIS
  in the Inline 120 of the 1994 and 2001 Sleipner seismic surveys (2007 reprocessing version)

- :orange_book: ``4DPoststackInversion_Sleipner_94_01_gpu.ipynb``:  Notebook that contains the implementation of 4D
  JIS (GPU-based) of subvolumes of the 1994 and 2001 Sleipner seismic surveys (2007 reprocessing version). This notebook
  requires the 3D time-shift corrected output of the notebook `Timeshift_Sleipner.ipynb`.

- :orange_book: ``Timeshift_Sleipner.ipynb ``: Notebook that contains the time-shift estimation (through non-linear
  inversion) of the 2001 Sleipner dataset.

- :orange_book: ``WellTie_Sleipner.ipynb``: Notebook that contains the well-to-seismic correlation of the Sleipner
  dataset. This notebook includes well-log editing, statistical wavelet estimation, and time-depth calibration.

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:

```
./install_env.sh
```

It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can
simply install your package:

```
pip install -e .
```

Remember to always activate the environment by typing:

```
conda activate prox4d
```

Similarly, you can use `environment_gpu.yml` and `install_env_gpu.sh` to create an environment that runs the 4D JIS
algorithm in GPU.

**Disclaimer:** For computer time, this research used the resources of the Supercomputing Laboratory at KAUST in Thuwal,
Saudi Arabia. All experiments have been carried on an AMD EPYC 7713P 64-Core Processor equipped with a single NVIDIA
TESLA A100 . Different environment configurations may be required for different combinations of workstation and GPU.

## Cite us

```
@article{romero2023,
      title={Seeing through the CO2 plume: joint inversion-segmentation of the Sleipner 4D Seismic Dataset}, 
      author={Juan Romero and Nick Luiken and Matteo Ravasi},
      year={2023},
      eprint={2303.11662},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph}
}
```



