# GMX NNPot Wrapper

This repository contains a collection of example wrappers around popular Neural Network potentials for use with the [NNPot interface in GROMACS](https://manual.gromacs.org/2025.0/reference-manual/special/nnpot.html). We also provide an example notebook to export the models, as well as an example use case for a simulation of Alanine Dipeptide solvated in water. To try these examples out for yourself, clone this repository, create the conda environment specified in `environment.yaml` and go through the example notebook.

## How-to

If you want to learn how to write a wrapper for your own models for use within GROMACS, have a look at the examples in the `models` directory, as well as the `export.ipynb` notebook for how to export them. The following guide only references instructions on Linux, as this is the only platform for which NNP support in GROMACS is tested. 

To run the code in this repository, first create a conda environment with the necessary packages:
```console
$ conda env create -f environment.yaml
$ conda activate nnpot
```

Try to export one/some of the models in the example notebook. There, you will also find out the Torch and CUDA versions you are using. This is important, because in order to be able to run simulations with the compiled NNPs, we need to rebuild GROMACS with Libtorch support. Libtorch is the C++ API of Pytorch, and it is important for the versions to match. For example, if youre using Pytorch 2.4.1 and CUDA 12.4, you can get a matching Libtorch version by runninng
```console
$ wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip
$ unzip libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip
```

For a full guide on how to build GROMACS for your system, please check out the [install guide](https://manual.gromacs.org/2025.0/install-guide/index.html). Assuming you have downloaded the latest version of the GROMACS tarball, to install GROMACS with NNP support:
```console
$ tar xfz gromacs-2025.0.tar.gz
$ cd gromacs-2025.0
$ mkdir build
$ cd build
$ cmake .. -DGMX_BUILD_OWN_FFTW=ON -DGMX_NNPOT=TORCH -DCMAKE_PREFIX_PATH=/path/to/libtorch -DREGRESSIONTEST_DOWNLOAD=ON
$ make
$ make check
$ sudo make install
$ source /usr/local/gromacs/bin/GMXRC
```

During the CMake stage, make sure that NNPot support was properly configured by checking the CMake log.

Now that we have installed GROMACS, let's see how to run the example in this repository. The example system is an alanine dipeptide molecule solvated in water, and we want to model the peptide with our NNP, while treating the rest of the system (i.e. the water molecules) classically.
Check out the `md.mdp` file for the parameters we need to set. Make sure that the path in `nnpot-modelfile` points to a compiled model (absolute or relative to the GROMACS working directory). The `nnpot-input-group` specifies an index group that we want to use as an input for the NNP, and to apply the resulting forces on. There are some default groups available, but you can also create your own (see [`gmx make_ndx`](https://manual.gromacs.org/2025.0/onlinehelp/gmx-make_ndx.html)).

Then, we can simply run
```console
# To run models on the GPU (required with the CUAEV and NNPOps extensions), uncomment the following line
# export GMX_NN_DEVICE=cuda
$ gmx grompp -f md.mdp -c conf.gro -p topol.top -o md.tpr
$ gmx mdrun -deffnm md
```

## References

1. TorchANI repo: https://github.com/aiqm/torchani
2. MACE repo: https://github.com/ACEsuit/mace
3. emle-engine repo: https://github.com/chemle/emle-engine