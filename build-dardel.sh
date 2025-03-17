#!/bin/bash
# This is a build script to compile GROMACS with NNPot support using the ROCm stack
# on the Dardel system at PDC.

set -e

ml PDC/23.12
ml cpe/23.12
ml rocm/5.7.0
ml craype-accel-amd-gfx90a
ml cray-fftw/3.3.10.6
ml heffte/2.4.0-cpeAMD-23.12-gpu
ml hwloc
ml cmake

echo "Getting source ..."
DIR=$(pwd)
mkdir -p ${HOME}/software/sources
cd ${HOME}/software/sources
git clone https://gitlab.com/gromacs/gromacs.git
cd gromacs
git checkout v2025.1

mkdir -p build
cd build
rm -rf *

echo "Configuring GROMACS ..."
export PYTORCH_ROCM_ARCH=gfx90a
cmake .. \
	-DCMAKE_INSTALL_PREFIX=${HOME}/software/packages/gromacs-2025.1-hip-torch \
	-DCMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang \
	-DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++ \
	-DCMAKE_HIP_COMPILER=${ROCM_PATH}/bin/amdclang++ \
	-DGMX_GPU=HIP -DGMX_HIP_TARGET_ARCH=gfx90a \
	-DGMX_NNPOT=TORCH \
	-DCMAKE_PREFIX_PATH="${ROCM_PATH};${HOME}/software/sources/libtorch-2.0.1-rocm5.4.2" \
	-DGMX_HWLOC=ON \
	-DGMX_BUILD_OWN_FFTW=OFF \
	-DGMX_GPU_FFT_LIBRARY=rocFFT \
	-DGMX_USE_HEFFTE=ON -DHeffte_ROOT=${EBROOTHEFFTE} \
	-DHeffte_ENABLE_ROCM=ON -DHeffte_ROCM_ROOT=${ROCM_PATH}

echo "Building GROMACS ..."
make -j 32

echo "Installing GROMACS ..."
make -j 32 install

echo "Successfully installed GROMACS."
cd $DIR
