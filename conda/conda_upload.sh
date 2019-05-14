#!/bin/bash
set -e

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`date +%Y.%m.%d`
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u mittagessen -l nightly $CONDA_BLD_PATH/$OS/kraken-`date +%Y.%m.%d`-0.tar.bz2 --force

echo "Successfully deployed to Anaconda.org."
exit 0
