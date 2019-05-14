#!/bin/bash
set -e

OS=$TRAVIS_OS_NAME-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=$TRAVIS_TAG
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u mittagessen $CONDA_BLD_PATH/$OS/kraken-`date +%Y.%m.%d`-0.tar.bz2 --force

echo "Successfully deployed to Anaconda.org."
exit 0
