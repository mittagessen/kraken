#!/bin/bash
set -e

echo "Converting conda package..."
conda convert --platform all $HOME/miniconda/conda-bld/linux-64/kraken-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/kraken-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
