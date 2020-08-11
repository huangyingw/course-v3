#!/bin/zsh
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

conda update conda
pip install fastai
conda install -c pytorch -c fastai fastai
conda install -c conda-forge \
    jupytext \
    neovim
