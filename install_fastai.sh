#!/bin/zsh
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

conda update conda
conda install -c pytorch -c fastai fastai pytorch
conda install -c conda-forge \
    jupytext \
    neovim
