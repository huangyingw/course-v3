#!/bin/zsh
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

find . -type f -name \*.ipynb -exec jupytext --set-formats ipynb,py {} \;
