#!/bin/bash
VENV_DIR=${1:-".venv/RLFN"}
conda env create -f conda.yaml --prefix $VENV_DIR
