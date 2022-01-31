#!/bin/sh

ENV_DIR='.venv'

if [[ ! -d $ENV_DIR ]]
then
    python3 -m venv $ENV_DIR
    echo "Activating environment."
    source $ENV_DIR/bin/activate
    pip install --upgrade pip
else
    echo "Activating environment."
    source $ENV_DIR/bin/activate
fi
pip install -r requirements_py3.txt -f https://download.pytorch.org/whl/torch_stable.html
cd src/
export PYTHONPATH=$(pwd)
