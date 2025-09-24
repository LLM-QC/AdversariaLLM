#!/bin/bash

# module load python/3.10 arrow
# # ENVDIR=$SLURM_TMPDIR/venv
# ENVDIR=venv
# virtualenv --no-download $ENVDIR
# source $ENVDIR/bin/activate
# 
# pip install -r requirements-drac.txt --no-index
# pip install $PYWHEELS/locate-1.1.1-py3-none-any.whl --no-index
# pip install $PYWHEELS/litellm-1.61.16-py3-none-any.whl --no-index
# pip install $PYWHEELS/wordfreq-3.1.1-py3-none-any.whl --no-index
# pip install -e $HOME/jailbreakbench --no-index
# pip install -e .

# Default to local ./venv
USE_SLURM_TMP=false

# Parse optional flag
for arg in "$@"; do
    case $arg in
        --use-slurm-tmp)
            USE_SLURM_TMP=true
            shift
            ;;
    esac
done

# Set environment directory
if [ "$USE_SLURM_TMP" = true ]; then
    ENVDIR="$SLURM_TMPDIR/venv"
else
    ENVDIR="./venv"
fi

echo "Creating virtual environment in: $ENVDIR"

module load python/3.11 arrow
virtualenv --no-download "$ENVDIR"
source "$ENVDIR/bin/activate"

pip install -r requirements-drac.txt --no-index
pip install "$PYWHEELS/locate-1.1.1-py3-none-any.whl" --no-index
pip install "$PYWHEELS/litellm-1.61.16-py3-none-any.whl" --no-index
pip install "$PYWHEELS/wordfreq-3.1.1-py3-none-any.whl" --no-index
pip install "$PYWHEELS/peft-0.14.0-py3-none-any.whl" --no-index
pip install -e "$HOME/jailbreakbench" --no-index
pip install -e .

echo "Virtual environment setup complete."