#!/usr/bin/env bash
source ~/venv3.6/bin/activate
pip uninstall -y keras-tcn
cd ..
pip install . --upgrade
cd understands
export CUDA_VISIBLE_DEVICES=; python main.py | grep acc