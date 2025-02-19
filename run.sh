#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main_sp.py --config cfg/config.yaml