#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main_sp_block.py --config cfg/config_500k_block.yaml --log main_b.log
python main_sp_block.py --config cfg/config_500k_block_2.yaml --log main_b.log
#python main_sp.py --config cfg/config_500k_2.yaml --log main.log
#python main.py --config cfg/config_500k.yaml