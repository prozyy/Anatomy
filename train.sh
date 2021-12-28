#!bin/bash
# Human3.6m
# python train.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 --randnum 50
# Human3.6m_24
# python train.py -d h36m_24 -e 80 -k gt -arc 3,3,3,3,3 --randnum 50 --num-joints-in 24 --num-joints-out 24

# python train.py  -d  h36m_24_aist  -e 80  -k  gt   -arc   3,3,3,3,3   --randnum 50 --num-joints-in 24 --num-joints-out 24

python train.py  -d  h36m_24_aist_megapose  -e 80  -k  gt   -arc   3,3,3,3,3   --randnum 50 --num-joints-in 24 --num-joints-out 24

