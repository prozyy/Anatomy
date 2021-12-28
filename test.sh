#!bin/bash

# test
# python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 --evaluate pretrained_model.bin

# render
python test.py  -d  h36m_24_aist_megapose   -k  gt   -arc   3,3,3,3,3   --evaluate epochfinal_60.bin --num-joints-in 24 --num-joints-out 24 --render --viz-video data/testData/test.mp4 --viz-output result.mp4