set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python prepare_data/gen_hard_example20.py --test_mode hardrnet --prefix model/pnet20_hard --epoch 13 --thresh 0.3
pause