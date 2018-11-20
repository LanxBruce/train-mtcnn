set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_R_net.py --lr 0.0003 --image_set train_24_with_hard --end_epoch 8 --prefix model/rnet --lr_epoch 8,14,100 --batch_size 500 --thread_num 24
python example\train_R_net.py --lr 0.001 --image_set train_24_with_hard --end_epoch 30 --prefix model/rnet --epoch 8 --begin_epoch 8 --resume --lr_epoch 8,14,100 --batch_size 10000 --thread_num 24 --frequent 20
pause 