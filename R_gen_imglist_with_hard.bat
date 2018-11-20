python prepare_data\gen_imglist20_with_hard.py --size 24
copy prepare_data\rnet\train_24_with_hard.txt data\mtcnn\imglists
del data\cache\mtcnn_train_24_with_hard_gt_roidb.pkl
pause