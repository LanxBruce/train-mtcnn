import numpy as np
import mxnet as mx
import argparse
import os
import cPickle
import cv2
import sys
sys.path.append(os.getcwd())
from config import config
from core.symbol import P_Net, P_Net20, R_Net, O_Net
from core.imdb import IMDB
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector20 import MtcnnDetector
from utils import *

def save_hard_example(net):

    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    anno_file = "%s/prepare_data/wider_annotations/anno.txt"%config.root
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    if net == "pnet20":
        image_size = 20
    if net == "rnet":
        image_size = 24
    if net == "onet":
        image_size = 48

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print "processing %d images in total"%num_of_images

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = annotation[0]

        boxes = map(float, annotation[1:])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    image_dir = "%s/data/WIDER_train/images"%config.root
    neg_save_dir = "%s/prepare_data/%s/negative_hard"%(config.root,image_size)
    save_path = "%s/prepare_data/%s"%(config.root,net)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    f2 = open(os.path.join(save_path, 'neg_hard_%d.txt'%image_size), 'w')
    
    det_boxes = cPickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    print len(det_boxes)
    print num_of_images
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print "%d images done"%image_done
        image_done += 1

        if dets.shape[0]==0:
            continue
        full_img_name = im_idx
        print full_img_name
        img = cv2.imread(full_img_name)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.1:
                # Iou with all gts must below 0.1
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("%s/%s"%(neg_save_dir, n_idx) + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

    f2.close()


def test_net(root_path, dataset_path, image_set, prefix, epoch,
             batch_size, ctx, test_mode="hardpnet20",
             thresh=[0.6, 0.6, 0.7], min_face_size=24,
             stride=4, slide_window=False, shuffle=False, vis=False):

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net20("test"), 20, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net20("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
        RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
        ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
        detectors[2] = ONet
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    
    imdb = IMDB("wider", image_set, root_path, dataset_path, 'test')
    gt_imdb = imdb.gt_imdb()

    test_data = TestLoader(gt_imdb)
    detections = mtcnn_detector.detect_face(imdb, test_data, vis=vis)
    

    if test_mode == "hardpnet20":
        net = "pnet20"
    if test_mode == "hardrnet":
        net = "rnet"
    elif test_mode == "hardonet":
        net = "onet"
    
    save_path = "%s/prepare_data/%s"%(config.root,net)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        cPickle.dump(detections, f, cPickle.HIGHEST_PROTOCOL)
    

    save_hard_example(net)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='%s/data'%config.root, type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='%s/data/mtcnn'%config.root, type=str)
    parser.add_argument('--image_set', dest='image_set', help='image set',
                        default='train', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be hardpnet20, hardrnet or hardonet',
                        default='hardpnet20', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['%s/model/pnet20'%config.root, '%s/model/rnet'%config.root, '%s/model/onet'%config.root], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.6, 0.7, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=4, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = mx.gpu(args.gpu_id)
    if args.gpu_id == -1:
        ctx = mx.cpu(0)
    test_net(args.root_path, args.dataset_path, args.image_set, args.prefix,
             args.epoch, args.batch_size, ctx, args.test_mode,
             args.thresh, args.min_face, args.stride,
             args.slide_window, args.shuffle, args.vis)
