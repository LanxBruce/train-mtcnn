import argparse
import numpy as np
import numpy.random as npr
import cv2
import os,sys
sys.path.append(os.getcwd())
from config import config
from tools.image_processing import darker
from tools.image_processing import brighter
from tools.image_processing import SaltAndPepper
from tools.image_processing import addGaussianNoise

	
def augment_data(size = 20):

    if size == 20:
        net = "pnet20"
    elif size == 24:
        net = "rnet"
    elif size == 48:
        net = "onet"

    im_save_dir = "%s/prepare_data/%s"%(config.root,size)
    txt_save_dir = "%s/prepare_data/%s"%(config.root,net)
    if not os.path.exists(im_save_dir):
        os.mkdir(im_save_dir)
    if not os.path.exists(txt_save_dir):
        os.mkdir(txt_save_dir)
    pos_aug_dir = '%s/positive_aug'%im_save_dir
    if not os.path.exists(pos_aug_dir):
        os.mkdir(pos_aug_dir)

    with open('%s/pos_%s.txt'%(txt_save_dir, size), 'r') as f:
        pos = f.readlines()

    n_idx = 0
    with open("%s/pos_aug_%s.txt"%(txt_save_dir, size), "w") as f:
        for i in range(len(pos)):
            m_splits = pos[i].split(' ')
            in_filename = m_splits[0] + '.jpg'
            print in_filename
            img = cv2.imread(in_filename)
		
            """
            img_noise1 = SaltAndPepper(img, 0.3)
            img_noise2 = addGaussianNoise(img, 0.3)
            filename = "%s/%s"%(pos_aug_dir, n_idx)
            f.write(filename + ' 1 %s %s %s %s\n'%(m_splits[2],m_splits[3],m_splits[4],m_splits[5]))
            cv2.imwrite(filename+'.jpg', img_noise1)
            n_idx = n_idx+1
            filename = "%s/%s"%(pos_aug_dir, n_idx)
            f.write(filename + ' 1 %s %s %s %s\n'%(m_splits[2],m_splits[3],m_splits[4],m_splits[5]))
            cv2.imwrite(filename+'.jpg', img_noise2)
            n_idx = n_idx+1

        
            """
            img_brighter = brighter(img, 1.2)
            filename = "%s/%s"%(pos_aug_dir, n_idx)
            f.write(filename + ' 1 %s %s %s %s\n'%(m_splits[2],m_splits[3],m_splits[4],m_splits[5].split('\n')[0]))
            cv2.imwrite(filename+'.jpg', img_brighter)
            n_idx = n_idx+1
        
            img_darker = darker(img, 0.85)
            filename = "%s/%s"%(pos_aug_dir, n_idx)
            f.write(filename + ' 1 %s %s %s %s\n'%(m_splits[2],m_splits[3],m_splits[4],m_splits[5].split('\n')[0]))
            cv2.imwrite(filename+'.jpg', img_darker)
            n_idx = n_idx+1
        
def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', dest='size', help='20 or 24 or 48',
                        default='20', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    augment_data(int(args.size))	