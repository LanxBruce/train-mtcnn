import argparse
import numpy as np
import numpy.random as npr
import os,sys
sys.path.append(os.getcwd())
from config import config

def gen_imglists(size=20):
    if size == 20:
        net = "pnet20"
    elif size == 24:
        net = "rnet"
    elif size == 48:
        net = "onet"

    with open('%s/prepare_data/%s/pos_%s.txt'%(config.root,net, size), 'r') as f:
        pos = f.readlines()
    
    with open('%s/prepare_data/%s/pos_aug_%s.txt'%(config.root,net, size), 'r') as f:
        pos_aug = f.readlines()
		
    with open('%s/prepare_data/%s/part_%s.txt'%(config.root,net, size), 'r') as f:
        part = f.readlines()
		
    with open('%s/prepare_data/%s/neg_%s.txt'%(config.root,net, size), 'r') as f:
        neg = f.readlines()

    with open("%s/prepare_data/%s/train_%s.txt"%(config.root,net, size), "w") as f:
        if len(pos) > 650000:
            pos_keep = npr.choice(len(pos), size=650000, replace=False)
            for i in pos_keep:
                f.write(pos[i])			
        else:
            f.writelines(pos)
        
        if len(pos_aug) > 650000:
            pos_aug_keep = npr.choice(len(pos_aug), size=650000, replace=False)
            for i in pos_aug_keep:
                f.write(pos_aug[i])			
        else:
            f.writelines(pos_aug)
        
        if len(part) > 1300000:
            part_keep = npr.choice(len(part), size=1300000, replace=False)
            for i in part_keep:
                f.write(part[i])			
        else:
            f.writelines(part)
        if len(neg) > 3900000:
            neg_keep = npr.choice(len(neg), size=3900000, replace=False)
            for i in neg_keep:
                f.write(neg[i])			
        else:
            f.writelines(neg)


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
    gen_imglists(int(args.size))
