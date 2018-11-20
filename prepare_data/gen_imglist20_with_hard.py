import argparse
import numpy as np
import numpy.random as npr
import os,sys
sys.path.append(os.getcwd())
from config import config

def gen_imglists_with_hard_aug(size=20):
    if size == 20:
        net = "pnet20"
    elif size == 24:
        net = "rnet"
    elif size == 48:
        net = "onet"

    with open('%s/prepare_data/%s/part_%s.txt'%(config.root,net, size), 'r') as f:
        part = f.readlines()
		
    with open('%s/prepare_data/%s/pos_%s.txt'%(config.root,net, size), 'r') as f:
        pos = f.readlines()

    with open('%s/prepare_data/%s/pos_aug_%s.txt'%(config.root,net, size), 'r') as f:
        pos_aug = f.readlines()

    with open('%s/prepare_data/%s/neg_%s.txt'%(config.root,net, size), 'r') as f:
        neg = f.readlines()
	
    with open('%s/prepare_data/%s/neg_hard_%s.txt'%(config.root,net, size), 'r') as f:
        neg_hard = f.readlines()

    if size == 20:
        pos_num = config.base_num*200000 #3000000
        pos_aug_num = config.base_num*100000 #2000000
        part_num = config.base_num*100000  #100000
        neg_num = config.base_num*600000  #12000000
        neg_hard_num = config.base_num*300000 #2000000
		
    if size == 24:
        pos_num = config.base_num*200000 #5700000
        pos_aug_num = config.base_num*100000 #0
        part_num = config.base_num*300000 #5700000 
        neg_num = config.base_num*600000 #11700000
        neg_hard_num = config.base_num*300000 #5300000	
        #pos = pos[0:5700000]
        #pos_aug = pos_aug[0:2400000]
        #part = part[0:5700000]
        #neg = neg[0:11700000]

    
    with open("%s/prepare_data/%s/train_%s_with_hard.txt"%(config.root,net, size), "w") as f:
        if len(pos) > pos_num:
            pos_keep = npr.choice(len(pos), size=pos_num, replace=False)
            print('pos_num=%d'%pos_num)
            for i in pos_keep:
                f.write(pos[i])			
        else:
            print('pos_num=%d'%len(pos))
            f.writelines(pos)
        if len(pos_aug) > pos_aug_num:
            pos_aug_keep = npr.choice(len(pos_aug), size=pos_aug_num, replace=False)
            print('pos_aug_num=%d'%pos_aug_num)
            for i in pos_aug_keep:
                f.write(pos_aug[i])			
        else:
            print('pos_aug_num=%d'%len(pos_aug))
            f.writelines(pos_aug)
        if len(part) > part_num:
            part_keep = npr.choice(len(part), size=part_num, replace=False)
            print('part_num=%d'%part_num)
            for i in part_keep:
                f.write(part[i])			
        else:
            print('part_num=%d'%len(part))
            f.writelines(part)
        if len(neg) > neg_num:
            neg_keep = npr.choice(len(neg), size=neg_num, replace=False)
            print('neg_num=%d'%neg_num)
            for i in neg_keep:
                f.write(neg[i])			
        else:
            print('neg_num=%d'%len(neg))
            f.writelines(neg)

        if len(neg_hard) > neg_hard_num:
            neg_hard_keep = npr.choice(len(neg_hard), size=neg_hard_num, replace=False)
            print('neg_hard_num=%d'%neg_hard_num)
            for i in neg_hard_keep:
                f.write(neg_hard[i])			
        else:
            print('neg_hard_num=%d'%len(neg_hard))
            f.writelines(neg_hard)


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
    gen_imglists_with_hard_aug(int(args.size))