import cv2
import threading
from tools import image_processing
import numpy as np
import math

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.labels, self.bboxes = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.labels, self.bboxes
        except Exception:
            return None

def get_minibatch_thread(imdb, num_classes, im_size):
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        h, w, c = im.shape
        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']

        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = image_processing.transform(im)
        processed_ims.append(im_tensor)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)

    return processed_ims, cls_label, bbox_reg_target

def get_minibatch(imdb, num_classes, im_size, thread_num = 4):
    # im_size: 12, 24 or 48
    #flag = np.random.randint(3,size=1)
    num_images = len(imdb)
    thread_num = max(2,thread_num)
    num_per_thread = math.ceil(num_images/thread_num)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb,num_classes,im_size))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()

    for t in range(thread_num):
        cur_process_ims, cur_cls_label, cur_bbox_reg_target = threads[t].get_result()
        #print len(cur_process_ims)
        #print len(cur_cls_label)
        #print len(cur_bbox_reg_target)
        processed_ims = processed_ims + cur_process_ims
        cls_label = cls_label + cur_cls_label
        bbox_reg_target = bbox_reg_target + cur_bbox_reg_target
    
    #print len(processed_ims)
    #print len(cls_label)
    #print len(bbox_reg_target)
    im_array = np.vstack(processed_ims)
    label_array = np.array(cls_label)
    bbox_target_array = np.vstack(bbox_reg_target)
    '''
    bbox_reg_weight = np.ones(label_array.shape)
    invalid = np.where(label_array == 0)[0]
    bbox_reg_weight[invalid] = 0
    bbox_reg_weight = np.repeat(bbox_reg_weight, 4, axis=1)
    '''
    if im_size == 12:
        label_array = label_array.reshape(-1, 1)

    data = {'data': im_array}
    label = {'label': label_array,
             'bbox_target': bbox_target_array}

    return data, label

def get_testbatch(imdb):
    assert len(imdb) == 1, "Single batch only"
    im = cv2.imread(imdb[0]['image'])
    im_array = im
    data = {'data': im_array}
    label = {}
    return data, label
