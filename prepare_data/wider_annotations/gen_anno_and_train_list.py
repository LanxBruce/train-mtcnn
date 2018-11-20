from wider_loader import WIDER
import sys,os
sys.path.append(os.getcwd())
from config import config
import cv2
import time

#wider face original images path
path_to_image = '%s/data/WIDER_train/images'%config.root

#matlab file path
file_to_label = '%s/prepare_data/wider_annotations/wider_face_train.mat'%config.root

#target file path
target_anno_file = '%s/prepare_data/wider_annotations/anno.txt'%config.root

target_train_list_file = '%s/data/mtcnn/imglists/train.txt'%config.root

line_count = 0
box_count = 0

print 'start transforming....'
t = time.time()

wider = WIDER(file_to_label, path_to_image)

with open(target_anno_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))

        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

wider = WIDER(file_to_label, path_to_image)

with open(target_train_list_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print 'end transforming'

print 'spend time:%ld'%st
print 'total line(images):%d'%line_count
print 'total boxes(faces):%d'%box_count


