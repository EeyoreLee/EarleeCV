# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/21 10:47:19
@author: lichunyu
'''
 
import xml.etree.ElementTree as ET
import glob
import os


classes = ["Crack", "Netv", "AbnormalManhole", "Pothole", "Marking"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def only_filename(file_path):
    return os.path.split(file_path)[-1]


def convert_annotation(file_list, output_file_path):

    all_bbox = []

    for file_path in file_list:

        tree=ET.parse(file_path)
        root = tree.getroot()
        size = root.find('size')  
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        b_list = []
    
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes :
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = [str(xmlbox.find('xmin').text), str(xmlbox.find('ymin').text), str(xmlbox.find('xmax').text), str(xmlbox.find('ymax').text), str(cls_id)]
            b_list.append(','.join(b))

        all_bbox.append([file_path, b_list])

    with open(output_file_path, 'w') as f:
        length = len(all_bbox)
        for idx, b in enumerate(all_bbox):
            f.write(only_filename(b[0]).replace('xml', 'jpg') + ' ' + ' '.join(b[1]) + '\n')

    return 0


if __name__ == '__main__':
    file_list = glob.glob('/ai/223/person/lichunyu/datasets/chelianwang/dataset/train/annotations/*.xml')
    _ = convert_annotation(file_list, '/ai/223/person/lichunyu/datasets/chelianwang/dataset/train/image/train.txt')
    # x = '/ai/223/person/lichunyu/datasets/chelianwang/dataset/train/annotations/*.xml'
    # y = os.path.split(x)[-1].replace('xml', 'jpg')
    pass