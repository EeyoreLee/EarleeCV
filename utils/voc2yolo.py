# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/21 10:47:19
@author: lichunyu
'''
 
import xml.etree.ElementTree as ET
import glob


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
            cls_id = classes.index(cls) + 1
            xmlbox = obj.find('bndbox')
            b = [str(xmlbox.find('xmin').text), str(xmlbox.find('xmax').text), str(xmlbox.find('ymin').text), str(xmlbox.find('ymax').text), str(cls_id)]
            b_list.append(','.join(b))

        all_bbox.append([file_path, b_list])

    with open(output_file_path, 'w') as f:
        length = len(all_bbox)
        for idx, b in enumerate(all_bbox):
            if idx == length - 2:
                f.write(b[0] + ' ' + ' '.join(b[1]))
            else:
                f.write(b[0] + ' ' + ' '.join(b[1]) + '\n')

    return 0


if __name__ == '__main__':
    file_list = glob.glob('/ai/223/person/lichunyu/datasets/chelianwang/dataset/train/annotations/*.xml')
    _ = convert_annotation(file_list, 'train.txt')
    pass