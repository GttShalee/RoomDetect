"""
 This file is aimed to transform xml to text so that the
 Yolo can use it for training
 Before using, you need to change the dir destination
"""
import os
import xml.etree.ElementTree as ET

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(os.path.join(output_dir, os.path.basename(xml_file).replace('.xml', '.txt')), 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            bb = convert_bbox((w, h), b)
            out_file.write(f"{cls} {' '.join([str(a) for a in bb])}\n")

def batch_convert(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".xml"):
            convert_annotation(os.path.join(input_dir, file), output_dir)

# 使用示例
batch_convert('datasets/annotations/xmls', 'datasets/class_head/labels/train')