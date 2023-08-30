import os
import xml.etree.ElementTree as ET
import json

def parse_voc_annotation(ann_dir, img_dir, data_type, labels=[]):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(os.path.join(ann_dir, ann))
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['file_name'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

def voc2coco(ann_dir, img_dir, output_json, labels=[]):
    all_imgs, seen_labels = parse_voc_annotation(ann_dir, img_dir, 'train', labels)

    data = {'images': [], 'categories': [], 'annotations': []}
    if len(labels) > 0:
        for label in labels:
            data['categories'].append({'id': len(data['categories']), 'name': label})
    else:
        for label in seen_labels:
            data['categories'].append({'id': len(data['categories']), 'name': label})

    for img_id, img in enumerate(all_imgs):
        image = {'file_name': img['file_name'], 'height': img['height'], 'width': img['width'], 'id': img_id}
        data['images'].append(image)

        for anno in img['object']:
            category_id = [cat['id'] for cat in data['categories'] if cat['name'] == anno['name']][0]
            bbox = [anno['xmin'], anno['ymin'], anno['xmax'] - anno['xmin'], anno['ymax'] - anno['ymin']]

            data['annotations'].append({
                'area': bbox[2] * bbox[3],
                'bbox': bbox,
                'category_id': category_id,
                'id': len(data['annotations']),
                'image_id': img_id,
                'iscrowd': 0,
                'segmentation': []
            })

    with open(output_json, 'w') as json_file:
        json.dump(data, json_file)

if __name__ == "__main__":
    #  paths to the Pascal VOC 2012 dataset directories
    VOC2012_PATH = r"C:\Users\katta\mscproject\VOCdevkit\VOC2012"
    ANNOTATIONS_DIR = os.path.join(VOC2012_PATH, "Annotations")
    IMAGES_DIR = os.path.join(VOC2012_PATH, "JPEGImages")

    # Converts annotations to COCO format
    voc2coco(ANNOTATIONS_DIR, IMAGES_DIR, "pascal_voc_2012_train.json")
