import os
import cv2
import json

class BoxData():
    def __init__(self, name="", content=[]):
        self.name = name
        self.content = content

def read_box(path):
    coordinates = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            content = line.split()
            coordinates.append(
                (int(content[0]), int(content[1]), int(content[2]), int(content[3]))
            )
    return coordinates

def read_all_box(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return None
    if not path.endswith('/'):
        path += '/'
    boxes = []
    
    for file in os.listdir(path):
        if file.endswith('.box'):
            name, _ = os.path.splitext(file)
            print("Loading %s.box..." % (name))
            content = read_box(path + file)
            boxes.append(BoxData(name, content))
    boxes.sort(key=lambda s: s.name)
    return boxes

def downsample_with_size(coordinates, scale):
    #scale is a tuple (scale_x, scale_y)
    downsampled = []
    for i in range(len(coordinates)):
        downsampled.append((
            int(coordinates[i][0] * scale[0]),
            int(coordinates[i][1] * scale[1]),
            int(coordinates[i][2] * scale[0]),
            int(coordinates[i][3] * scale[1])
        ))
    return downsampled

def box2coco(data, json_name):
    root_path = "10017_1024/"
    images, categories, annotations = [], [], []
    
    category_dict = {"Falcon": 1}
    
    for cat_n in category_dict:
        categories.append({"supercategory": "", "id": category_dict[cat_n], "name": cat_n})

    img_id = 0
    anno_id_count = 0
    for box in data:
        #anno_id_count = 0
        img_name = box.name + '.png'
        img_name = img_name.replace('_autopick','')
        img_name = img_name.replace('_DW', '')
        img_name = img_name.replace('_manualpick', '')
        img_name = img_name.replace('_empiar', '')
        print(img_name)
        img_cv2 = cv2.imread(root_path + img_name)
        [height, width, _] = img_cv2.shape
        # images info
        images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})
        for box in box.content:
            """
            annotation info:
            id : anno_id_count
            category_id : category_id
            bbox : bbox
            segmentation : [segment]
            area : area
            iscrowd : 0
            image_id : image_id
            """
            category_id = category_dict["Falcon"]
            w, h = box[2], box[3]
            x1 = max(box[0] - w/2, 1)
            y1 = max(box[1] - h/2, 1)
            x2 = min(box[0] + w/2, width)
            y2 = min(box[1] + h/2, height)

            bbox = [x1, y1, w, h]
            segment = [x1, y1, x2, y1, x2, y2, x1, y2]
            area = w * h

            anno_info = {'id': anno_id_count, 'category_id': category_id, 'bbox': bbox, 'segmentation': [segment],
                        'area': area, 'iscrowd': 0, 'image_id': img_id}
            annotations.append(anno_info)
            anno_id_count += 1
 
        img_id += 1
 
    all_json = {"images": images, "annotations": annotations, "categories": categories}
    with open(json_name+".json", "w") as outfile:
        json.dump(all_json, outfile)

if __name__ == '__main__':
    boxes = read_all_box('10017_mrc_1024')
    train = boxes[0:160]
    valid = boxes[160:180]
    test = boxes[180:]
    box2coco(train, 'train')
    box2coco(valid, 'val')
    box2coco(test, 'test')

#if __name__ == '__main__':
#   img = cv2.imread('EMPIAR-10017/Falcon_2012_06_13-00_19_28_0.png')
#   boxes = read_box('data/Falcon_2012_06_13-00_19_28_0.coord')
#   box_size = 80
#   boxes = 0
#   for b in boxes:
#       x1 = int(c[0] - box_size / 2)
#       y1 = int(c[1] - box_size / 2)
#       x2 = int(c[0] + box_size / 2)
#       y2 = int(c[1] + box_size / 2)
#       img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
#   cv2.imwrite('sample.png', img)