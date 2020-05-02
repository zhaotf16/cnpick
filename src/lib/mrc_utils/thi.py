import os
import cv2
import json

class ThiData():
    def __init__(self, name="", content=[]):
        self.name = name
        self.content = content

def read_thi(path):
    coordinates = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            content = line.split()
            coordinates.append((int(float(content[0])), int(float(content[1]))))
    return coordinates

def read_all_coord(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return None
    if not path.endswith('/'):
        path += '/'
    thi = []
    for file in os.listdir(path):
        if file.endswith('.thi'):
            name, _ = os.path.splitext(file)
            print("Loading %s.thi..." % (name))
            content = read_thi(path + file)
            thi.append(ThiData(name, content))
    thi.sort(key=lambda s: s.name)
    return thi

def downsample_with_size(coordinates, scale):
    #scale is a tuple (scale_x, scale_y)
    downsampled = []
    for i in range(len(coordinates)):
        downsampled.append((
            int(coordinates[i][0] * scale[0]),
            int(coordinates[i][1] * scale[1])
        ))
    return downsampled
'''
def write_thi(inputs, dst):
    print('write_thi')
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not dst.endswith('/'):
        dst += '/'
    #for mrc_data in inputs:
    for thi_data in inputs:
        print("Writing %s.thi ..." % (thi_data.name))
        with open(dst+thi_data.name+'.thi', "w") as f:
            f.write('[Particle Coordinates: X Y Value]\n')
            for item in thi_data.content:
                f.write("%d\t%d\t%f\n" % (item[0], item[1], item[2]))
            f.write('[End]')
'''
def write_thi(inputs, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not dst.endswith('/'):
        dst += '/'
    #for mrc_data in inputs:
    for k in inputs:
        print("Writing %s.thi ..." % (k))
        with open(dst+k+'.thi', "w") as f:
            f.write('[Particle Coordinates: X Y Value]\n')
            for item in inputs[k]:
                f.write("%d\t%d\t%f\n" % (item[0], item[1], item[2]))
            f.write('[End]')

