import os
import box
import coord
import mrcHelper
import starHelper

import numpy as np

from absl import app
from absl import flags


class MSData():
    def __init__(self, mrc_data, star_data):
        self.mrc_data = mrc_data
        self.star_data = star_data

def downsample(inputs, use_factor=False, para1=None, para2=None):
    #This method executes a downsampling on the mrc
    #Inputs is a list of MrcData
    #If use_factor is True, para1 represents factor and para2 represents shape
    #Else para1 and para2 represents the target size(y, x)
    
    for i in range(len(inputs)):
        if use_factor:
            print("Processing %s ..." % ( inputs[i].name))
            inputs[i].data = mrcHelper.downsample_with_factor(
                inputs[i].data,
                factor=para1,
                shape=para2
            )
        else:
            print("Prcocessing %s ..." % (inputs[i].name))
            inputs[i].data = mrcHelper.downsample_with_size(
                inputs[i].data,
                size1=para1,
                size2=para2
            )
    return inputs

def mrc2array(inputs, image_size):
    #inputs is a list of MrcData
    array = np.zeros(
        (len(inputs), image_size, image_size, 1),
        dtype=np.float32
    )
    for i in range(len(inputs)):
        inputs[i].data = inputs[i].data.astype(np.float32)
        inputs[i].data /= 255
        mean = np.mean(inputs[i].data)
        std = np.std(inputs[i].data)
        inputs[i].data -= mean
        inputs[i].data /= std
        array[i,...] = np.expand_dims(inputs[i].data.astype(np.float32), axis=-1)
    return array

def label_downsample(data, label, t, w, h):
    downsampled_label = []
    if t == 'star':
        for i in range(len(label)):
            name = label[i].name
            content = starHelper.downsample_with_size(
                label[i].content,
                (w / data[i].header[0], h / data[i].header[1])
            )
            downsampled_label.append(starHelper.StarData(name, content))
    elif t == 'coord':
        for i in range(len(label)):
            name = label[i].name
            content = coord.downsample_with_size(
                label[i].content,
                (w / data[i].header[0], h / data[i].header[1])
            )
            downsampled_label.append(coord.CoordData(name, content))
    elif t == 'box':
        for i in range(len(label)):
            name = label[i].name
            content = box.downsample_with_size(
                label[i].content,
                (w / data[i].header[0], h / data[i].header[1])
            )
            downsampled_label.append(coord.CoordData(name, content))
    else:
        print('A valid type of label is required: star | coord | box')
    return downsampled_label 

def write_label(label):
    #TODO: merge all types of label
    pass

def read_label(label_path, label_type):
    if label_type == 'star':
        return starHelper.read_all_star(label_path)
    elif label_type == 'coord':
        return coord.read_all_coord(label_path)
    elif label_type == 'box':
        print('reading box')
        return box.read_all_box(label_path)
    else:
        print('A valid type is required: star | coord | box')
        return []

def main(argv):
    del argv
    data_path = FLAGS.data_path
    label_path = FLAGS.label_path
    data_dst = FLAGS.data_dst_path
    label_dst = FLAGS.label_dst_path
    target_size = FLAGS.target_size
    label_type = FLAGS.label_type
    
    mrc_data = mrcHelper.load_all_mrc(data_path)
    #averge frame
    for i in range(len(mrc_data)):
        if mrc_data[i].header[2] < 2:
            break
        avg_mrc = np.zeros_like(mrc_data[i].data[0,...])
        for j in range(mrc_data[i].header[2]):
            avg_mrc += mrc_data[i].data[j, ...]
        avg_mrc /= mrc_data[i].header[2]
        mrc_data[i].data = avg_mrc
    label = read_label(label_path, label_type)
    #debug:
    for k in range(len(mrc_data)):
        print(mrc_data[k].name, '\t', label[k].name)
    # downsampled_data = preprocess(data, False, para1=1024, para2=1024)
    mrc_data = downsample(mrc_data, False, para1=target_size, para2=target_size)
    
    #downsampled_label = []
    #for i in range(len(label)):
    #    name = label[i].name
    #    content = starHelper.downsample_with_size(
    #        label[i].content,
    #        (1024 / data[i].header[0], 1024 / data[i].header[1])
    #    )
    #   downsampled_label.append(starHelper.StarData(name, content))
    downsampled_label = label_downsample(
        mrc_data, label, 
        label_type, 
        target_size, target_size
    )
    print(len(downsampled_label))

    mrcHelper.write_mrc(mrc_data, dst=data_dst)
    #write_label(downsampled_label, label_type)
    print('writing label...')
    starHelper.write_star(downsampled_label, dst=label_dst)
    
def normalize_uint8(data):
    for i in range(np.shape(data)[0]):
        maximum, minimum = np.max(data[i,...]), np.min(data[i,...])
        data[i, ...] = (data[i, ...] - minimum) * 255 / (maximum - minimum) + 0.5
        data[i, ...] = max(data[i, ...], np.zeros_like(data[i, ...]))
        data[i, ...] = min(data[i, ...], 255 * np.ones_like(data[i, ...]))
    data = data.astype(np.float32)    

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_path", None, "path of data(mrc, etc.)")
    flags.DEFINE_string("label_path", None, "path of labels(star, etc.)")
    flags.DEFINE_string("data_dst_path", None, "target to store processed data")
    flags.DEFINE_string("label_dst_path", None, "target to store processed labels")
    flags.DEFINE_string("label_type", "star", "type of label, star | coord | box")
    flags.DEFINE_integer("target_size", 1024, "the size of processed data")
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("label_path")
    flags.mark_flag_as_required("data_dst_path")
    flags.mark_flag_as_required("label_dst_path")
    app.run(main)