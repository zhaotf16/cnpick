import os
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

def star2label(inputs, image_size, grid_size=64, particle_size=220):
    #inputs is a list of 
    '''
    label = np.zeros(
        (len(inputs), grid_size, grid_size, 5), 
        dtype=np.float32
    )
    grid_scale = image_size // grid_size
    for i in range(len(inputs)):
        for coord in inputs[i].content:
            x_index = int(coord[0] - 1) // grid_scale
            y_index = int(coord[1] - 1) // grid_scale
            label[i, x_index, y_index, 0] = coord[0] / image_size
            label[i, x_index, y_index, 1] = coord[1] / image_size
            label[i, x_index, y_index, 2] = particle_size[0] / image_size
            label[i, x_index, y_index, 3] = particle_size[1] / image_size
            label[i, x_index, y_index, 4] = 1.0
    '''
    label = np.zeros(
        (len(inputs), grid_size, grid_size, 1),
        dtype = np.float32
    )
    grid_scale = image_size // grid_size
    for i in range(len(inputs)):
        particles = 0
        for coord in inputs[i].content:
            x = int(coord[0] - 1) // grid_scale
            y = int(coord[1] - 1) // grid_scale
            label[i, y, x, 0] = 1.0
            particles += 1
        print(particles, ' particles loaded')
    return label

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

def main(argv):
    del argv
    data_path = FLAGS.data_path
    label_path = FLAGS.label_path
    data_dst = FLAGS.data_dst_path
    label_dst = FLAGS.label_dst_path

    data = mrcHelper.load_mrc_file(data_path)
    label = starHelper.read_all_star(label_path)
    #debug:
    for i in range(len(data)):
        print(data[i].name, '\t', label[i].name)
    # downsampled_data = preprocess(data, False, para1=1024, para2=1024)
    data = downsample(data, False, para1=1024, para2=1024)
    downsampled_label = []
    for i in range(len(label)):
        name = label[i].name
        content = starHelper.downsample_with_size(
            label[i].content,
            (1024 / data[i].header[0], 1024 / data[i].header[1])
        )
        downsampled_label.append(starHelper.StarData(name, content))

    mrcHelper.write_mrc(data, dst=data_dst)
    starHelper.write_star(downsampled_label, dst=label_dst)

def normalize_uint8(data):
    for i in range(np.shape(data)[0]):
        maximum, minimum = np.max(data[i,...]), np.min(data[i,...])
        data[i, ...] = (data[i, ...] - minimum) * 255 / (maximum - minimum) + 0.5
        data[i, ...] = max(data[i, ...], np.zeros_like(data[i, ...]))
        data[i, ...] = min(data[i, ...], 255 * np.ones_like(data[i, ...]))
    data = data.astype(np.float32)    

if __name__ == '__main__':
    #This is a test on eml1/user/ztf
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_path", None, "path of data(mrc, etc.)")
    flags.DEFINE_string("label_path", None, "path of labels(star, etc.)")
    flags.DEFINE_string("data_dst_path", None, "target to store processed data")
    flags.DEFINE_string("label_dst_path", None, "target to store processed labels")

    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("label_path")
    flags.mark_flag_as_required("data_dst_path")
    flags.mark_flag_as_required("label_dst_path")
    app.run(main)
