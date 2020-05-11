import os
import box
import coord
import mrc
import star

import numpy as np

from absl import app
from absl import flags


def downsample(inputs, use_factor=False, para1=None, para2=None):
    #This method executes a downsampling on the mrc
    #Inputs is a list of MrcData
    #If use_factor is True, para1 represents factor and para2 represents shape
    #Else para1 and para2 represents the target size(y, x)
    
    for i in range(len(inputs)):
        if use_factor:
            print("Processing %s ..." % ( inputs[i].name))
            inputs[i].data = mrc.downsample_with_factor(
                inputs[i].data,
                factor=para1,
                shape=para2
            )
        else:
            print("Prcocessing %s ..." % (inputs[i].name))
            inputs[i].data = mrc.downsample_with_size(
                inputs[i].data,
                size1=para1,
                size2=para2
            )
    return inputs

def label_downsample(data, label, t, w, h):
    downsampled_label = []
    if t == 'star':
        for i in range(len(label)):
            name = label[i].name
            content = star.downsample_with_size(
                label[i].content,
                (w / data[i].header[0], h / data[i].header[1])
            )
            downsampled_label.append(star.StarData(name, content))
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
        return star.read_all_star(label_path)
    elif label_type == 'coord':
        return coord.read_all_coord(label_path)
    elif label_type == 'box':
        print('reading box')
        return box.read_all_box(label_path)
    else:
        print('A valid type is required: star | coord | box')
        return []

def load_and_downsample(path):
    with open(path, "rb") as f:
        content = f.read()
    data, header, _ = mrc.parse(content=content)
    name = path.split('/')[-1].split('.')[0]
    #averge frame

    if header[2] > 1:
        avg_mrc = np.zeros_like(data, data[0,...])
        for j in range(header[2]):
            avg_mrc += data[j, ...]
        avg_mrc /= header[2]
        data = avg_mrc
    return mrc.MrcData(name, data, header)

def process(opt):
    path = opt.data
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return

    mrc_data = []
    for file in os.listdir(path):
        if file.endswith('.mrc'):
            print("Loading %s ..." % (file))
            data = load_and_downsample(os.path.join(path, file))
            # TODO: load and process label according to STAR or EMAN
            mrc_data.append(data)
    mrc_data.sort(key=lambda m: m.name)
    
    label = read_label(opt.data, opt.label_type)
    #debug:
    for k in range(len(mrc_data)):
        print(mrc_data[k].name, '\t', label[k].name)

    downsampled_label = label_downsample(
        mrc_data, label, 
        opt.label_type, 
        opt.target_size, opt.target_size
    )
    print(len(downsampled_label))
    image_path = os.join(opt.data_dir, opt.exp_id, 'images')
    if not os.exist(image_path):
        os.mkdir(image_path)
    for m in mrc_data:
        mrc.save_image(m.data, os.join(image_path, m.name), f='png', verbose=True)
    num_train = 40
    num_val = 10
    num_test = 10
    train = downsampled_label[0:num_train]
    val = downsampled_label[num_train:num_train+num_val]
    test = downsampled_label[num_train+num_val:num_train+num_val+num_test]
    star.star2coco(train, image_path)
    #mrc.write_mrc(mrc_data, dst=data_dst)
    #write_label(downsampled_label, label_type)
    #print('writing label...')
    #star.write_star(downsampled_label, dst=label_dst)
    return get_mean_and_var(image_path)
def get_mean_and_var(filepath):
    dir = os.listdir(filepath)
    
    r, g, b = 0, 0, 0
    for idx in range(len(dir)):
        filename = dir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        r = r + np.sum(img[:, :, 0])
        g = g + np.sum(img[:, :, 1])
        b = b + np.sum(img[:, :, 2])
    
    pixels = len(dir) * 1024  * 1024 
    r_mean = r / pixels
    g_mean = g / pixels
    b_mean = b / pixels

    r, g, b = 0, 0, 0
    for i in range(len(dir)):
        filename = dir[i]
        img = imread(os.path.join(filepath, filename)) / 255.0
        r = r + np.sum((img[:, :, 0] - r_mean) ** 2)
        g = g + np.sum((img[:, :, 1] - g_mean) ** 2)
        b = b + np.sum((img[:, :, 2] - b_mean) ** 2)
    
    r_var = np.sqrt(r / pixels)
    g_var = np.sqrt(g / pixels)
    b_var = np.sqrt(b / pixels)
    print("r_mean is %f, g_mean is %f, b_mean is %f" % (r_mean, g_mean, b_mean))
    print("r_var is %f, g_var is %f, b_var is %f" % (r_var, g_var, b_var))
    return [r_mean, g_mean, g_mean], [r_var, g_var, b_var]
'''
def main(argv):
    del argv
    data_path = FLAGS.data_path
    label_path = FLAGS.label_path
    data_dst = FLAGS.data_dst_path
    label_dst = FLAGS.label_dst_path
    target_size = FLAGS.target_size
    label_type = FLAGS.label_type
    
    mrc_data = mrc.load_all_mrc(data_path)
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
    mrc_data = downsample(mrc_data, False, para1=target_size, para2=target_size)
    downsampled_label = label_downsample(
        mrc_data, label, 
        label_type, 
        target_size, target_size
    )
    print(len(downsampled_label))

    mrc.write_mrc(mrc_data, dst=data_dst)
    #write_label(downsampled_label, label_type)
    print('writing label...')
    star.write_star(downsampled_label, dst=label_dst)  

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
