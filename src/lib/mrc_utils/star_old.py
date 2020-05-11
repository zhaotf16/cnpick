import os

class StarData():
    def __init__(self, name="", content=[]):
        self.name = name
        self.content = content

def read_star(path):
    coordinates = []
    with open(path) as f:
        x_index, y_index = 0, 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('data_') or line.startswith('loop_'):
                continue
            #_rlnCoordinateX #N means the Nth item stands for x-coordinate
            if line.startswith('_rlnCoordinateX'):
                #relion parameters start from 1, not 0
                #x_index = int(line[len(line)-2]) - 1
                content = line.split()[-1].strip('#')
                x_index = int(content) - 1
                continue
            if line.startswith('_rlnCoordinateY'):
                #y_index = int(line[len(line)-2]) - 1
                content = line.split()[-1].strip('#')
                y_index = int(content) - 1
                continue
            if line.startswith('_rln') or not line.split():
                continue
            content = line.split()
           
            coordinates.append((int(float(content[x_index])), int(float(content[y_index]))))
    return coordinates

def read_all_star(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return None
    if not path.endswith('/'):
        path += '/'
    stars = []
    for file in os.listdir(path):
        if file.endswith('.star'):
            name, _ = os.path.splitext(file)
            print("Loading %s.star ..." % (name))
            content = read_star(path + file)
            stars.append(StarData(name, content))
    stars.sort(key=lambda s: s.name)
    return stars

def downsample_with_size(coordinates, scale):
    #scale is a tuple (scale_x, scale_y)
    downsampled = []
    for i in range(len(coordinates)):
        downsampled.append((
            int(coordinates[i][0] * scale[0]),
            int(coordinates[i][1] * scale[1])
        ))
    return downsampled

def write_star(inputs, dst):
    print('write_star')
    print(len(inputs))
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not dst.endswith('/'):
        dst += '/'
    #for mrc_data in inputs:
    for star_data in inputs:
        print("Writing %s.star ..." % (star_data.name))
        with open(dst+star_data.name+'.star', "w") as f:
            f.write('\ndata_\n')
            f.write('\nloop_\n')
            f.write('_rlnCoordinateX #1\n')
            f.write('_rlnCoordinateY #2\n')
            f.write('_rlnClassNumber #3\n')
            f.write('_rlnAnglePsi #4\n')
            f.write('_rlnAutopickFigureOfMerit  #5\n')
            for item in star_data.content:
                f.write("%d.0\t%d.0\t-999\t-999.0\t-999.0\n"%(item[0], item[1]))
            f.write('\n')
                 

if __name__ == '__main__':
    path = '../dataset/EMPIAR-10025/rawdata/label_for_training/'
    stars = read_all_star(path)
    for star in stars:
        star.content = downsample_with_size(star.content, (1024/7676, 1024/7420))
    print(stars[0].content)
    #content = read_star(path