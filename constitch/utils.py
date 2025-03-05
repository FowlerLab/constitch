import numpy as np
import io
import pickle
import time
import sys
import math


def memory_report():
    import psutil
    return ' '.join([key + '=' + human_readable(val) for key,val in psutil.Process().memory_info()._asdict().items()])

def human_readable(number):
    if number < 1000:
        return str(number)
    scale = min(4, int(math.log10(number)) // 3)
    return '{:.3f}'.format(number / (1000 ** scale)) + ['', 'K', 'M', 'G', 'T'][scale]

def format_time(secs):
    timestr = '{:02}:{:02}'.format(int(secs) // 60 % 60, int(secs) % 60)
    if secs > 3600:
        timestr = '{:02}:'.format(int(secs) // 3600) + timestr
    return timestr

def simple_progress(iterable, total=None):
    if total is None:
        total = len(iterable)

    print_points = [total // 20, total // 4, total // 2, total * 3 // 4]

    if 5 < print_points[0]:
        print_points.insert(0, 5)

    denom_str = str(total)

    start = time.time()
    lasttime = 0
    for i,value in enumerate(iterable):
        yield value
        dtime = time.time() - start
        index = i + 1

        if ((dtime - lasttime >= 2 and (lasttime < 2 or index in print_points))
                or (dtime >= 2 and index == total)):
            est_time = (dtime / index) * total

            padded_index = ('{:' + str(len(denom_str)) + '}').format(index)
            print ("  -- {}/{} {:3}% {} elapsed, {} left, done at {}".format(
                padded_index, denom_str, int(index / total * 100),
                format_time(dtime),
                format_time(est_time - dtime),
                time.strftime("%I:%M %p", time.localtime(start + est_time)),
            ), file=sys.stderr)
            lasttime = dtime
            

def log_env(debug, progress):
    if debug is True:
        debug = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)
    if debug is False:
        debug = lambda *args, **kwargs: None

    if progress is True:
        #import tqdm
        progress = simple_progress
    if progress is False:
        progress = lambda x, **kwargs: x

    return debug, progress

def to_rgb8(image, percent_norm=0.1, colormap=None):
    image = standardize_format(image, 3)
    num_channels, width, height = image.shape

    print (image.min(), image.max())
    minvals = np.percentile(image, percent_norm, axis=(1,2)).reshape(-1,1,1).astype(np.float32)
    maxvals = np.percentile(image, 100 - percent_norm, axis=(1,2)).reshape(-1,1,1).astype(np.float32)
    print (minvals, maxvals)
    image = (image - minvals) / np.maximum(maxvals - minvals, np.finfo(np.float32).tiny)
    print (image.min(), image.max())
    
    if colormap is None:
        if num_channels == 1:
            colormap = np.array([[1],[1],[1]])
        if num_channels >= 2:
            colormap = np.array([
                [1, 0],
                [0, 1],
                [0, 0]])
        if num_channels == 3:
            colormap = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
        if num_channels == 4:
            colormap = np.array([
                [0.66, 0, 0, 0.33],
                [0, 0.66, 0, 0.33],
                [0, 0, 1, 0]])
        if num_channels == 5:
            colormap = np.array([
                [0.5, 0, 0, 0.25, 0.25],
                [0, 0.5, 0, 0.25, 0],
                [0, 0, 0.5, 0, 0.25]])
        if num_channels == 6:
            colormap = np.array([
                [0.5, 0, 0, 0.25, 0.25, 0],
                [0, 0.5, 0, 0.25, 0, 0.25],
                [0, 0, 0.5, 0, 0.25, 0.25]])
        if num_channels == 7:
            colormap = np.array([
                [0.43, 0, 0, 0.21, 0.21, 0, 0.14],
                [0, 0.43, 0, 0.21, 0, 0.21, 0.14],
                [0, 0, 0.43, 0, 0.21, 0.21, 0.14]])

    print (colormap)

    flatimage = image.reshape(num_channels, -1)
    newimage = np.matmul(colormap, flatimage).T
    image = newimage.reshape(width, height, 3)

    print (image.min(), image.max())
    
    image[image<0] = 0
    image[image>1] = 1
    image = (image * 255).astype('uint8')
    
    return image


def standardize_format(image, expected_dims):
    """ Converts an input image into the format
    expected, this being:
    (width, height), (channels, width, height), (cycle, channels, width, height)
    depending on the expected_dims. """

    if len(image.shape) == 3:
        if expected_dims < 3:
            raise ValueError("Expected image with shape: (CHANNELS, WIDTH, HEIGHT), got {}".format(image.shape))
        if image.shape[2] < 32:
            image = image.transpose((2,0,1))

    if len(image.shape) == 4:
        if expected_dims < 4:
            raise ValueError("Expected image with shape: (CYCLE, CHANNELS, WIDTH, HEIGHT), got {}".format(image.shape))
        if image.shape[2] < 32:
            image = image.transpose((0,3,1,2))
    
    return image.reshape((1,) * (expected_dims - len(image.shape)) + image.shape)

def percent_normalize(image, percent=0.1):
    if len(image.shape) == 2:
        image = image.reshape([1] + image.shape)

    mins = np.percentile(image, [percent], axis=(1,2)).reshape((-1,1,1))
    maxes = np.percentile(image, [100-percent], axis=(1,2)).reshape((-1,1,1))

    image = (image - mins) / (maxes - mins)
    image[image<0] = 0
    image[image>1] = 1

    return image

"""
def to_rgb8(image):
    if image.dtype != np.uint8:
        image = percent_normalize(image).astype(np.uint8)

    image = image.transpose((1,2,0))
    return image[:,:,:3]
"""


def save(path, composite, *constraint_sets, save_images=True):
    const_objs = []
    for constraint_set in constraint_sets:
        const_objs.append([const.to_obj() for const in constraint_set._constraint_iter()])

    obj = dict(
        composite = composite.to_obj(save_images=save_images),
        constraint_sets = const_objs,
    )

    if isinstance(path, io.IOBase):
        pickle.dump(obj, path)
    else:
        with open(path, 'wb') as ofile:
            pickle.dump(obj, ofile)

def load(path, **kwargs):
    from .composite import CompositeImage
    from .constraints import Constraint, ConstraintSet
    if isinstance(path, io.IOBase):
        obj = pickle.load(path)
    else:
        obj = pickle.load(open(path, 'rb'))

    composite = CompositeImage.from_obj(obj['composite'], **kwargs)
    constraint_sets = []
    for const_set_obj in obj['constraint_sets']:
        constraint_sets.append(ConstraintSet(Constraint(composite, **const_obj) for const_obj in const_set_obj))

    if len(constraint_sets) == 0:
        return composite
    else:
        return [composite] + constraint_sets

lfsr_table = [
    0x3,
    0x6,
    0xC,
    0x14,
    0x30,
    0x60,
    0xB8,
    0x110,
    0x240,
    0x500,
    0xE08,
    0x1C80,
    0x3802,
    0x6000,
    0xD008,
    0x12000,
    0x20400,
    0x72000,
    0x90000,
    0x140000,
    0x300000,
    0x420000,
    0xE10000,
]

def parity(x):
    res = 0
    while x:
        res ^= x & 1
        x >>= 1
    return res

def lfsr(value, bounds):
    bits = math.ceil(math.log2(bounds))
    taps = lfsr_table[max(0,bits-2)]
    while True:
        newbit = parity(value & taps)
        value = ((value << 1) & ((1 << bits) - 1)) | newbit
        if value <= bounds:
            return value

