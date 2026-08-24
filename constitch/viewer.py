import os
import time
import re
import json
import io
import flask
import numpy as np
import tifffile
import PIL

from . import visualization, utils

app = flask.Flask(__name__)

@app.route("/static/bundle.js")
def bundle():
    return flask.send_file(open('viewer/build/bundle.js', 'rb'), mimetype='application/javascript')

@app.route("/")
@app.route("/<path:path>/")
def dir_listing(path=None):
    path = path or './'
    names = [name + '/' if os.path.isdir(name) else name for name in os.listdir(path)]
    links = ['<li><a href="{}">{}</a></li>'.format(name, name) for name in names]
    return '<h1>{}/</h1><ul>{}</ul>'.format(path, '\n'.join(links))


def isiterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False

def isint(obj):
    try:
        return type(obj) == int or np.issubdtype(obj, np.integer)
    except:
        return False


def tiffinfo(path):
    with tifffile.TiffFile(path) as ifile:
        shape = ifile.pages[0].shape
        dtype = ifile.pages[0].dtype

        if ifile.shaped_metadata and 'shape' in ifile.shaped_metadata[0]:
            shape = tuple(ifile.shaped_metadata[0]['shape'])

        elif len(ifile.pages) > 1 and all(ifile.pages[0].shape == page.shape and ifile.pages[0].dtype == page.dtype for page in ifile.pages):
            shape = (len(ifile.pages),) + shape

    return shape, dtype


class ImageInfo:
    def __init__(self, path, shape=None, dtype=None, indices=None):
        self.path = path

        if indices is not None:
            self._shape = shape
            self.dtype = dtype
            self.indices = indices
            return

        if '{' in path:
            fileseq = utils.open_sequence(path)
            tileshape, tiledtype = tiffinfo(fileseq[0])
            self._shape = fileseq.shape + tileshape
            self.dtype = tiledtype
        else:
            self._shape, self.dtype = tiffinfo(path)

        self.indices = [slice(None) for i in range(len(self._shape))]
        self.indices = [range(dim) for dim in self._shape]

    @property
    def shape(self):
        return tuple(len(idx) for idx in self.indices if type(idx) == range)

    @property
    def ndim(self):
        return sum(type(idx) == range for idx in self.indices)

    def __repr__(self):
        return '{}(path={}, shape={}, dtype={})'.format(self.__class__.__name__, self.path, self.shape, self.dtype)

    def __str__(self):
        parts = []
        for idx, dim in zip(self.indices, self._shape):
            if isinstance(idx, np.ndarray):
                idx = idx.tolist()

            if type(idx) != range:
                parts.append(str(idx).replace(', ', ','))
                continue

            part = '{}:{}{}'.format(
                    idx.start if idx.start != 0 else '',
                    idx.stop if idx.stop != dim else '',
                    ':{}'.format(idx.step) if idx.step != 1 else '')
            parts.append(part)

        while len(parts) > 1 and parts[-1] == ':':
            parts.pop(len(parts)-1)

        return '{}[{}]'.format(self.path, ','.join(parts))

    REGEX = r'\[((\[[0-9,\-]*\]|[0-9:\-,]+)+)\]$'
    @classmethod
    def fromstr(cls, path, section=None):
        if section is None:
            match = re.search(cls.REGEX, path)
            assert match is not None, "Path does not have valid slice spec"
            path = path[:match.start()]
            section = match.group(1)
        #print (path, section)

        info = cls(path)
        try:
            info = eval('info[{}]'.format(section))
        except:
            assert False, "Path does not have valid slice spec"

        return info

    def __getitem__(self, indices):
        if type(indices) != tuple:
            indices = (indices,)

        i, j = 0, 0
        newindices = self.indices.copy()

        while i < len(indices) and j < len(self.indices):
            if type(self.indices[j]) == int:
                j += 1
                continue

            if indices[i] is Ellipsis:
                if len(indices) - i <= len(self.indices) - j:
                    j += 1
                    continue
                else:
                    i += 1
                    continue

            if type(self.indices[j]) == range:
                if isiterable(indices[i]):
                    newindices[j] = [self.indices[j][k] for k in indices[i]]
                else:
                    newindices[j] = self.indices[j][indices[i]]

            elif isiterable(self.indices[j]):
                newindices[j] = np.asarray(self.indices[j])[indices[i]]


            i += 1
            j += 1

        return ImageInfo(self.path, self.shape, self.dtype, newindices)

    @property
    def slice(self):
        return tuple(slice(idx.start, idx.stop, idx.step) if type(idx) == range else idx for idx in self.indices)

    def open(self):
        if '{' in self.path:
            import zarr
            fileseq = utils.open_sequence(self.path)
            image = zarr.open(fileseq.aszarr())
        else:
            try:
                image = tifffile.memmap(self.path)
            except:
                image = tifffile.imread(self.path)

        image = image[self.slice]
        return image


def open_image_info(path):
    with tifffile.TiffFile(path) as ifile:
        info = ImageInfo()
        info.shape = ifile.pages[0].shape
        consistent_pages = all(ifile.pages[0].shape == page.shape and ifile.pages[0].dtype == page.dtype for page in ifile.pages)
        if len(ifile.pages) == 1 or not consistent_pages:
            return 

@app.route("/<path:path>.tif")
@app.route("/<path:path>.tif[<section>]")
def viewer(path, section=None):
    path = path + '.tif'

    if not '{' in path and not os.path.exists(path):
        return flask.abort(404)

    #image = tifffile.memmap(path)

    #if section is not None and section != ':':
        #parsed_slice = map(parse_slice, section.split(','))
        #image = image[tuple(parsed_slice)]

    if section is not None:
        image = ImageInfo.fromstr(path, section=section)
    else:
        image = ImageInfo(path)

    print (image.shape, image.dtype)

    channel_axis = None
    composite_axes = 0

    if len(image.shape) > 2:
        composite_axes = len(image.shape) - 2
        if image.shape[-3] <= 7:
            channel_axis = len(image.shape) - 3
            composite_axes -= 1

    #print ('image', image.shape, image.dtype)
    #print ('composite, channel', composite_axes, channel_axis)
    #print (image.indices)

    downscale = 11

    if composite_axes > 0:

        downscaled_image = image[...,::downscale,::downscale]

        if composite_axes == 2:
            poses = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
            poses = np.stack(poses, axis=-1).reshape(-1, 2)
            indices = poses.copy()
        else:
            num_images = np.prod(image.shape[:composite_axes])
            dims = np.ceil(np.sqrt(num_images))
            poses = np.meshgrid(np.arange(dims), np.arange(dims), indexing='ij')
            poses = np.stack(poses, axis=-1).reshape(-1, 2)[:num_images]
            indices = np.meshgrid(*[np.arange(dim) for dim in image.shape[:composite_axes]], indexing='ij')
            indices = np.stack(indices, axis=-1).reshape(-1, composite_axes)[:num_images]

        padding = 100 if composite_axes == 1 else 0
        downscaled_poses = poses.copy()
        downscaled_poses[:,0] *= downscaled_image.shape[-2] + padding // downscale
        downscaled_poses[:,1] *= downscaled_image.shape[-1] + padding // downscale
        poses[:,0] *= image.shape[-2] + padding
        poses[:,1] *= image.shape[-1] + padding

        #print (poses)
        #print (indices)

        #new_section = section + ',' if section is not None else ''
        #urls = ['/{}[{}].png'.format(path, new_section + ','.join(map(str, ids)) + ',:,::50,::50') for ids in indices]
        #boxes = [[int(pos[0]), int(pos[1]), image.shape[-2], image.shape[-1]] for pos in poses]

        images = []
        for pos, ids in zip(poses, indices):
            box = [int(pos[0]), int(pos[1]), image.shape[-2], image.shape[-1]]
            #url_downscale = '/{}[{}].png'.format(path, new_section + ','.join(map(str, ids)) + ',:,::50,::50')
            #url = '/{}[{}].png'.format(path, new_section + ','.join(map(str, ids)))
            url_downscale = '/' + str(downscaled_image[*ids]) + '.png'
            url = '/' + str(image[*ids]) + '.png'

            newimage = {
                'box': box,
                'dims': (np.array(image.shape[-2:]) // 50).tolist(),#list(image.shape[-2:]),
                'shape': [image.shape[channel_axis]] + (np.array(image.shape[-2:]) // 50).tolist(),#list(image.shape[composite_axes:]),
                'numChannels': 1 if channel_axis is None else image.shape[channel_axis],
                'url': url_downscale,
                'images': [{
                    'box': box,
                    'dims': list(image.shape[-2:]),
                    'shape': list(image.shape[composite_axes:]),
                    'numChannels': 1 if channel_axis is None else image.shape[channel_axis],
                    'url': url,
                }],
            }
            images.append(newimage['images'][0])

        #downscaled_padding = padding // downscale
        #downscaled_dims = (np.max(poses // downscale, axis=0) + np.array(image.shape[-2:]) // downscale).astype(int)
        full_shape = int(poses[:,0].max() + image.shape[-2]), int(poses[:,1].max() + image.shape[-1])
        full_shape_downscaled = int(downscaled_poses[:,0].max() + downscaled_image.shape[-2]), int(downscaled_poses[:,1].max() + downscaled_image.shape[-1])
        #print ('full_shape viewer', full_shape)
        #if channel_axis is not None:
            #full_shape = (image.shape[channel_axis],) + full_shape
        #print (full_shape)
        #print (image[...,::downscale,::downscale].indices)
        #print (image[...,::downscale,::downscale])

        image = {
            'box': [0, 0, int(poses[:,0].max() + image.shape[-2]), int(poses[:,1].max() + image.shape[-1])],
            'dims': [int(full_shape_downscaled[0]), int(full_shape_downscaled[1])],
            'shape': [image.shape[channel_axis], int(full_shape_downscaled[0]), int(full_shape_downscaled[1])],
            #'dims': [int(poses[:,0].max() + image.shape[-2]) // 50, int(poses[:,1].max() + image.shape[-1]) // 50],
            #'shape': [image.shape[channel_axis], int(poses[:,0].max() + image.shape[-2]) // 50, int(poses[:,1].max() + image.shape[-1]) // 50],
            'numChannels': 1 if channel_axis is None else image.shape[channel_axis],
            #'url': '/{}[{}].png'.format(path, ':,:,:,::50,::50'),
            'url': '/' + str(downscaled_image) + '.png',
            'images': images,
            'loading': 'auto',
        }

    else:
        image = {
            'box': [0, 0, image.shape[-2], image.shape[-1]],
            'dims': list(image.shape[-2:]),
            'shape': list(image.shape[composite_axes:]),
            'numChannels': 1 if channel_axis is None else image.shape[channel_axis],
            #'url': '/{}[{}].png'.format(path, section or ':'),
            'url': '/' + str(image) + '.png',
            'loading': 'auto',
        }

    response = io.StringIO()

    htmlpage = open('viewer/build/index.html').read()
    #script = open('viewer/build/bundle.js').read()
    #htmlpage = htmlpage.replace('src="bundle.js">', '>' + script)
    htmlpage = htmlpage.replace('src="bundle.js"', 'src="/static/bundle.js"')

    #info = str(info).replace("'", '"')
    info = '{}'
    #image = str(image).replace("'", '"')
    infoindex = htmlpage.index('"{{info}}"')
    imageindex = htmlpage.index('"{{image}}"')

    response.write(htmlpage[:infoindex])
    response.write(info)
    response.write(htmlpage[infoindex+10:imageindex])
    json.dump(image, response)
    response.write(htmlpage[imageindex+11:])

    #htmlpage = htmlpage.replace('"{{info}}"', info).replace('"{{image}}"', image)
    #response.seek(0)
    #return flask.send_file(response, mimetype='text/html')

    return response.getvalue()


@app.route("/<path:path>.tif[<section>].<fmt>")
def raw_image(path, section, fmt):
    #image = None
    times = []
    times.append(time.time())
    path = path + '.tif'

    if not '{' in path and not os.path.exists(path):
        return flask.abort(404)

    imageinfo = ImageInfo.fromstr(path, section=section)
    #print (imageinfo, imageinfo.indices, str(imageinfo))
    times.append(time.time())
    image = imageinfo.open()
    times.append(time.time())

    #if section != ':':
        #try:
            #image = tifffile.memmap(path)
        #except:
            #pass

    #if image is None:
        #image = tifffile.imread(path)

    #if section != ':':
        #parsed_slice = tuple(map(parse_slice, section.split(',')))
        #image = image[parsed_slice]

    channel_axis = None
    composite_axes = 0

    if len(image.shape) > 2:
        composite_axes = len(image.shape) - 2
        if image.shape[-3] <= 7:
            channel_axis = len(image.shape) - 3
            composite_axes -= 1


    if composite_axes > 0:

        if composite_axes == 2:
            poses = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
            poses = np.stack(poses, axis=-1).reshape(-1, 2)
            indices = poses.copy()
        else:
            num_images = np.prod(image.shape[:composite_axes])
            dims = np.ceil(np.sqrt(num_images))
            poses = np.meshgrid(np.arange(dims), np.arange(dims), indexing='ij')
            poses = np.stack(poses, axis=-1).reshape(-1, 2)[:num_images]
            indices = np.meshgrid(*[np.arange(dim) for dim in image.shape[:composite_axes]], indexing='ij')
            indices = np.stack(indices, axis=-1).reshape(-1, composite_axes)[:num_images]

        padding = 100 if composite_axes == 1 else 0
        #padding //= max(parsed_slice[-2].step, parsed_slice[-2].step)
        padding //= max(imageinfo.indices[-2].step, imageinfo.indices[-1].step)

        poses[:,0] *= image.shape[-2] + padding
        poses[:,1] *= image.shape[-1] + padding

        full_shape = int(poses[:,0].max() + image.shape[-2]), int(poses[:,1].max() + image.shape[-1])
        #print ('full_shape rawimage', full_shape)
        if channel_axis is not None:
            full_shape = (image.shape[channel_axis],) + full_shape
        full_image = np.full(full_shape, np.iinfo(image.dtype).max, image.dtype)

        for ids, pos in zip(indices, poses):
            x1, y1, x2, y2 = int(pos[0]), int(pos[1]), int(pos[0] + image.shape[-2]), int(pos[1] + image.shape[-1])
            curimage = image[tuple(ids.astype(int))]
            if channel_axis is not None:
                full_image[:,x1:x2,y1:y2] = curimage
            else:
                full_image[x1:x2,y1:y2] = curimage

        image = full_image
        #print (image.shape)

    #print (image.shape, image.dtype)
    encoded = visualization.encode_image(image)
    times.append(time.time())

    pilimage = PIL.Image.fromarray(encoded)
    times.append(time.time())
    image_data = io.BytesIO()
    pilimage.save(image_data, format=fmt)
    times.append(time.time())
    image_data.seek(0)

    print ('times', [end - begin for begin, end in zip(times, times[1:])])

    return flask.send_file(image_data, mimetype='image/' + fmt)

def parse_slice(section):
    parts = [None if part == '' else int(part) for part in section.split(':')]
    return parts[0] if len(parts) == 1 else slice(*parts)


