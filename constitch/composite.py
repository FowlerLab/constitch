import sys
import random
import time
import collections
import pickle
import math
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters
import sklearn.linear_model
import concurrent.futures
import sklearn.mixture
import imageio.v3 as iio
import warnings

from .alignment import calculate_offset, score_offset
from .stage_model import SimpleOffsetModel, GlobalStageModel
from .constraints import Constraint, ConstraintType, ConstraintSet, ImplicitConstraintDict
from . import merging, alignment, solving
from . import utils



@dataclasses.dataclass
class BBox:
    pos1: np.ndarray
    pos2: np.ndarray

    def collides(self, otherbox):
        contains = (((self.pos1 <= otherbox.pos1) & (self.pos2 > otherbox.pos1))
                  | ((self.pos1 >= otherbox.pos1) & (self.pos1 < otherbox.pos2)))
        collides = (((self.pos1 <= otherbox.pos1) & (self.pos2 >= otherbox.pos1))
                  | ((self.pos1 >= otherbox.pos1) & (self.pos1 <= otherbox.pos2)))
        result = np.all(collides) and np.sum(contains) >= contains.shape[0] - 1
        return result

    def overlaps(self, otherbox):
        contains = (((self.pos1 <= otherbox.pos1) & (self.pos2 > otherbox.pos1))
                  | ((self.pos1 >= otherbox.pos1) & (self.pos1 < otherbox.pos2)))
        return np.all(contains)

    def contains(self, otherbox):
        if type(otherbox) == BBox:
            return np.all((self.pos1 <= otherbox.pos1) & (self.pos2 >= otherbox.pos2))
        else:
            return np.all((self.pos1 <= otherbox) & (self.pos2 >= otherbox))

    @property
    def size(self):
        return self.pos2 - self.pos1

    @property
    def center(self):
        return (self.pos1 + self.pos2) / 2

    def as2d(self):
        return BBox(self.pos1[:2], self.pos2[:2])



class BBoxList:
    def __init__(self, pos1=None, pos2=None):
        self.pos1 = pos1
        self.pos2 = pos2
        self.boxes = []
        if pos2 is not None:
            for i in range(len(self.pos1)):
                self.boxes.append(BBox(self.pos1[i], self.pos2[i]))

    def append(self, box):
        self.boxes.append(box)
        if self.pos1 is None:
            self.pos1 = box.pos1.reshape(1,-1)
            self.pos2 = box.pos2.reshape(1,-1)
        else:
            self.pos1 = np.concatenate([self.pos1, box.pos1.reshape(1,-1)], axis=0)
            self.pos2 = np.concatenate([self.pos2, box.pos2.reshape(1,-1)], axis=0)
            for i in range(len(self.boxes)):
                self.boxes[i].pos1 = self.pos1[i]
                self.boxes[i].pos2 = self.pos2[i]

    def resize(self, n_dims):
        if self.pos1.shape[1] < n_dims:
            padding = n_dims - self.pos1.shape[1]
            self.pos1 = np.pad(self.pos1, [(0, 0), (0, padding)])
            self.pos2 = np.pad(self.pos2, [(0, 0), (0, padding)])
            for i in range(len(self.boxes)):
                self.boxes[i].pos1 = self.pos1[i]
                self.boxes[i].pos2 = self.pos2[i]

    def __getitem__(self, index):
        return self.boxes[index]

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def size(self):
        return self.pos2 - self.pos1

    def center(self):
        return (self.pos1 + self.pos2) / 2

    def __repr__(self):
        return "BBoxList(pos1={}, pos2={})".format(repr(self.pos1), repr(self.pos2))

    def __str__(self):
        return "BBoxList(pos1={}, pos2={})".format(self.pos1, self.pos2)

class SequentialExecutorFuture:
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def result(self):
        return self.func(*self.args, **self.kwargs)


class SequentialExecutor(concurrent.futures.Executor):
    def submit(self, func, *args, **kwargs):
        return SequentialExecutorFuture(func, args, kwargs)




class CompositeConstraintSet(ConstraintSet):
    def __init__(self, composite, pair_func):
        super().__init__()
        self.constraints = ImplicitConstraintDict(composite, pair_func)

    def add(self, obj):
        raise "Cannot add constraints to composite, composite.constraints is read-only"

    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)



class CompositeImage:
    """
    This class encapsulates the whole stitching process, the smallest example of stitching is
    shown below:

        composite = fisseq.stitching.CompositeImage()
        composite.add_images(images)
        composite.calc_constraints()
        composite.filter_constraints()
        composite.solve_constraints(filter_outliers=True)
        full_image = composite.stitch_images()

    This class is meant to be adaptable to many different stitching use cases, and each step
    can be customized and configured. The general steps for the stitching of a group images are as follows:


    Creating the composite

    To begin we have to instantiate the CompositeImage class.
    The full method signature can be found at
    __init__() but some important parameters are described below:

    The executor is what the composite uses to perform intensive computation
    tasks, namely calculating the alignment of all the images. If provided
    it should be a concurrent.futures.Executor object, for example
    concurrent.futures.ThreadPoolExecutor. Importantly, concurrent.futures.ProcessPoolExecutor
    does not work very well as the images need to be passed to the executor
    and in the case of ProcessPoolExecutor this means they need to be pickled
    and unpickled to get to the other process. ThreadPoolExecutor doesn't need
    this as the threads can share memory, but it doesn't take full advantage of
    multithreading as the python GIL prevents python code from running in parallel.
    Luckily most of the intensive computation happens in numpy functions which don't
    hold the GIL, so ThreadPoolExecutor is usually the best choice.

    The arguments debug and process define how logging should happen with the composite.
    If debug is True logging messages summarizing the results of different operations
    will be printed out to stderr. Setting it to False will disable these messages.
    If process is True a progress bar will be printed out during long running steps.
    The default progress bar is a simple ascii bar that works whether the output is
    a tty or a file, but if you want you can pass a replacement in instead of setting
    it to True, such as tqdm.

    An example of setting up the composite would be something similar to this:

    import fisseq
    import concurrent.futures
    import tqdm

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        composite = fisseq.stitching.CompositeImage(executor=executor, debug=True, progress=tqdm.tqdm)


    Setting the aligner

    The aligner of the composite is a class that encapsulates the algorithm used to align two images onto
    each other. The base class Aligner and the fisseq.alignment module have more information on the specifics but basically
    an aligner has a function that takes two images and returns the offset from one image to another
    that has the best overlap. This is normally measured using the normalized cross correlation
    of the overlapping regions, using the function alignment.ncc, but different aligner classes
    can use different ways to find and score overlapping regions.

    The main aligner provided is the FFTAligner, which uses the phase cross correlation algorithm
    to find the best overlapping region. By default this one is used, however if you want to modify
    the parameters you can set the aligner to a new instance of it. For example:
        composite.set_aligner(fisseq.stitching.FFTAligner(num_peaks=5, precalculate_fft=False))

    The other aligner provided is the FeatureAligner, which uses feature based methods to align the images.
    This is quite a bit faster than the FFTAligner, but it is not as accurate or reliable, so be aware that
    results might not be as good.

    As with many of the classes used for stitching, the aligner is meant to be customized and you are
    encouraged to subclass the Aligner class and make your own method. The two methods needed for an
    Aligner class are described in its docs.


    Adding the images

    Once the composite is set up we can add the images, and this is done through the add_images()
    method. There are a couple ways of adding images, depending on how much information you have on the images:

    First of all, you can just add the images with no positions, meaning they will all default to being at 0,0.
    This will work out, as when you calculate constraints between images it will calculate constraints for all
    possible images and filter out constraints that have no overlap. However the number of constraints that have
    to be calculated grows exponentially with the number of images, and if you have positional information on your
    images it is best to pass that in to help with the alignment. If you would like to use this method but are running
    into computational limits, the section on pruning constraints below can be helpful.
        composite.add_images(images)

    If your images are taken on a grid you can pass in their positions as grid positions, by setting the scale
    parameter to 'tile'. For example:
        positions=[(0,0), (0,1), (1,0), (1,1)]
        composite.add_images(images, positions=positions, scale='tile')
    Now when constraints are calculated only nearby images will be checked, speeding up computation a lot.

    If your images are not on a grid or you have the exact position they were taken in, you can also specify
    positions in pixels instead of grid positions, to do this simply set the scale parameter to 'pixel'
    and the positions passed in will be interpreted as pixel coordinates.

    When specifying positions, you can also specify more than two dimensions. The first two are the x and y
    dimensions of the images, but a z dimension can be added if you are doing 3 dimensional stitching or in our case
    if you are doing fisseq and want to make sure all the cycles line up perfectly. In the case of fisseq,
    you can add the cycle index as the z coordinate for the image.


    Calculating constraints

    Once images have been added we need to calculate the constraints between overlapping images. This is done
    with the calc_constraints() function. For usual usage you can run it with no parameters,
    but if you want to be specific about which images are overlapping it takes in an argument called pairs,
    which should be a sequence of tuples, each being a pair of image indices that should be checked for overlap.
    If pairs is not specified it defaults to find_unconstrained_pairs(), which finds all image
    pairs that have overlap but don't have a constraint between them. The two lines below both work to calculate
    constraints:
        composite.calc_constraints()
        composite.calc_constraints(composite.find_unconstrained_pairs())

    One important parameter is precalculate, if True the results of the precalculation step will be cached
    for each image, saving on computation time but increasing memory usage.

    The find_unconstrained_pairs function also has some parameters that affect how it looks for overlap,
    the most useful one being overlap_threshold. This is a value in pixels that specifies how far images
    have to overlap to be considered overlapping. This can also be negative, meaning images that are within
    a certain pixel distance to touching will be considered overlapping. This is useful if you want to make sure
    that you find all overlapping images, even if your original positions are not very accurate. For example,
    the line below will expand the search by 2000 pixels and hopefully find more overlap
        composite.calc_constraints(composite.find_unconstrained_pairs(overlap_threshold=(-2000, -2000)))

    Another use would be for fisseq, where you want to calculate constraints across all cycles and not just
    adjacent cycles. As described before fisseq cycles can be represented using the z axis, so we can expand
    the search in that z axis and calculate constraints between all cycles:
        composite.calc_constraints(composite.find_unconstrained_pairs(overlap_threshold=(0, 0, -12))


    Filtering constraints

    After calculating constraints for all overlapping images, we have to filter out constraints that aren't accurate.
    Unfortunately the alignment algorithms are not perfect and sometimes they miss overlap, so we need to filter out
    constraints that have low scores or don't fit in with the other constraints. There are a couple steps of this,
    the first being filtering based on scores:
        composite.filter_constraints()
    This method, filter_constraints(), looks at the scores of all constraints and calculates a random
    set of bad constraints to find a good score threshold to eliminate any constraints that are not accurate. This
    is necessary because the scores returned by the alignment algorithms depend on the features present in the images,
    and using a fixed threshold would not work for all image types.

    An optional secondary filtering step that can be used is the solve_constraints() method. Normally
    we call this method when we want to solve for the global position of the images, but we can also call this method
    to attempt to solve the constraints, and remove any that don't seem to fit with the other constraints present.
    To do this we pass in the arguments as shown:
        composite.solve_constraints(apply_positions=False, filter_outliers=True)


    Estimating missing constraints

    After filtering out all the inaccurate constraints, we may be left with images that don't have any constraints
    to other images. This is especially common when there are not enough features in some areas of the image to
    successfully line up the images. To account for this, we can use a stage model to learn the movement of the microscope,
    and estimate where images that weren't able to be aligned should be. Importantly this only will work if the movement
    of your microscope is predictable, ie it scans in a grid pattern, and you have the positions or grid positions for
    images. To estimate these constraints we use the methods estimate_stage_model() and model_constraints().
        composite.estimate_stage_model()
        composite.model_constraints()
    More information can be found in the docs for each function, and about the different possible stage models at fisseq.stage_model


    Solving constraints

    After all the constraints are ready, we can solve them all globally to get the positions
    for each image. This is done by representing the whole composite as an overconstrained linear system,
    where each constraint is two equations linking the x and y coordinates of two images.
    This is all done in the solve_constraints() function, and calling it will globally solve
    all the constraints and apply the new positions to all images.
        composite.solve_constraints()
    Sometimes the solver can have trouble solving the constraints, especially if inaccurate constraints are still
    present when solving. If this happens and it isn't able to find a good solution, it will throw an error.
    A simple way to try to solve this is by passing the filter_outliers=True argument, which will tell it to try
    to remove constraints that do not align with others around them. However this doesn't always work, if the error
    persists a larger change may be needed to help. See the troubleshooting section for more information.


    Stitching images

    The final step left is to stitch the images together into a single composite image. This is done
    with the stitch_images() function.
        full_image = composite.stitch_images()

    The method that this function merges the images together can be configured by passing a Merger instance,
    by default it will use MeanMerger. More information on mergers can be found in the docs of the Merger class
    and the fisseq.stitching.merging module.
    """

    def __init__(self, images=None, positions=None, boxes=None, scale='pixel', channel_axis=None,
            grid_size=None, tile_shape=None, overlap=0.1,
            aligner=None, precalculate=False, debug=True, progress=False, executor=None):
        self.images = []
        self.boxes = BBoxList()
        self.constraints = CompositeConstraintSet(self, self.pair_func)
        self.scale = 1
        self.set_logging(debug, progress)
        self.set_executor(executor)
        self.set_aligner(aligner)
        self.multichannel = False

        self.precalculate = precalculate

        if images is not None:
            if grid_size is not None or tile_shape is not None:
                return self.add_split_image(images, grid_size=grid_size, tile_shape=tile_shape, overlap=overlap, channel_axis=channel_axis)
            return self.add_images(images, positions, boxes=boxes, scale=scale, channel_axis=channel_axis)

    def pair_func(self):
        for i in range(len(self.images)):
            for j in range(i+1, len(self.images)):
                yield i, j

    def set_executor(self, executor):
        self.executor = executor or SequentialExecutor()

    def set_aligner(self, aligner, rescore_constraints=False):
        self.aligner = aligner or alignment.FFTAligner()

    def set_logging(self, debug=True, progress=False):
        self.debug, self.progress = utils.log_env(debug, progress)

    def print_mem_usage(self):
        mem_images = sum((image.nbytes if type(image) == np.ndarray else 0) for image in self.images)
        self.debug("Using {} bytes ({}) for images".format(mem_images, utils.human_readable(mem_images)))
        self.debug("Total: {} ({})".format(mem_images, utils.human_readable(mem_images)))

    def to_obj(self, save_images=True):
        obj = dict(
            boxes = (self.boxes.pos1, self.boxes.pos2),
            #constraints = self.constraints,
            scale = self.scale,
            debug = bool(self.debug),
            progress = bool(self.progress),
        )
        if save_images:
            obj['images'] = self.images
        else:
            obj['images'] = [None] * len(self.images)

        return obj

    @classmethod
    def from_obj(cls, obj, **kwargs):
        params = dict(
            debug = obj.pop('debug'),
            progress = obj.pop('progress'),
        )
        params.update(kwargs)

        composite = cls(**params)
        pos1, pos2 = obj.pop('boxes')
        composite.boxes = BBoxList(pos1, pos2)
        composite.__dict__.update(obj)
        return composite

    def add_images(self, images, positions=None, boxes=None, scale='pixel', channel_axis=None, imagescale=1):
        """ Adds images to the composite

        Args:
            images (np.ndarray shape (N, W, H) or list of N np.ndarrays shape (W, H) or list of strings):
                The images that will be stitched together. Can pass a list of
                paths that will be opened by imageio.v3.imread when needed.
                Passing paths will require less memory as images are not stored,
                but will increase computation time.

            positions (np.ndarray shape (N, D) ):
                Specifies the extimated positions of each image. The approx values are
                used to decide which images are overlapping. These values are interpreted
                using the scale argument, default they are pixel values.

            boxes (sequence of BBox):
                An alternative to specifying the positions, the full bounding boxes of every image can also
                be passed in. The units of the boxes are interpreted the same as image positions,
                with the scale argument deciding their relation to the scale of pixels.

            scale ('pixel', 'tile', float, or sequence):
                The scale argument is used to interpret the position values given.
                'pixel' means the values are pixel values, equivalent to putting 1.
                'tile' means the values are indices in a tile grid, eg a unit of 1 is
                the width of an image.
                a float value means the position values are a units where one unit is
                the given number of pixels.
                If a sequence is given, each element can be any of the previous values,
                which are applied to each axis.
        """
        if positions is None and boxes is None:
            positions = [(0,0)] * len(images)
        #assert positions is not None or boxes is not None, "Must specify positions or boxes"
        if positions is None:
            n_dims = len(boxes[0].pos1)
        else:
            positions = np.asarray(positions)
            n_dims = positions.shape[1]
        #assert len(self.imageshape(images[0])) == 2, "Only 2d images are supported"

        if channel_axis is not None:
            self.multichannel = True

        self.n_dims = n_dims

        if scale == 'pixel':
            scale = 1
        if scale == 'tile':
            #assert type(images) == np.ndarray, ("Using scale='tile' is only supported with"
            #        " images as a np.ndarray, not a list of ndarrays")
            scale = np.full(n_dims, 1)
            scale[:2] = images[0].shape[:2]
        if np.isscalar(scale):
            scale = np.full(n_dims, scale)

        if boxes is not None:
            boxes.pos1[:,:2] *= scale
            boxes.pos2[:,:2] *= scale
        elif positions is not None:
            boxes = []
            for i in range(len(images)):
                imageshape = np.ones_like(positions[i])
                imageshape[:2] = np.array(images[i].shape[:2])
                boxes.append(BBox(
                    positions[i] * scale,
                    positions[i] * scale + imageshape * self.scale * imagescale
                ))
        
        #self.images.extend(images)
        for image, box in zip(images, boxes):
            if len(image.shape) == 3:
                if channel_axis is None:
                    raise ValueError('Expected images of dimension (W, H), got {}'.format(image.shape))

                axes = list(range(3))
                axes.pop(channel_axis)
                image = image.transpose(*axes, channel_axis)

            else:
                if channel_axis is not None:
                    raise ValueError('Expected images with at least 3 dimensions, as channel_axis is set')
                if self.multichannel:
                    image = image.reshape(*image.shape, 1)

            self._add_image(image, box)

    def add_image(self, image, position=None, box=None, scale='pixel', imagescale=1):
        return self.add_images([image], 
            positions = position and [position],
            boxes = box and [box],
            scale = scale, imagescale = imagescale)

    def _add_image(self, image, box):
        self.images.append(image)
        self.boxes.append(box)

    def add_split_image(self, image, grid_size=None, tile_shape=None, overlap=0.1, channel_axis=None):
        """ Adds an image split into a number of tiles. This can be used to divide up
        a large image into smaller pieces for efficient processing. The resulting
        images are guaranteed to all be the same size.
        A common pattern would be:

        composite.add_split_image(image, 10)
        for i in range(len(composite.images)):
            composite.images[i] = process(composite.images[i])
        result = composite.stitch_images()

            image: ndarray
                the image that will be split into tiles
            grid_size: int or (int, int)
                the number of tiles to split the image into. Either this or tile_shape
                should be specified.
            tile_shape: (int, int)
                The shape of the resulting tiles, if grid_size isn't specified the maximum
                number of tiles that fit in the image are extracted. Whether specified or not,
                the size of all tiles created is guaranteed to be uniform.
            overlap: float, int or (float or int, float or int)
                The amount of overlap between neighboring tiles. Zero will result in no overlap,
                a floating point number represents a percentage of the size of the tile, and an
                integer number represents a flat pixel overlap. The overlap is treated as a lower bound,
                as it is not always possible to get the exact overlap requested due to rounding issues,
                and in some cases more overlap will exist between some tiles
        """
        assert grid_size or tile_shape, "Must specify either grid_size or tile_shape"

        if channel_axis is not None:
            axes = list(range(3))
            axes.pop(channel_axis)
            image = image.transpose(*axes, channel_axis)

        if type(grid_size) == int:
            grid_size = (grid_size, grid_size)
        if type(overlap) in (int, float):
            overlap = (overlap, overlap)
        
        if grid_size:
            if type(overlap[0]) == float:
                tile_offset = (
                    #int(image.shape[0] // (grid_size[0] + overlap[0])),
                    #int(image.shape[1] // (grid_size[1] + overlap[1])),
                    image.shape[0] / (grid_size[0] + overlap[0]),
                    image.shape[1] / (grid_size[1] + overlap[1]),
                )
                overlap = tile_offset[0] * overlap[0], tile_offset[1] * overlap[1]
            else:
                tile_offset = (
                    #int((image.shape[0] - overlap[0]) // grid_size[0]),
                    #int((image.shape[1] - overlap[1]) // grid_size[1]),
                    (image.shape[0] - overlap[0]) / grid_size[0],
                    (image.shape[1] - overlap[1]) / grid_size[1],
                )
            tile_shape = math.ceil(tile_offset[0] + overlap[0]), math.ceil(tile_offset[1] + overlap[1])
        else:
            if type(overlap[0]) == float:
                overlap = tile_shape[0] * overlap[0], tile_shape[1] * overlap[1]
            tile_offset = tile_shape[0] - overlap[0], tile_shape[1] - overlap[1]
            grid_size = math.ceil((image.shape[0] - overlap[0]) / tile_offset[0]), math.ceil((image.shape[1] - overlap[1]) / tile_offset[1])

        images = []
        positions = []
        for xpos in np.linspace(0, image.shape[0] - tile_shape[0], grid_size[0]):
            for ypos in np.linspace(0, image.shape[1] - tile_shape[1], grid_size[1]):
        #for xpos in range(0, tile_offset[0] * grid_size[0], tile_offset[0]):
            #for ypos in range(0, tile_offset[1] * grid_size[1], tile_offset[1]):
                xpos, ypos = round(xpos), round(ypos)
                images.append(image[xpos:xpos+tile_shape[0],ypos:ypos+tile_shape[1]])
                positions.append((xpos, ypos))

        if channel_axis is None:
            self.add_images(images, positions)
        else:
            self.add_images(images, positions, channel_axis=-1)


    def apply(self, positions):
        """ Applies new positions to images in this composite. positions is either a dict
        mapping image indices to new positions or a sequence of new positions.
        """
        if type(positions) == dict:
            iterable = positions.items()
        else:
            iterable = enumerate(positions)

        for index, pos in iterable:
            self.boxes[index].pos2[:2] = pos + self.boxes[index].size[:2]
            self.boxes[index].pos1[:2] = pos

    @property
    def positions(self):
        return self.boxes.pos1

    def merge(self, other_composite, *other_constraint_sets, new_layer=False, align_coords=False):
        """ Adds all images and constraints from another montage into this one.
            
            other_composite: CompositeImage
                Another composite instance that will be added to this one. All images from
                it are added to this instance. All image positions are added, mantaining
                the scale_factors of both composites.

            Returns: list of indices
                returns the list of indices of the images added from the other composite.
        """
        scale_conversion = 1 if other_composite.scale == 1 else 1 / other_composite.scale
        start_index = len(self.images)

        if new_layer:
            if len(self.boxes) and self.boxes.pos1.shape[1] < 3:
                self.boxes.resize(3)
                new_layer = int(len(self.boxes) != 0)
            else:
                new_layer = self.boxes.pos2[:,2].max() + 1

            for image, box in zip(other_composite.images, other_composite.boxes):
                newbox = BBox(box.pos1 * scale_conversion, box.pos2 * scale_conversion)
                newbox.pos1.resize(3)
                newbox.pos2.resize(3)
                newbox.pos1[2] = new_layer
                newbox.pos2[2] = new_layer
                self._add_image(image, newbox)

        else:
            for image, box in zip(other_composite.images, other_composite.boxes):
                newbox = BBox(box.pos1 * scale_conversion, box.pos2 * scale_conversion)
                self._add_image(image, newbox)

        if align_coords:
            pass

        subcomposite = self.subcomposite(list(range(start_index, len(self.images))))
        constraint_sets = [subcomposite.convert(const_set) for const_set in other_constraint_sets]

        if len(constraint_sets) == 0:
            return subcomposite
        else:
            return [subcomposite] + constraint_sets

    def align_disconnected_regions(self, num_test_points=0.05, expand_range=5):
        """ Looks at the current constraints in this composite and sees if there are any images or
        groups of images that are fully disconnected from the rest of the images. If any are found,
        they are attempted to be joined back together by calculating select constraints between the
        two groups
        """

        connections = {}
        for pair in self.constraints:
            connections.setdefault(pair[0], set()).add(pair[1])
            connections.setdefault(pair[1], set()).add(pair[0])

        def get_connected(starting_image):
            images = set()
            new_images = {starting_image}

            while len(new_images):
                images.update(new_images)
                new_groups = [connections[image] - images for image in new_images]
                new_images = set().union(*new_groups)

            return images

        """
        def get_connected(image, all_images=None):
            all_images = all_images or set()
            if image not in all_images:
                all_images.add(image)
                for next_image in connections[image]:
                    get_connected(next_image, all_images)
            return all_images
            """
        
        groups = []
        images_left = set(range(len(self.images)))
        while len(images_left) > 0:
            start_image = next(iter(images_left))
            group = get_connected(start_image)
            images_left -= group
            groups.append(group)
        
        if len(groups) == 1:
            return

        groups.sort(key=lambda group: -len(group))
        self.debug('Found', len(groups), 'disconnected groups, with', list(map(len, groups)), 'sizes')

        #rng = random.Random(random_state)

        while len(groups) > 1:
            self.debug ('Merging groups', len(groups[0]), len(groups[1]))
            #merge two largest groups
            maingroup = groups[0]
            newgroup = groups[1]
            mainboxes = [self.boxes[i] for i in maingroup]
            newboxes = [self.boxes[i] for i in newgroup]

            num_test_points_group = int(num_test_points * (len(maingroup) + len(newgroup)))

            all_poses = self.boxes.center[list(maingroup|newgroup),:2]
            self.debug ('   ', all_poses.shape)

            all_poses = []
            for i in maingroup:
                if any(self.boxes[i].as2d().overlaps(obox.as2d()) for obox in newboxes):
                    all_poses.append(self.boxes[i].center[:2])

            for i in newgroup:
                if any(self.boxes[i].as2d().overlaps(obox.as2d()) for obox in mainboxes):
                    all_poses.append(self.boxes[i].center[:2])
            all_poses = np.array(all_poses)
            self.debug (all_poses.shape)

            poses = sklearn.cluster.KMeans(n_clusters=num_test_points_group).fit(all_poses).cluster_centers_
            """
            poses = []
            while len(poses) < num_test_points_group // 2:
                box = rng.choice(mainboxes)
                if any(box.overlaps(obox) for obox in newboxes):
                    poses.append(box.pos1 + box.size/2)

            while len(poses) < num_test_points_group:
                box = rng.choice(newboxes)
                if any(box.overlaps(obox) for obox in mainboxes):
                    poses.append(box.pos1 + box.size/2)
                    """

            align_boxes = [BBox(pos - 1, pos + 1) for pos in poses.astype(int)]
            matched = False

            thresh = self.calc_score_threshold()
            expand_amount = self.boxes.size[:2].max(axis=0).astype(int)

            for i in range(expand_range):
                self.debug('Testing overlap with expanded box', align_boxes[0].size)
                pairs = set()
                for box in align_boxes:
                    indices1 = [i for i in maingroup if self.boxes[i].as2d().overlaps(box)]
                    indices2 = [i for i in newgroup if self.boxes[i].as2d().overlaps(box)]
                    for i in indices1:
                        pairs.update({(i,j) for j in indices2})

                constraints = self.calc_constraints(pairs=pairs, return_constraints=True)
                offsets = []
                for (i,j), constraint in constraints.items():
                    if constraint.score >= thresh:
                        offsets.append(self.boxes[i].pos1[:2] - self.boxes[j].pos1[:2])
                offsets = np.array(offsets)
                self.debug (' Good constraints:', len(offsets), '/', len(constraints))
                self.debug (' Offsets:', offsets.mean(axis=0), offsets.std(axis=0), np.percentile(offsets, [0,1,5,50,95,99,100], axis=0))

                if len(offsets) > len(align_boxes) * 0.8:
                    offset = np.mean(offsets, axis=0).astype(int)
                    self.debug ('Found offset', offset)
                    if i != 0:
                        self.boxes.pos1[list(newgroup),:2] += offset
                        self.boxes.pos2[list(newgroup),:2] += offset
                    groups[0] = groups[0] | groups.pop(1)
                    break

                for box in align_boxes:
                    box.pos1[:2] -= expand_amount
                    box.pos2[:2] += expand_amount

    def subcomposite(self, indices):
        """ Returns a new composite with a subset of the images and constraints in this one.
        The images and positions are shared, so modifing them on the new composite will
        change them on the original.

            indices: sequence of ints, sequence of bools, function
                A way to select the images to be included in the new composite. Can be:
                a sequence of indices, a sequence of boolean values the same length as images,
        """
        
        if type(indices[0]) in (bool, np.bool_):
            indices = [i for i in range(len(indices)) if indices[i]]

        composite = SubCompositeImage(self, indices)

        return composite

    def layer(self, index):
        """ Returns a SubComposite with only images that are on the specified layer, that is
        all images where box.pos1[2] == index.
        Layers can be created when calling merge() with new_layer=True
        or manually by specifying a third dimension when adding images
        """
        assert self.boxes.pos1.shape[1] == 3, "this composite doesn't contain layers"
        return self.subcomposite(self.boxes.pos1[:,2] == index)

    def set_scale(self, scale_factor):
        """ Sets the scale factor of the composite. Normally this doesn't need to be changed,
        however if you are trying to stich together images taken at different magnifications you
        may need to modify the scale factor.
            
            scale_factor: float or int
                Scale of images in this composite, as a multiplier. Eg a scale_factor of
                10 will result in each pixel in images corresponding to 10 pixels in the
                output of functions like `CompositeImage.stitch_images()` or when merging composites together.
        """
        self.scale = scale_factor

    def calc_score_threshold(self, num_samples=None, random_state=12345):
        """ Estimates a threshold for selecting constraints with good overlap.

        Done by calculating random constraints and using a gaussian mixture model
        to distinguish random constraints from real constraints

        Args:
            num_samples (float): optional
                The number of fake constraints to be generated, defaults to 0.25*len(images).
                In general the more samples the better the estimate, at the expense of speed

            random_state (int): Used as a seed to get reproducible results

        Returns (float):
            threshold for score where all scores lower are likely to be bad constraints
        """
        num_samples = num_samples or min(250, max(10, len(self.images) // 4))
        rng = np.random.default_rng(random_state)

        fake_consts = self.constraints(max_overlap_ratio_x=-2, max_overlap_ratio_y=-2, limit=num_samples)
        if len(fake_consts) == 0:
            fake_consts = self.constraints(max_overlap_ratio_x=-0.1, max_overlap_ratio_y=-0.1, limit=num_samples)
            if len(fake_consts) == 0:
                warnings.warn("Unable to find enough non-overlapping constraints to estimate score threshold. Defaulting to 0.5")
                return 0.5

        fake_consts = fake_consts.calculate()
        
        real_consts = rng.choice([const for const in self.constraints.values() if not const.modeled], size=num_samples)
        fake_pairs = []

        fake_scores = np.array([const.score for const in fake_consts])

        thresh = np.percentile(fake_scores, 98)
        return thresh

        for i in rng.permutation(len(self.images)):
            for j in rng.permutation(len(self.images)):
                if ((i,j) not in self.constraints
                        and np.any(np.abs(self.boxes[i].pos1[:2] - self.boxes[j].pos1[:2])
                            > np.abs(self.boxes[i].pos2[:2] - self.boxes[i].pos1[:2]) * 1.5)):
                    fake_pairs.append((i,j))

                if len(fake_pairs) == len(real_consts): break
            if len(fake_pairs) == len(real_consts): break

        fake_consts = self.calc_constraints(fake_pairs, return_constraints=True, debug=False).values()
        fake_scores = np.array([const.score for const in fake_consts])
        real_scores = np.array([const.score for const in real_consts])

        thresh = np.percentile(fake_scores, 98)
        return thresh

        scores = np.array([const.score for const in real_consts] + [const.score for const in fake_consts])

        #thresh = max(const.score for const in fake_consts)
        #print (thresh, 'thresh')
        #return thresh

        #fig, axis = plt.subplots()
        #axis.hist(scores[:len(real_consts)], bins=15, alpha=0.5)
        #axis.hist(scores[len(real_consts):], bins=15, alpha=0.5)
        #axis.hist(scores, bins=15)
        #fig.savefig('plots/ncc_hist.png')

        #thresh = skimage.filters.threshold_otsu(scores)
        mix_model = sklearn.mixture.GaussianMixture(n_components=2, random_state=random_state)
        mix_model.fit(scores.reshape(-1,1))
        
        scale1, scale2 = mix_model.weights_.flatten()
        mean1, mean2 = mix_model.means_.flatten()
        var1, var2 = mix_model.covariances_.flatten()

        if (mean2 < mean1):
            scale1, scale2 = scale2, scale1
            mean1, mean2 = mean2, mean1
            var1, var2 = var2, var1

        a = 1 / (2 * var2) - 1 / (2 * var1)
        b = mean1 / var1 - mean2 / var2
        c = mean2**2 / (2 * var2) - mean1**2 / (2 * var1) + math.log(scale1 / scale2)

        inside_sqrt = b*b - 4*a*c
        if (inside_sqrt < 0):
            warnings.warn("unable to find decision boundary for gaussian model: no solution to equation."
                            "Falling back on mean threshold")
            return (mean1 + mean2) / 2

        if (a == 0):
            # same variances means only one decision boundary
            solution = -c / b
            return solution

        solution1 = -(b + math.sqrt(inside_sqrt)) / (2*a)
        solution2 = -(b - math.sqrt(inside_sqrt)) / (2*a)

        if (solution1 < mean1 or solution1 > mean2) and (solution2 < mean1 or solution2 > mean2):
            # neither solution between means, only case is means are the same
            warnings.warn("unable to find decision boundary for gaussian model: means are the same."
                            " Falling back on mean threshold")
            return (mean1 + mean2) / 2

        # choose the one between the means, the other one is usually negligable when deciding
        solution = solution1
        if mean1 < solution1 < mean2:
            return solution1
        return solution2

    def stitch(self, indices=None, real_images=None, out=None, bg_value=None, return_bg_mask=False,
            mins=None, maxes=None, keep_zero=False, merger=None):
        """ Combines images in the composite into a single image
            
            indices: sequence of int
                Indices of images in the composite to be stitched together

            real_images: sequence of np.ndarray
                An alternative image list to be used in the stitching, instead of
                the stored images. Must be same length and each image must have the
                first two dimensions the same size as self.images

            bg_value: scalar or array
                Value to fill empty areas of the image.

            return_bg_mask: bool
                If True a boolean mask of the background, pixels with no images
                in them, is returned.

            keep_zero: bool
                Whether or not to keep the origin in the result. If true this could
                result in extra blank space, which might be necessary when lining up
                multiple images.

            Returns: np.ndarray
                image stitched together
        """

        if indices is None:
            indices = list(range(len(self.images)))
        merger = merger or merging.MeanMerger()

        if type(indices[0]) == bool:
            indices = [i for i in range(len(indices)) if indices[i]]

        if keep_zero:
            mins = 0

        start_mins = np.array(self.boxes.pos1.min(axis=0)[:2])
        start_maxes = np.array(self.boxes.pos2.max(axis=0)[:2])
        
        if mins is not None:
            start_mins[:] = mins
        if maxes is not None:
            start_maxes[:] = maxes

        mins, maxes = start_mins, start_maxes

        if keep_zero:
            mins = np.zeros_like(mins)
        
        if real_images is None:
            real_images = self.images

        example_image = real_images[0]
        
        full_shape = tuple((maxes - mins) * self.scale) + example_image.shape[2:]
        merger.create_image(full_shape, example_image.dtype)
        if out is not None:
            assert merger.image.shape == out.shape and merger.image.dtype == out.dtype, (
                "Provided output image does not match expected shape or dtype: {} {}".format(merger.image.shape, merger.image.dtype))
            merger.image = out

        import matplotlib.pyplot as plt
        fig, axis = plt.subplots()

        for i in indices:
            pos1 = ((self.boxes[i].pos1[:2] - mins) * self.scale).astype(int)
            pos2 = ((self.boxes[i].pos2[:2] - mins) * self.scale).astype(int)
            image = real_images[i]

            if np.any(pos2 - pos1 != image.shape[:2]):
                warnings.warn("resizing some images")
                image = skimage.transform.resize(image, pos2 - pos1, preserve_range=True).astype(image.dtype)
            
            image = image[max(0,-pos1[0]):,max(0,-pos1[1]):]

            pos1 = np.maximum(0, np.minimum(pos1, maxes - mins))
            pos2 = np.maximum(0, np.minimum(pos2, maxes - mins))

            image = image[:pos2[0]-pos1[0],:pos2[1]-pos1[1]]

            if image.size == 0: continue

            x1, y1 = pos1
            x2, y2 = pos2
            axis.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])
            position = (slice(pos1[0], pos2[0]), slice(pos1[1], pos2[1]))
            merger.add_image(image, position)

        full_image, mask = merger.final_image()

        #fig.savefig('plots/merger_locations.png')

        if bg_value is not None:
            full_image[mask] = bg_value
        
        if return_bg_mask:
            return full_image, mask
        return full_image

    def plot_scores(self, path, constraints=None, score_func=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches

        constraints = constraints or {}

        groups = [list(range(len(self.boxes)))]
        const_groups = [constraints.keys()]
        names = ['']
        if score_func == 'accuracy':
            score_func = lambda const: np.linalg.norm(const.difference)

        if self.boxes[0].pos1.shape[0] == 3:
            groups = []
            const_groups = []
            names = []

            vals = sorted(set(box.pos1[2] for box in self.boxes))
            print (vals)
            for val in vals:
                groups.append([i for i in range(len(self.boxes)) if self.boxes[i].pos1[2] == val])
                const_groups.append([(i,j) for (i,j) in constraints.keys() if self.boxes[i].pos1[2] == val and self.boxes[j].pos1[2] == val])
                names.append('(plane z={})'.format(val))

            new_const_groups = {}
            for i,j in constraints.keys():
                pair = (self.boxes[i].pos1[2], self.boxes[j].pos1[2])
                if pair[0] == pair[1]: continue
                new_const_groups[pair] = new_const_groups.get(pair, [])
                new_const_groups[pair].append((i,j))
            
            for pair, consts in new_const_groups.items():
                groups.append([])
                const_groups.append(consts)
                names.append('(consts z={} -> z={})'.format(pair[0], pair[1]))

        axis_size = 12
        grid_size = math.ceil(np.sqrt(len(groups)))
        fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(axis_size*grid_size,axis_size*grid_size), squeeze=False, sharex=True, sharey=True)

        for indices, const_pairs, axis, name in zip(groups, const_groups, axes.flatten(), names):

            for i,index in enumerate(indices):
                x, y = self.boxes[index].pos1[:2]
                width, height = self.boxes[index].size[:2]
                axis.text(y + height / 2, -x - width / 2, "{}\n({})".format(index, i), horizontalalignment='center', verticalalignment='center')
                axis.add_patch(matplotlib.patches.Rectangle((y, -x - width), height, width, edgecolor='grey', facecolor='none'))

            poses = []
            colors = []
            sizes = []
            #for (i,j), constraint in constraints.items():
                #if i not in indices or j not in indices: continue
            for i,j in const_pairs:
                constraint = constraints[(i,j)]

                pos1, pos2 = self.boxes[i].center[:2], self.boxes[j].center[:2]
                #if np.all(pos1 == pos2):
                    #print (i, j, constraint)
                pos = np.mean((pos1, pos2), axis=0)
                poses.append((pos[1], -pos[0]))
                score = constraint.score if score_func is None else score_func(constraint)
                colors.append(score)
                sizes.append(50 if constraint.modeled else 200)
                axis.arrow(pos[1] - constraint.dy/2, -pos[0] + constraint.dx/2, constraint.dy/1, -constraint.dx/1,
                        width=5, head_width=60, length_includes_head=True, color='black')
                #axis.plot((pos1[0], pos2[0]), (pos1[1], pos2[1]), linewidth=1, color='red' if constraint.modeled else 'black')
            poses = np.array(poses)

            #self.debug('READY')
            #self.debug(poses)
            if len(poses):
                points = axis.scatter(poses[:,0], poses[:,1], c=colors, s=sizes, alpha=0.5)
                fig.colorbar(points, ax=axis)

            axis.set_title('Scores of constraints ' + name)
            axis.xaxis.set_tick_params(labelbottom=True)
            axis.yaxis.set_tick_params(labelbottom=True)

        fig.savefig(path)

    def html_summary(self, path, score_func=None):
        import xml.etree.ElementTree as ET
        html = ET.Element('html')

        head = ET.SubElement(html, 'head')
        style = ET.SubElement(head, 'style')
        style.text = """
        g .fade-hover {
            opacity: 0.2;
        }
        g:hover .fade-hover {
            opacity: 1;
        }
        g .show-hover {
            display: none;
        }
        g:hover .show-hover {
            display: block;
        }
        """

        body = ET.SubElement(html, 'body')
        
        mins, maxes = self.boxes.pos1.min(axis=0), self.boxes.pos2.max(axis=0)
        svg = ET.SubElement(body, 'svg', viewbox="{} {} {} {}".format(mins[0], mins[1], maxes[0], maxes[1]))

        for i,box in enumerate(self.boxes):
            class_names = 'box'
            if len(box.pos1) == 3:
                class_names += ' start{0} end{0}'.format(int(box.pos1[2]))
            group = ET.SubElement(svg, 'g', attrib={
                "class": class_names,
            })
            rect = ET.SubElement(group, 'rect', attrib={
                "class": "fade-hover",
                "x": str(box.pos1[0]), "y": str(box.pos1[1]),
                "width": str(box.size[0]), "height": str(box.size[1]),
                "stroke": 'black', "fill": 'transparent',
                "stroke-width": str(int(box.size[:2].min()) // 10),
            })

            text = ET.SubElement(group, 'text', attrib={
                "class": "show-hover",
                "x": str(box.pos1[0] + box.size[0] // 2),
                "y": str(box.pos1[1] + box.size[1] * 2 // 3),
                "font-size": str(box.size[1] // 2),
                "text-anchor": "middle",
            })
            text.text = str(i)

        for (i,j), constraint in self.constraints.items():
            box1, box2 = self.boxes[i], self.boxes[j]
            class_names = 'constraint'
            if len(box1.pos1) == 3:
                class_names += ' start{} end{}'.format(int(box1.pos1[2]), int(box2.pos1[2]))

            group = ET.SubElement(svg, 'g')
            center = (box1.pos1 + box1.pos2 + box2.pos1 + box2.pos2) // 4
            line = ET.SubElement(group, 'line', attrib={
                "class": "fade-hover",
                "x1": str(center[0] - constraint.dx // 2),
                "y1": str(center[1] - constraint.dy // 2),
                "x2": str(center[0] + constraint.dx // 2),
                "y2": str(center[1] + constraint.dy // 2),
                "stroke": "rgb(50% {}% 50%)".format(int(constraint.score * 100)),
                "stroke-width": str(min(box1.size[:2].min(), box2.size[:2].min()) // 2),
                "stroke-linecap": "round",
            })

            line = ET.SubElement(group, 'line', attrib={
                "class": "show-hover",
                "x1": str(box1.pos1[0] + box1.size[0] / 2),
                "y1": str(box1.pos1[1] + box1.size[1] / 2),
                "x2": str(box2.pos1[0] + box2.size[0] / 2),
                "y2": str(box2.pos1[1] + box2.size[1] / 2),
                "stroke": "black",
                "stroke-width": str(min(box1.size[:2].min(), box2.size[:2].min()) // 40),
            })

            #"""
            rect = ET.SubElement(group, 'rect', attrib={
                "class": "show-hover",
                "x": str(box1.pos1[0]), "y": str(box1.pos1[1]),
                "width": str(box1.size[0]), "height": str(box1.size[1]),
                "stroke": 'black', "fill": 'transparent',
                "stroke-width": str(int(box1.size.mean()) // 10),
            })
            text = ET.SubElement(group, 'text', attrib={
                "class": "show-hover",
                "x": str(box1.pos1[0] if box1.pos1[0] <= box2.pos1[0] else box1.pos2[0]),
                "y": str(box1.pos1[1] + box1.size[1] * 2 // 3),
                "font-size": str(box1.size[1] // 2),
                "text-anchor": "end" if box1.pos1[0] <= box2.pos1[0] else "start",
            })
            text.text = str(i)

            rect = ET.SubElement(group, 'rect', attrib={
                "class": "show-hover",
                "x": str(box2.pos1[0]), "y": str(box2.pos1[1]),
                "width": str(box2.size[0]), "height": str(box2.size[1]),
                "stroke": 'black', "fill": 'transparent',
                "stroke-width": str(int(box2.size.mean()) // 10),
            })
            text = ET.SubElement(group, 'text', attrib={
                "class": "show-hover",
                "x": str(box2.pos2[0] if box1.pos1[0] <= box2.pos1[0] else box2.pos1[0]),
                "y": str(box2.pos1[1] + box2.size[1] * 2 // 3),
                "font-size": str(box2.size[1] // 2),
                "text-anchor": "start" if box1.pos1[0] <= box2.pos1[0] else "end",
            })
            text.text = str(j)
            #"""

        with open(path, 'wb') as ofile:
            ET.ElementTree(html).write(ofile, encoding='utf-8', method='html')


    def score_heatmap(self, path, score_func=None):
        import matplotlib.pyplot as plt

        n_axes = self.boxes.pos1.shape[1]

        fig, axes = plt.subplots(nrows=n_axes, figsize=(8, 5*n_axes))

        for index, axis in enumerate(axes):
            values = np.unique(self.boxes.pos1[:,index])
            if len(values) > 25:
                values = np.linspace(values[0], values[-1], 25)

            scores = np.zeros((len(values), len(values)))
            counts = np.zeros(scores.shape, int)

            for (i,j), constraint in self.constraints.items():
                posi = self.boxes[i].pos1
                posj = self.boxes[j].pos1
                xval, yval = np.digitize([posi[index], posj[index]], values)
                #xval, yval = min(xval, len(values)-1), min(yval, len(values)-1)
                xval, yval = xval-1, yval-1
                score = constraint.score if score_func is None else score_func(i, j, constraint)
                scores[xval, yval] += score
                counts[xval, yval] += 1

            scores[counts!=0] /= counts[counts!=0]

            heatmap = axis.imshow(scores)
            #axis.set_xlabels(values)
            #axis.set_ylabels(values)
            fig.colorbar(heatmap, ax=axis)

        fig.savefig(path)
    
    def constraint_error(self, i, j, constraint):
        new_offset = self.boxes[j].pos1[:2] - self.boxes[i].pos1[:2]
        diff = (new_offset[0] - constraint.dx, new_offset[1] - constraint.dy)
        return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])



class SubCompositeConstraintSet(CompositeConstraintSet):
    def __init__(self, composite, pair_func, mapping):
        super().__init__(composite, pair_func)
        self.mapping = mapping

    def __getitem__(self, pair):
        pair = self.mapping[pair[0]], self.mapping[pair[1]]
        return super().__getitem__(pair)


class SubCompositeList:
    def __init__(self, items, mapping):
        self.items = items
        self.mapping = mapping

    def __getitem__(self, index):
        index = self.mapping[index]
        return self.items[index]

    def append(self, item):
        raise "appending to subcomposite list"

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return iter(self.items[i] for i in self.mapping)


class SubCompositeBBoxList(BBoxList):
    def __init__(self, boxes, mapping):
        self.boxes = SubCompositeList(boxes, mapping)

    def append(self, box):
        raise "appending to subcomposite list"

    @property
    def pos1(self):
        return self.boxes.items.pos1[self.boxes.mapping]

    @property
    def pos2(self):
        return self.boxes.items.pos2[self.boxes.mapping]


class SubCompositeImage(CompositeImage):
    def __init__(self, composite, mapping, layer=None, debug=True, progress=False, executor=None, aligner=None):
        self.composite = composite
        self.mapping = mapping

        self.images = SubCompositeList(self.composite.images, self.mapping)
        self.boxes = SubCompositeBBoxList(self.composite.boxes, self.mapping)

        self.constraints = SubCompositeConstraintSet(self.composite, self.pair_func, mapping)
        #self.scale = 1
        self.layer = layer

    def pair_func(self):
        for i in self.mapping:
            for j in self.mapping:
                if i < j:
                    yield i, j

    def _add_image(self, image, box):
        self.mapping.append(len(self.composite.images))
        if self.layer is not None and len(box.pos1) == 2:
            box.pos1 = np.array([*box.pos1, self.layer])
            box.pos1 = np.array([*box.pos1, self.layer + 1])

        self.composite._add_image(image, box)

    @property
    def debug(self):
        return self.composite.debug

    @property
    def progress(self):
        return self.composite.progress

    @property
    def scale(self):
        return self.composite.scale

    @property
    def multichannel(self):
        return self.composite.multichannel

    def contains(self, index):
        return index in self.mapping

    def subcomposite(self, indices):
        return self.composite.subcomposite(self, np.array(self.mapping)[indices])

    def merge(self, other_composite):
        raise "Cannot merge into a SubComposite"

    def layer(self, index):
        raise "SubComposites cannot contain layers"

    def convert(self, constraints):
        if constraints.composite is self.composite:
            raise "Passed constraints are already converted"

        consts = []
        for const in constraints._constraint_iter():
            const = Constraint(self.composite, **const.to_obj())
            const.index1 = self.mapping[const.index1]
            const.index2 = self.mapping[const.index2]
            consts.append(const)

        newset = ConstraintSet(consts)
        return newset

