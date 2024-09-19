import sys
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
from .constraints import Constraint
from . import merging, alignment, solving
from .. import utils



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

    def size(self):
        return self.pos2 - self.pos1

    def center(self):
        return (self.pos1 + self.pos2) / 2


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
        
        #result = func(*args, **kwargs)
        #future = concurrent.futures.Future()
        #future.set_result(result)
        #return future


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

    def __init__(self, aligner=None, precalculate=False, debug=True, progress=False, executor=None):
        self.images = []
        self.greyimages = []
        self.boxes = BBoxList()
        self.constraints = {}
        self.scale = 1
        self.stage_model = None
        self.set_logging(debug, progress)
        self.set_executor(executor)
        self.set_aligner(aligner)

        self.precalculate = precalculate

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

    @classmethod
    def view(cls, other):
        composite = cls()
        composite.root = other
        composite.mapping = None

    def save(self, path, save_images=True):
        """ Saves this composite to the given path. Can be restored with the `CompositeImage.load()`
        function.

            save_images: bool, whether the images should be included in the save file. If they are
                excluded the programmer needs to restore them when loading. Not saving the images
                can reduce the memory needed for the composite file and speed up loading and saving
                dramatically
        """

        obj = dict(
            boxes = (self.boxes.pos1, self.boxes.pos2),
            constraints = self.constraints,
            scale = self.scale,
            stage_model = self.stage_model,
            debug = bool(self.debug),
            progress = bool(self.progress),
        )
        if save_images:
            obj['images'] = self.images
        else:
            obj['images'] = [None] * len(self.images)

        with open(path, 'wb') as ofile:
            pickle.dump(obj, ofile)

    @classmethod
    def load(cls, path, **kwargs):
        """ Loads a composite previously saved with `CompositeImage.save()` to the given
        path. If the composite was saved without images they should be restored by setting
        composite.images to a list of the images.
        """
            
        obj = pickle.load(open(path, 'rb'))
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

    def imagearr(self, image):
        image = self.fullimagearr(image)
        while len(image.shape) > 2:
            image = image[:,:,0]
        return image

    def fullimagearr(self, image):
        if type(image) == str:
            image = iio.imread(image)
        return image

    def imageshape(self, image):
        if type(image) == str:
            return iio.improps(image).shape[:2]
        return image.shape[:2]

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
            scale[:2] = self.imageshape(images[0])
        if np.isscalar(scale):
            scale = np.full(n_dims, scale)

        if boxes is not None:
            for i in range(len(images)):
                self.boxes.append(BBox(boxes[i].pos1 * scale, boxes[i].pos2 * scale))
        elif positions is not None:
            for i in range(len(images)):
                imageshape = np.ones_like(positions[i])
                imageshape[:2] = np.array(self.imageshape(images[i]))
                self.boxes.append(BBox(
                    positions[i] * scale,
                    positions[i] * scale + imageshape * self.scale * imagescale
                ))
        
        #self.images.extend(images)
        for image in images:
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

            self.images.append(image)

    def add_image(self, image, position=None, box=None, scale='pixel', imagescale=1):
        return self.add_images([image], 
            positions = position and [position],
            boxes = box and [box],
            scale = scale, imagescale = imagescale)

    def add_split_image(self, image, num_tiles=None, tile_shape=None, overlap=0.1, channel_axis=None):
        """ Adds an image split into a number of tiles. This can be used to divide up
        a large image into smaller peices for efficient processing. The resulting
        images are guaranteed to all be the same size.
        A common pattern would be:

        composite.add_split_image(image, 10)
        for i in range(len(composite.images)):
            composite.images[i] = process(composite.images[i])
        result = composite.stitch_images()

            image: ndarray
                the image that will be split into tiles
            num_tiles: int or (int, int)
                the number of tiles to split the image into. Either this or tile_shape
                should be specified.
            tile_shape: (int, int)
                The shape of the resulting tiles, if num_tiles isn't specified the maximum
                number of tiles that fit in the image are extracted. Whether specified or not,
                the size of all tiles created is guaranteed to be uniform.
            overlap: float, int or (float or int, float or int)
                The amount of overlap between neighboring tiles. Zero will result in no overlap,
                a floating point number represents a percentage of the size of the tile, and an
                integer number represents a flat pixel overlap. The overlap is treated as a lower bound,
                as it is not always possible to get the exact overlap requested due to rounding issues,
                and in some cases more overlap will exist between some tiles
        """
        assert num_tiles or tile_shape, "Must specify either num_tiles or tile_shape"

        if channel_axis is not None:
            axes = list(range(3))
            axes.pop(channel_axis)
            image = image.transpose(*axes, channel_axis)

        if type(num_tiles) == int:
            num_tiles = (num_tiles, num_tiles)
        if type(overlap) in (int, float):
            overlap = (overlap, overlap)
        
        if num_tiles:
            if type(overlap[0]) == float:
                tile_offset = (
                    #int(image.shape[0] // (num_tiles[0] + overlap[0])),
                    #int(image.shape[1] // (num_tiles[1] + overlap[1])),
                    image.shape[0] / (num_tiles[0] + overlap[0]),
                    image.shape[1] / (num_tiles[1] + overlap[1]),
                )
                overlap = tile_offset[0] * overlap[0], tile_offset[1] * overlap[1]
            else:
                tile_offset = (
                    #int((image.shape[0] - overlap[0]) // num_tiles[0]),
                    #int((image.shape[1] - overlap[1]) // num_tiles[1]),
                    (image.shape[0] - overlap[0]) / num_tiles[0],
                    (image.shape[1] - overlap[1]) / num_tiles[1],
                )
            tile_shape = math.ceil(tile_offset[0] + overlap[0]), math.ceil(tile_offset[1] + overlap[1])
        else:
            if type(overlap[0]) == float:
                overlap = tile_shape[0] * overlap[0], tile_shape[1] * overlap[1]
            tile_offset = tile_shape[0] - overlap[0], tile_shape[1] - overlap[1]
            num_tiles = math.ceil((image.shape[0] - overlap[0]) / tile_offset[0]), math.ceil((image.shape[1] - overlap[1]) / tile_offset[1])

        images = []
        positions = []
        for xpos in np.linspace(0, image.shape[0] - tile_shape[0], num_tiles[0]):
            for ypos in np.linspace(0, image.shape[1] - tile_shape[1], num_tiles[1]):
        #for xpos in range(0, tile_offset[0] * num_tiles[0], tile_offset[0]):
            #for ypos in range(0, tile_offset[1] * num_tiles[1], tile_offset[1]):
                xpos, ypos = round(xpos), round(ypos)
                images.append(image[xpos:xpos+tile_shape[0],ypos:ypos+tile_shape[1]])
                positions.append((xpos, ypos))

        print (positions)
        if channel_axis is None:
            self.add_images(images, positions)
        else:
            self.add_images(images, positions, channel_axis=-1)
    
    def image_positions():
        """ Returns the positions of all images in pixel values
        """
        return np.array([box.pos1 for box in self.boxes])

    def merge(self, other_composite, new_layer=False, align_coords=False):
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

        for image in other_composite.images:
            self.images.append(image)

        if new_layer:
            if len(self.boxes) and len(self.boxes[0].pos1) < 3:
                for box in self.boxes:
                    box.pos1.resize(3)
                    box.pos2.resize(3)

            new_layer = self.boxes.pos2[:,2].max() + 1

            for box in other_composite.boxes:
                newbox = BBox(box.pos1 * scale_conversion, box.pos2 * scale_conversion)
                newbox.pos1.resize(3)
                newbox.pos2.resize(3)
                newbox.pos1[2] = new_layer
                newbox.pos2[2] = new_layer
                self.boxes.append(newbox)

        else:
            for box in other_composite.boxes:
                self.boxes.append(BBox(box.pos1 * scale_conversion, box.pos2 * scale_conversion))

        for (i,j), constraint in other_composite.constraints.items():
            self.constraints[(i+start_index,j+start_index)] = Constraint(
                    dx=constraint.dx * scale_conversion, dy=constraint.dy * scale_conversion, 
                    score=constraint.score, modeled=constraint.modeled, error=constraint.error * scale_conversion)

        if align_coords:
            pass

        return list(range(start_index, len(self.images)))

    def align_coordinate_space(indices, num_samples=10, search_radius=5, random_state=12345):
        """ Takes all images specified by indices and makes sure that their coordinate space is consistent
        with the coordinate space of the composite. If not it is corrected by calculating some
        pairwise constraints
        """
        rng = np.random.default_rng(random_state)

        selected_images = []
        selected_scores = []
        for i in rng.shuffle(len(self.images)):
            if i in indices: continue
            selected_images.append(i)
            selected_scores.append(ncc(self.imagearr(self.images[i]), self.imagearr(self.images[i])))
            if len(selected_images) > num_samples * 2: break
        
        #take images that have higher score against themeselvs, means more features for alignment
        selected_images = np.array(selected_images)[np.argsort(selected_scores)[num_samples:]]

        radius = 1
        while radius <= search_radius:
            for index in selected_images:
                pass

    def align_disconnected_regions(self):
        """ Looks at the current constraints in this composite and sees if there are any images or
        groups of images that are fully disconnected from the rest of the images. If any are found,
        they are attempted to be joined back together by calculating select constraints between the
        two groups
        """

        connections = {}
        for pair in self.constraints:
            connections.setdefault(pair[0], []).append(pair[1])
            connections.setdefault(pair[1], []).append(pair[0])

        def get_connected(image, all_images=None):
            all_images = all_images or set()
            if image not in all_images:
                all_images.add(image)
                for next_image in connections[image]:
                    get_connected(next_image, all_images)
            return all_images
        
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
        self.debug('Found', len(groups) - 1, 'disconnected groups, with', list(map(len, groups[1:])), 'sizes')

    def subcomposite(self, indices):
        """ Returns a new composite with a subset of the images and constraints in this one.
        The images and positions are shared, so modifing them on the new composite will
        change them on the original.

            indices: sequence of ints, sequence of bools, function
                A way to select the images to be included in the new composite. Can be:
                a sequence of indices, a sequence of boolean values the same length as images,
        """
        
        if type(indices[0]) == bool:
            indices = [i for i in range(len(indices)) if indices[i]]

        composite = type(self)(
            aligner = self.aligner,
            executor = self.executor,
            debug = self.debug,
            progress = self.progress,
        )

        composite.images = [self.images[i] for i in indices]
        #composite.boxes = [self.boxes[i] for i in indices]
        composite.boxes = BBoxList(self.boxes.pos1[indices], self.boxes.pos2[indices])

        for (i,j), constraint in self.constraints.items():
            if i in indices and j in indices:
                composite.constraints[(indices.index(i), indices.index(j))] = constraint

        return composite

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

    def find_pairs(self, overlap_threshold=None, needs_overlap=False, max_pairs=None):
        """ Finds all pairs of images that overlap, based on the estimated positions
            
            Args:
                overlap_threshold (scalar or ndarray):
                    Specifies the amount of overlap that is necessary to consider
                    two images overlapping, in pixels. Defaults to zero, meaning that
                    images that are overlapping any or next to each other count as a pair.
                    Can be negative, in which case images that are within said amount
                    of touching would be overlapping.
                    Additionally a ndarray with a value for each dimension can be passed.

                needs_overlap (bool):
                    Whether or not images need to be fully overlapping or if touching
                    on the edge is enough to count it as a pair

                connectivity (int):
                    If this number is greater than 1, some pairs will be pruned, while
                    keeping the maximum di

            Returns: np.ndarray shape (N, 2)
                The sequence of pairs of indices of the images that overlap.
        """
        pairs = []
        for i in range(len(self.images)):
            start_len = len(pairs)
            for j in range(i+1,len(self.images)):
                box1, box2 = self.boxes[i], self.boxes[j]
                
                if overlap_threshold:
                    box1 = BBox(box1.pos1 + overlap_threshold, box1.pos2 - overlap_threshold)
                    box2 = BBox(box2.pos1 + overlap_threshold, box2.pos2 - overlap_threshold)

                if needs_overlap:
                    if box1.overlaps(box2):
                        pairs.append((i,j))
                else:
                    if box1.collides(box2):
                        pairs.append((i,j))

                if max_pairs and len(pairs) - start_len >= max_pairs:
                    break

        return np.array(pairs)

    def find_unconstrained_pairs(self, *args, **kwargs):
        """ Finds all pairs of images that overlap based on the estimated positions and
        that don't already have a constraint, or that have a constraint with an error value

        Returns (np.ndarray shape (N, 2)):
            The sequence of pairs of indices of the images that overlap without constraints.
        """
        pairs = self.find_pairs(*args, **kwargs)
        mask = [(i,j) not in self.constraints or self.constraints[i,j].error > 0 for i,j in pairs]
        return pairs[mask]

    def prune_pairs(self, pairs):
        pass

    def test_pairs(self, pairs, scale=16):
        """ Estimates the constraints between the given pairs with scaled down images, and
        removes any that don't seem to have any overlap. Useful when microscope positions are
        not very accurate and you need to expand the search space, as this will avoid fully calculating extra
        contraints

            scale: factor to downscale images, 16 is a good balance between speed and accuracy.
        """
        composite = CompositeImage(debug=self.debug, progress=self.progress)
        self.debug('got composite')
        #images = [skimage.transform.rescale(image, 1/scale) for image in self.progress(self.images)]
        images = [skimage.transform.downscale_local_mean(image, scale).astype(image.dtype) for image in self.progress(self.images)]
        self.debug('scaled images')
        composite.add_images(images, self.boxes.pos1 // scale)
        self.debug('put in images')

        composite.calc_constraints(pairs, precalculate=True, num_peaks=2)
        self.debug('cacled constraints')
        composite.filter_constraints()
        self.debug('filtered')
        composite.filter_outliers()
        self.debug('filtered out')
        #composite.plot_scores('plots/tmp_constraints.png')
        composite.save('selected_composite.bin', save_images=False)

        return composite.constraints.keys()

    def calc_constraints(self, pairs=None, precalculate=False, return_constraints=False, use_previous_constraints=True, debug=True, **kwargs):
        """ Estimates the pairwise translations of images and add thems as constraints
        in the montage

        Args:
            pairs (sequence of tuples, optional): optional.
                The indices of image pairs to add constraints to. Defaults to
                all images that overlap or are adjacent based on the estimated
                positions that don't already have constraints, see 
                `CompositeImage.find_unconstrained_pairs()` for more info.
                Important: the pairs given are not checked for overlap, so invalid
                constraints could be generated if specific indices are passed in.

            precalculate (bool): default false
                Most methods of alignments have some calculation that happens per image
                and some that is per image pair, so this flag means that the per image
                calculations will happen first and only once for each image. This will
                improve speed at the expense of memory.

            return_constraints (bool): default False,
                If true returns constraints instead of adding them to the montage

        Returns:
            dict: if return_constraints is True, a dict of the calculated constraints
                is returned, otherwise nothing.
        """
        if pairs is None:
            pairs = self.find_unconstrained_pairs()

        constraints = {}

        if precalculate:
            precalcs = [self.executor.submit(precalc_job, aligner=self.aligner, image=image, shape=box.size()[:2])
                        for image,box in zip(self.images, self.boxes)]
            precalcs = [future.result() for future in self.progress(precalcs)]
        else:
            precalcs = [None] * len(self.images)

        futures = []
        for index1, index2 in pairs:
            futures.append(self.executor.submit(align_job,
                aligner = self.aligner,
                image1 = self.images[index1], image2 = self.images[index2],
                precalc1 = precalcs[index1], precalc2 = precalcs[index2],
                shape1 = self.boxes[index1].size()[:2], shape2 = self.boxes[index2].size()[:2],
                previous_constraint = self.constraints.get((index1, index2), None) if use_previous_constraints else None,
            ))

        for (index1, index2), future in self.progress(zip(pairs, futures), total=len(futures)):
            result = future.result()
            if result is not None:
                constraints[(index1,index2)] = result

        if debug and len(constraints) != 0:
            scores = np.array([const.score for const in constraints.values()])
            self.debug("Calculated {} constraints, score values: min {} mean {} max {}".format(len(scores), scores.min(), scores.mean(), scores.max()))

        if return_constraints:
            return constraints
        else:
            self.constraints.update(constraints)

    def estimate_constraints(self, pairs=None, error=0.3, score=0, return_constraints=False):
        """ Estimates constraints based on the current stage positions of the images
        Useful if you are pretty confident in the stage positions as these constraints
        will keep the calculated ones within a certain range of the stage position.
        """
        if pairs is None:
            pairs = self.find_unconstrained_pairs()
            print (pairs)

        constraints = {}

        for index1, index2 in pairs:
            offset = self.boxes[index2].pos1 - self.boxes[index1].pos1
            cur_error = error
            if error < 1:
                cur_error = int(max(*self.boxes[index1].size()[:2], *self.boxes[index2].size()[:2]) * error)
            constraint = Constraint(dx=offset[0], dy=offset[1], score=score, error=cur_error)
            constraints[index1,index2] = constraint

        if return_constraints:
            return constraints
        else:
            self.constraints.update(constraints)
            print (constraints)

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
        
        real_consts = rng.choice([const for const in self.constraints.values() if not const.modeled], size=num_samples)
        fake_pairs = []

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

    def filter_constraints(self, score_threshold=None, remove_modeled=False):
        """ Removes constraints that don't meet a score threshold

        Args:
            score_threshold (float): optional
                Cutoff value for score filtering, all constraints below are
                removed. If not specified a threshold is estimated by calc_score_threshold

            remove_modeled (bool):
                Whether or not constraints that were calculated from the stage model should be
                removed. Normally these constraints should not be removed because they have
                mucho lower scores than others, but you still want to include them for tiles
                that may have no features for alignment.
        """
        if score_threshold is None:
            score_threshold = self.calc_score_threshold()

        start_size = len(self.constraints)
        for pair in list(self.constraints): # make list now because modifing in loop
            if self.constraints[pair].score < score_threshold and (not self.constraints[pair].modeled or remove_modeled):
                del self.constraints[pair]

        self.debug('Filtered out', start_size - len(self.constraints), 'constraints with the threshold', score_threshold)

    def estimate_stage_model(self, model=None, filter_outliers=False, pairs=None, random_state=12345):
        """ Estimages a stage model that translates between estimated
        position differences and pairwise alignment differences.
            
        Args:
            model (sklearn model instance):
                Used as the model to estimate the stage model. fit is called on it
                with the estimated offsets as X and the offset from constraints as y.
                Defaults to LinearRegression
        """
        model = model or SimpleOffsetModel()
        pairs = pairs or self.constraints.keys()
        if filter_outliers:
            model = sklearn.linear_model.RANSACRegressor(model,
                    min_samples=self.boxes[0].pos1.shape[0]*2,
                    max_trials=1000,
                    random_state=random_state)

        est_poses = []
        const_poses = []
        indices = []
        for (i,j) in pairs:
            constraint = self.constraints[(i,j)]
            if not constraint.modeled:
                #est_poses.append(self.boxes[j].pos1 - self.boxes[i].pos1)
                est_poses.append(np.concatenate([self.boxes[i].pos1, self.boxes[j].pos1]))
                const_poses.append((constraint.dx, constraint.dy))
                indices.append((i,j))

        est_poses, const_poses = np.array(est_poses), np.array(const_poses)
        indices = np.array(indices)

        model.fit(est_poses, const_poses)
        #print (model.estimator.model.coef_)

        if filter_outliers:
            self.debug('Filtered out', np.sum(~model.inlier_mask_), 'constraints as outliers')

            if np.mean(model.inlier_mask_.astype(int)) < 0.8:
                warnings.warn("Stage model filtered out over 20% of constraints as outilers."
                        " It may have hyperoptimized to the data, make sure all are actually outliers")

            est_poses, const_poses = est_poses[model.inlier_mask_], const_poses[model.inlier_mask_]
            self.debug("Estimated stage model", model, "with an r2 score of", model.score(est_poses, const_poses),
                    ", classifying {}/{} constraints as outliers".format(np.sum(~model.inlier_mask_), len(self.constraints)))

            for (i,j) in indices[~model.inlier_mask_]:
                del self.constraints[(i,j)]

            model = model.estimator_

            """
            error = model.predict(est_poses) - const_poses
            print (error.astype(int))
            error = np.sum(error * error, axis=1)
            thresh = np.sum((const_poses/3) * (const_poses/3), axis=1).mean()
            print (error.astype(int))
            print (thresh)
            mask = error > thresh
            if np.any(mask):
                self.debug('Filtered out', np.sum(mask), 'constraints as outliers')
                for (i,j) in indices[mask]:
                    del self.constraints[(i,j)]
            """
        else:
            self.debug("Estimated stage model", model, "with an r2 score of", model.score(est_poses, const_poses))

        if (model.predict([[0] * est_poses.shape[1]]).max() > const_poses.max() * 100 or 
                model.predict([[1] * est_poses.shape[1]]).max() > const_poses.max() * 100):
            warnings.warn("Stage model is predicting very large values for simple constraints,"
                " it may have hyperoptimized to the training data.")

        self.stage_model = model

        # calculate variance
        error = model.predict(est_poses) - const_poses
        print ("Stage model error", np.percentile(np.abs(error), [0,5,50,75,95,100]))
        error = np.percentile(np.abs(error), 75)
        self.stage_model_error = error

    def filter_outliers(self, pairs=None):
        """ Filters out any constraints that are clearly outliers, ie have a much larger magnitude than
        any other constraint.
        This doesn't take into account the stage model or the positions, to have the best outlier removal
        follow this call with a call to `CompositeImage.estimate_stage_model(filter_outilers=True)`
        
        pairs: sequence of index pairs to be considered
        """

        pairs = pairs if pairs is not None else list(self.constraints.keys())
        translations = []
        real_pairs = []
        for pair in pairs:
            if (pair[0], pair[1]) in self.constraints:
                constraint = self.constraints[(pair[0], pair[1])]
                translations.append((constraint.dx, constraint.dy))
                real_pairs.append(pair)

        if len(real_pairs) == 0:
            return
        pairs = np.array(real_pairs)
        print (pairs.shape)
        translations = np.array(translations)

        magnitudes = np.sqrt(np.sum(translations * translations, axis=1))
        thresh = np.percentile(magnitudes, 95) * 10
        mask = magnitudes > thresh
        if np.sum(mask) > 0:
            self.debug("Filtered out {} constraints as outilers".format(np.sum(mask)))

        for pair in pairs[mask]:
            del self.constraints[(pair[0], pair[1])]

    def model_constraints(self, pairs=None, score_multiplier=0.5, use_stage_model_error=True, error=0, return_constraints=False):
        """ Uses the stored stage model (estimated by estimage_stage_model) to
        fill the specified constraints.
            
        Args:
            pairs (sequence of (i,j)): optional
                The indices of image pairs to add constraints to. Defaults to
                all images that overlap or are adjacent based on the estimated
                positions that don't already have constraints, see 
                `CompositeImage.find_unconstrained_pairs()` for more info.
                Important: the pairs given are not checked for overlap, so invalid
                constraints could be generated if specific indices are passed in.
                Also passing a pair that already has a constraint will overwrite it

            score_multiplier (float):
                multiplier to scores of constraints calculated. Used to prioritize
                non modeled constraints when solving positions

            return_constraints (bool):
                whether to store constraints in self.constraints or to return them
                as a new dictionary.

            Returns (dict or None)
                Returns a dictionary of the calculated constraints if return_constraints
                is True, otherwise nothing.
        """
        if pairs is None:
            pairs = self.find_unconstrained_pairs()

        if self.stage_model is None:
            self.estimate_stage_model()

        if use_stage_model_error:
            error = self.stage_model_error

        constraints = {} if return_constraints else self.constraints

        start_size = len(constraints)
        for i,j in pairs:
            dx, dy = self.stage_model.predict(np.array([self.boxes[i].pos1, self.boxes[j].pos1]).reshape(1,-1)).flatten().astype(int)
            #dx, dy = self.stage_model.predict((self.boxes[j].pos1 - self.boxes[i].pos1).reshape(1,-1)).flatten().astype(int)
            assert (max(abs(dx), abs(dy)) <= max(self.boxes[i].size()[:2])), (
                "Image offset from stage model does not contain any overlap."
                " The stage model may not have correctly modeled the movement")
            #score = score_offset(self.images[i], self.images[j], dx, dy) * score_multiplier
            score = 0.2
            constraints[(i,j)] = Constraint(score=score, dx=dx, dy=dy, modeled=True, error=error)

        self.debug('Added', len(constraints) - start_size, 'calculated constraints using stage model')

        if return_constraints:
            return constraints

    def inspect_constraint(self, pair, save_path=None):
        image1, image2 = self.images[pair[0]], self.images[pair[1]]
        constraint = self.constraints[pair]
        x1, y1 = max(0, -constraint.dx), max(0, -constraint.dy)
        x2, y2 = max(0, constraint.dx), max(0, constraint.dy)
        new_image = np.zeros((2, max(x1 + image1.shape[0], x2 + image2.shape[0]),
                max(y1 + image1.shape[1], y2 + image2.shape[1])), dtype=image1.dtype)
        new_image[0,x1:x1+image1.shape[0],y1:y1+image1.shape[1]] = image1
        new_image[1,x2:x2+image2.shape[0],y2:y2+image2.shape[1]] = image2

        if save_path is None:
            return new_image
        else:
            skimage.io.imsave(save_path, new_image)

    def score_positions(self, pairs=None):
        """ Scores the current position of images using the normalized cross correlation of each overlapping
        image
        """

        if pairs is None:
            pairs = self.find_pairs()

        scores = []
        for i,j in pairs:
            posdiff = self.boxes[j].pos1[:2] - self.boxes[i].pos1[:2]
            scores.append(score_offset(self.images[i], self.images[j], posdiff[0], posdiff[1]))

        return np.array(scores)

    def make_constraint_matrix(self, constraints=None):
        constraints = constraints or self.constraints

        solution_mat = np.zeros((len(constraints)*2+2, len(self.images)*2))
        solution_vals = np.zeros(len(constraints)*2+2)
        
        for index, ((id1, id2), constraint) in enumerate(constraints.items()):
            dx, dy = constraint.dx, constraint.dy
            score = max(0, constraint.score)
            score = 0.1 + score
            #score = 0.5 + score
            #score = 0.5 + constraint.score * constraint.score
            #score = max(0.000001, score)

            solution_mat[index*2, id1*2] = -score
            solution_mat[index*2, id2*2] = score
            solution_vals[index*2] = score * dx

            solution_mat[index*2+1, id1*2+1] = -score
            solution_mat[index*2+1, id2*2+1] = score
            solution_vals[index*2+1] = score * dy

        # anchor tile 0 to 0,0, otherwise there are inf solutions
        solution_mat[-2, 0] = 1
        solution_mat[-1, 1] = 1

        return solution_mat, solution_vals

    def solve_constraints(self, constraints=None, solver=None, apply_positions=True, ignore_bad_constraints=True, filter_outliers=True):
        """ Solves all contained constraints to get absolute image positions.

        This is done by constructing a set of linear equations, with every constraint
        being an equation of the positions of the two images.
        Scores are incorporated by multiplying the whole equation by the score value,
        as the solution with the least squared error is found, prioritizing solution
        that use highly scored constraints.

        Args:
            apply_positions: (bool)
                Whether the solved positions should be applied to the images in the composite.
                If True the current image positions are overwritten.

            Returns: np.ndarray shape (len(images), n_dims)
                The solved positions of images.
        """
        solver = solver or solving.LinearSolver()
        constraints = constraints or self.constraints.copy()
        result = solver.solve(constraints, dict(zip(range(len(self.boxes)), self.boxes.pos1)))

        if type(result) == tuple and len(result) == 2:
            poses, constraints = result
            if filter_outliers:
                self.constraints = constraints
        else:
            poses = result

        diffs = []
        for (id1, id2), constraint in constraints.items():
            new_offset = poses[id2] - poses[id1]
            diffs.append((new_offset[0] - constraint.dx, new_offset[1] - constraint.dy))

        diffs = np.abs(np.array(diffs))

        self.debug("Solved", len(constraints), "constraints, with error: min {} max".format(
                np.percentile(diffs, (0,1,5,50,95,99,100)).astype(int)))

        if diffs.max() > 50:
            if ignore_bad_constraints:
                warnings.warn(("Final solution has some constraints that are off by more than 50px. "
                        "This usually means that some erronious constraints were still present before "
                        "solving. Make sure you performed all proper filtering steps before solving."))
            else:
                raise ValueError(("Final solution has some constraints that are off by more than 50px. "
                        "This usually means that some erronious constraints were still present before "
                        "solving. Make sure you performed all proper filtering steps before solving."))

        if apply_positions:
            #self.boxes.pos2[:] = self.boxes.size()
            #self.boxes.pos1[:] = 0
            for i in poses.keys():
                self.boxes[i].pos2[:2] = poses[i] + self.boxes[i].size()[:2]
                self.boxes[i].pos1[:2] = poses[i]
            #for i, box in enumerate(self.boxes):
                #box.pos2[:2] = poses[i] + box.pos2[:2] - box.pos1[:2]
                #box.pos1[:2] = poses[i]

        self.constraints = constraints

        return self.boxes.pos1


    def solve_constraints_old(self, apply_positions=True, filter_outliers=False, max_outlier_ratio=0.75, outlier_threshold=5,
                replace_with_modeled=False, ignore_bad_constraints=False, scores_plot_path=None, use_outiler_model=False):
        """ Solves all contained constraints to get absolute image positions.

        This is done by constructing a set of linear equations, with every constraint
        being an equation of the positions of the two images.
        Scores are incorporated by multiplying the whole equation by the score value,
        as the solution with the least squared error is found, prioritizing solution
        that use highly scored constraints.

        Args:
            apply_positions: (bool)
                Whether the solved positions should be applied to the images in the composite.
                If True the current image positions are overwritten.

            Returns: np.ndarray shape (len(images), n_dims)
                The solved positions of images.
        """


        if filter_outliers:
            print ('outlier')
            constraints = self.constraints.copy()

            i = 0
            while True:
                solution_mat, solution_vals = self.make_constraint_matrix(constraints)
                print ('start')
                begin = time.time()
                solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
                print ('end', time.time() - begin)
                poses = solution.reshape(-1,2)

                max_consts = {}
                dists = []
                for index, ((id1, id2), constraint) in enumerate(constraints.items()):
                    new_offset = poses[id2] - poses[id1]
                    diff = (new_offset[0] - constraint.dx, new_offset[1] - constraint.dy)
                    dist = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
                    dists.append(dist)
                    
                    if max_consts.get(id1, (0,0))[0] < dist:
                        max_consts[id1] = (dist, (id1, id2))
                    if max_consts.get(id2, (0,0))[0] < dist:
                        max_consts[id2] = (dist, (id1, id2))
                    
                dists = np.array(dists)
                
                self.debug('dists', np.percentile(dists, (0,1,5,50,95,99,100)).astype(int))

                max_dist = dists.max()
                if max_dist < outlier_threshold:
                    self.constraints = constraints
                    break

                if len(constraints) < len(self.constraints) * max_outlier_ratio:
                    break

                pairs = []
                self.debug ('before filter', len(constraints))
                for dist, (id1, id2) in max_consts.values():
                    if dist >= max(outlier_threshold, max_dist * 0.2) and dist >= max_consts[id1][0] and dist >= max_consts[id1][0]:
                        if (id1, id2) in constraints and (not replace_with_modeled or not constraints[id1,id2].modeled):
                            #self.debug ('del', id1, id2)
                            del constraints[(id1, id2)]
                            pairs.append((id1, id2))
                self.debug ('after filter', len(constraints))

                if len(pairs) == 0:
                    break

                if replace_with_modeled:
                    new_consts = self.model_constraints(pairs, return_constraints=True)
                    constraints.update(new_consts)

                if scores_plot_path:
                    tmp = self.constraints
                    tmp2 = self.boxes
                    poses = np.round(solution.reshape(-1,2)).astype(int)
                    poses -= poses.min(axis=0).reshape(1,2)
                    self.boxes = BBoxList(poses, poses + self.boxes.size())
                    self.constraints = constraints
                    self.plot_scores(scores_plot_path.format(i), score_func=self.constraint_accuracy)
                    self.constraints = tmp
                    self.boxes = tmp2
                i += 1

        #if use_outiler_model:
            #solution_mat, solution_vals = self.make_constraint_matrix()
            #huber = sklearn.linear_model.HuberRegressor(max_iter=10000, fit_intercept=False).fit(solution_mat, solution_vals)
            #print (huber.coef_)
            #print (huber.scale_)
            #print (huber.coef_.shape)
            #solution = huber.coef_
            #solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
        else:
            solution_mat, solution_vals = self.make_constraint_matrix()
            solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)

        poses = np.round(solution.reshape(-1,2)).astype(int)
        poses -= poses.min(axis=0).reshape(1,2)

        diffs = []
        for index, ((id1, id2), constraint) in enumerate(self.constraints.items()):
            new_offset = poses[id2] - poses[id1]
            diffs.append((new_offset[0] - constraint.dx, new_offset[1] - constraint.dy))

        diffs = np.abs(np.array(diffs))

        self.debug("Solved", len(self.constraints), "constraints, with error: min {} max".format(
                np.percentile(diffs, (0,1,5,50,95,99,100)).astype(int)))

        if diffs.max() > 50:
            if ignore_bad_constraints:
                warnings.warn(("Final solution has some constraints that are off by more than 50px. "
                        "This usually means that some erronious constraints were still present before "
                        "solving. Make sure you performed all proper filtering steps before solving."))
            else:
                raise ValueError(("Final solution has some constraints that are off by more than 50px. "
                        "This usually means that some erronious constraints were still present before "
                        "solving. Make sure you performed all proper filtering steps before solving."))

        if apply_positions:
            for i, box in enumerate(self.boxes):
                box.pos2[:2] = poses[i] + box.pos2[:2] - box.pos1[:2]
                box.pos1[:2] = poses[i]
        return poses


    def stitch_images(self, indices=None, real_images=None, out=None, bg_value=None, return_bg_mask=False,
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

        example_image = self.fullimagearr(real_images[0])
        
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
            image = self.fullimagearr(real_images[i])

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

        fig.savefig('plots/merger_locations.png')

        if bg_value is not None:
            full_image[mask] = bg_value
        
        if return_bg_mask:
            return full_image, mask
        return full_image


    def plot_scores(self, path, score_func=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches

        groups = [list(range(len(self.boxes)))]
        const_groups = [self.constraints.keys()]
        names = ['']
        if score_func == 'accuracy':
            score_func = self.constraint_accuracy

        if self.boxes[0].pos1.shape[0] == 3:
            groups = []
            const_groups = []
            names = []

            vals = sorted(set(box.pos1[2] for box in self.boxes))
            print (vals)
            for val in vals:
                groups.append([i for i in range(len(self.boxes)) if self.boxes[i].pos1[2] == val])
                const_groups.append([(i,j) for (i,j) in self.constraints if self.boxes[i].pos1[2] == val and self.boxes[j].pos1[2] == val])
                names.append('(plane z={})'.format(val))

            new_const_groups = {}
            for i,j in self.constraints:
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
                width, height = self.boxes[index].size()[:2]
                axis.text(y + height / 2, -x - width / 2, "{}\n({})".format(index, i), horizontalalignment='center', verticalalignment='center')
                axis.add_patch(matplotlib.patches.Rectangle((y, -x - width), height, width, edgecolor='grey', facecolor='none'))

            poses = []
            colors = []
            sizes = []
            #for (i,j), constraint in self.constraints.items():
                #if i not in indices or j not in indices: continue
            for i,j in const_pairs:
                constraint = self.constraints[(i,j)]

                pos1, pos2 = self.boxes[i].center()[:2], self.boxes[j].center()[:2]
                #if np.all(pos1 == pos2):
                    #print (i, j, constraint)
                pos = np.mean((pos1, pos2), axis=0)
                poses.append((pos[1], -pos[0]))
                score = constraint.score if score_func is None else score_func(i, j, constraint)
                colors.append(score)
                sizes.append(50 if constraint.modeled else 200)
                axis.arrow(pos[1] - constraint.dy/2, -pos[0] + constraint.dx/2, constraint.dy/1, -constraint.dx/1,
                        width=5, head_width=20, length_includes_head=True, color='black')
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
                "width": str(box.size()[0]), "height": str(box.size()[1]),
                "stroke": 'black', "fill": 'transparent',
                "stroke-width": str(int(box.size()[:2].min()) // 10),
            })

            text = ET.SubElement(group, 'text', attrib={
                "class": "show-hover",
                "x": str(box.pos1[0] + box.size()[0] // 2),
                "y": str(box.pos1[1] + box.size()[1] * 2 // 3),
                "font-size": str(box.size()[1] // 2),
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
                "stroke-width": str(min(box1.size()[:2].min(), box2.size()[:2].min()) // 2),
                "stroke-linecap": "round",
            })

            line = ET.SubElement(group, 'line', attrib={
                "class": "show-hover",
                "x1": str(box1.pos1[0] + box1.size()[0] / 2),
                "y1": str(box1.pos1[1] + box1.size()[1] / 2),
                "x2": str(box2.pos1[0] + box2.size()[0] / 2),
                "y2": str(box2.pos1[1] + box2.size()[1] / 2),
                "stroke": "black",
                "stroke-width": str(min(box1.size()[:2].min(), box2.size()[:2].min()) // 40),
            })

            #"""
            rect = ET.SubElement(group, 'rect', attrib={
                "class": "show-hover",
                "x": str(box1.pos1[0]), "y": str(box1.pos1[1]),
                "width": str(box1.size()[0]), "height": str(box1.size()[1]),
                "stroke": 'black', "fill": 'transparent',
                "stroke-width": str(int(box1.size().mean()) // 10),
            })
            text = ET.SubElement(group, 'text', attrib={
                "class": "show-hover",
                "x": str(box1.pos1[0] if box1.pos1[0] <= box2.pos1[0] else box1.pos2[0]),
                "y": str(box1.pos1[1] + box1.size()[1] * 2 // 3),
                "font-size": str(box1.size()[1] // 2),
                "text-anchor": "end" if box1.pos1[0] <= box2.pos1[0] else "start",
            })
            text.text = str(i)

            rect = ET.SubElement(group, 'rect', attrib={
                "class": "show-hover",
                "x": str(box2.pos1[0]), "y": str(box2.pos1[1]),
                "width": str(box2.size()[0]), "height": str(box2.size()[1]),
                "stroke": 'black', "fill": 'transparent',
                "stroke-width": str(int(box2.size().mean()) // 10),
            })
            text = ET.SubElement(group, 'text', attrib={
                "class": "show-hover",
                "x": str(box2.pos2[0] if box1.pos1[0] <= box2.pos1[0] else box2.pos1[0]),
                "y": str(box2.pos1[1] + box2.size()[1] * 2 // 3),
                "font-size": str(box2.size()[1] // 2),
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
    
    def constraint_accuracy(self, i, j, constraint):
        new_offset = self.boxes[j].pos1[:2] - self.boxes[i].pos1[:2]
        diff = (new_offset[0] - constraint.dx, new_offset[1] - constraint.dy)
        return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])


def align_job(aligner, image1, image2, shape1=None, shape2=None, precalc1=None, precalc2=None, previous_constraint=None):
    return aligner.align(image1=image1, image2=image2, shape1=shape1, shape2=shape2, precalc1=precalc1, precalc2=precalc2, previous_constraint=previous_constraint)

def precalc_job(aligner, image, shape=None):
    return aligner.precalculate(image, shape=shape)

