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

from .estimate_translation import calculate_offset, score_offset
from .stage_model import SimpleOffsetModel, GlobalStageModel
from . import merging, estimate_translation
from .. import utils


@dataclasses.dataclass
class Constraint:
    score: float
    dx: int
    dy: int
    modeled: bool = False

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


class SequentialExecutor(concurrent.futures.Executor):
    def submit(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        future = concurrent.futures.Future()
        future.set_result(result)
        return future


class CompositeImage:
    def __init__(self, aligner=None, precalculate=False, debug=True, progress=False, executor=None):
        self.images = []
        self.greyimages = []
        self.boxes = BBoxList()
        self.constraints = {}
        self.scale = 1
        self.stage_model = None
        self.set_logging(debug, progress)
        self.set_executor(executor)
        self.executor = executor

        self.aligner = aligner or estimate_translation.FFTAligner()
        self.precalculate = precalculate

    def set_executor(self, executor):
        self.executor = executor or SequentialExecutor()

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
        """ Saves this composite to the given path. Can be restored with the `CompositeImage.load`
        function.

            save_images: bool, whether the images should be included in the save file. If they are
                excluded the programmer needs to restore them when loading.
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
        """ Loads a composite previously saved with `CompositeImage.save` to the given
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

    def add_images(self, images, positions=None, boxes=None, scale='pixel', imagescale=1):
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
        positions = np.asarray(positions)
        #assert len(self.imageshape(images[0])) == 2, "Only 2d images are supported"
        assert positions is not None or boxes is not None, "Must specify positions or boxes"

        n_dims = positions.shape[1]
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
        
        self.images.extend(images)

    def add_split_image(self, image, num_tiles=None, tile_shape=None, overlap=0.1):
        """ Adds an image split into a number of tiles

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
        self.add_images(images, positions)
    
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
                    constraint.score, constraint.dx * scale_conversion, constraint.dy * scale_conversion, constraint.modeled)

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
                output of functions like `CompositeImage.stitch_images` or when merging composites together.
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
        that don't already have a constraint.

        Returns (np.ndarray shape (N, 2)):
            The sequence of pairs of indices of the images that overlap without constraints.
        """
        pairs = self.find_pairs(*args, **kwargs)
        mask = [(i,j) not in self.constraints for i,j in pairs]
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

        composite.calc_constraints(pairs, num_peaks=2)
        self.debug('cacled constraints')
        composite.filter_constraints()
        self.debug('filtered')
        composite.filter_outliers()
        self.debug('filtered out')
        composite.plot_scores('plots/tmp_constraints.png')
        composite.save('selected_composite.bin', save_images=False)

        return composite.constraints.keys()

    def calc_constraints(self, pairs=None, precalculate=False, return_constraints=False, debug=True, **kwargs):
        """ Estimates the pairwise translations of images and add thems as constraints
        in the montage

        Args:
            pairs (sequence of tuples, optional): optional.
                The indices of image pairs to add constraints to. Defaults to
                all images that overlap or are adjacent based on the estimated
                positions that don't already have constraints, see 
                `CompositeImage.find_unconstrained_pairs` for more info.
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
            precalcs = [self.executor.submit(precalc_job, aligner=aligner, image=image) for image in self.images]
            precalcs = [future.result() for future in self.progress(precalcs)]
        else:
            precalcs = [None] * len(self.images)

        futures = []
        for index1, index2 in pairs:
            futures.append(self.executor.submit(align_job,
                aligner = self.aligner,
                image1 = self.images[index1], image2 = self.images[index2],
                precalc1 = precalcs[index1], precalc2 = precalcs[index2],
            ))

        for (index1, index2), future in self.progress(zip(pairs, futures), total=len(futures)):
            score, dx, dy = future.result()
            constraints[(index1,index2)] = Constraint(score, dx, dy)

        if debug and len(constraints) != 0:
            scores = np.array([const.score for const in constraints.values()])
            self.debug("Calculated {} constraints, score values: min {} mean {} max {}".format(len(scores), scores.min(), scores.mean(), scores.max()))

        if return_constraints:
            return constraints
        else:
            self.constraints.update(constraints)

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
        scores = np.array([const.score for const in real_consts] + [const.score for const in fake_consts])

        thresh = max(const.score for const in fake_consts)
        print (thresh, 'thresh')
        return thresh

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

        self.debug("Estimated stage model", model, "with an r2 score of", model.score(est_poses, const_poses))

        if (model.predict([[0] * est_poses.shape[1]]).max() > const_poses.max() * 100 or 
                model.predict([[1] * est_poses.shape[1]]).max() > const_poses.max() * 100):
            warnings.warn("Stage model is predicting very large values for simple constraints,"
                " it may have hyperoptimized to the training data.")

        self.stage_model = model

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

    def model_constraints(self, pairs=None, score_multiplier=0.5, return_constraints=False):
        """ Uses the stored stage model (estimated by estimage_stage_model) to
        fill the specified constraints.
            
        Args:
            pairs (sequence of (i,j)): optional
                The indices of image pairs to add constraints to. Defaults to
                all images that overlap or are adjacent based on the estimated
                positions that don't already have constraints, see 
                `CompositeImage.find_unconstrained_pairs` for more info.
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

        constraints = {} if return_constraints else self.constraints

        start_size = len(constraints)
        for i,j in pairs:
            dx, dy = self.stage_model.predict(np.array([self.boxes[i].pos1, self.boxes[j].pos1]).reshape(1,-1)).flatten().astype(int)
            #dx, dy = self.stage_model.predict((self.boxes[j].pos1 - self.boxes[i].pos1).reshape(1,-1)).flatten().astype(int)
            assert (max(abs(dx), abs(dy)) <= max(self.boxes[i].size()[:2])), (
                "Image offset from stage model does not contain any overlap."
                " The stage model may not have correctly modeled the movement")
            score = score_offset(self.images[i], self.images[j], dx, dy) * score_multiplier
            constraints[(i,j)] = Constraint(score, dx, dy, modeled=True)

        self.debug('Added', len(constraints) - start_size, 'calculated constraints using stage model')

        if return_constraints:
            return constraints

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

    def make_constraint_matrix(self):
        solution_mat = np.zeros((len(self.constraints)*2+2, len(self.images)*2))
        solution_vals = np.zeros(len(self.constraints)*2+2)
        
        for index, ((id1, id2), constraint) in enumerate(self.constraints.items()):
            dx, dy = constraint.dx, constraint.dy
            score = max(0, constraint.score)
            score = constraint.score * constraint.score
            score = max(0.000001, score)

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

    def solve_constraints(self, apply_positions=True, filter_outliers=False, outlier_threshold=5):
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

            while True:
                solution_mat, solution_vals = self.make_constraint_matrix()
                solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
                poses = solution.reshape(-1,2)

                max_consts = {}
                dists = []
                for index, ((id1, id2), constraint) in enumerate(self.constraints.items()):
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
                    break

                self.debug ('before filter', len(self.constraints))
                for dist, (id1, id2) in max_consts.values():
                    if dist >= max(outlier_threshold, max_dist * 0.2) and dist >= max_consts[id1][0] and dist >= max_consts[id1][0]:
                        if (id1, id2) in self.constraints:
                            self.debug ('del', id1, id2)
                            del self.constraints[(id1, id2)]
                self.debug ('after filter', len(self.constraints))

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
            warnings.warn(("Final solution has some constraints that are off by more than 50px. "
                    "This usually means that some erronious constraints were still present before "
                    "solving. Make sure you performed all proper filtering steps before solving."))

        if apply_positions:
            for i, box in enumerate(self.boxes):
                box.pos2[:2] = poses[i] + box.pos2[:2] - box.pos1[:2]
                box.pos1[:2] = poses[i]
        return poses


    def stitch_images(self, indices=None, real_images=None, out=None, bg_value=None, return_bg_mask=False,
            mins=None, maxes=None, keep_zero=False, merger=merging.MeanMerger):
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

        if type(indices[0]) == bool:
            indices = [i for i in range(len(indices)) if indices[i]]

        if keep_zero:
            mins = 0

        self.debug(mins, maxes)
        start_mins = np.array(self.boxes[0].pos1[:2])
        start_maxes = np.array(self.boxes[0].pos2[:2])
        self.debug(start_mins, start_maxes)
        
        if mins is not None:
            start_mins[:] = mins
        if maxes is not None:
            start_maxes[:] = maxes
        self.debug(start_mins, start_maxes)

        mins, maxes = start_mins, start_maxes

        self.debug (mins, maxes)

        if keep_zero:
            mins = np.zeros_like(mins)
        
        if real_images is None:
            real_images = self.images

        example_image = self.fullimagearr(real_images[0])
        self.debug(example_image.shape, 'example shape')
        full_shape = tuple((maxes - mins) * self.scale) + example_image.shape[2:]
        merger = merger(full_shape, example_image.dtype)
        if out is not None:
            assert merger.image.shape == out.shape and merger.image.dtype == out.dtype, (
                "Provided output image does not match expected shape or dtype: {} {}".format(merger.image.shape, merger.image.dtype))
            merger.image = out

        for i in indices:
            pos1 = ((self.boxes[i].pos1[:2] - mins) * self.scale).astype(int)
            pos2 = ((self.boxes[i].pos2[:2] - mins) * self.scale).astype(int)
            image = self.fullimagearr(real_images[i])
            #self.debug(image.shape)
            if np.any(pos2 - pos1 != image.shape[:2]):
                self.debug(pos1, pos2, pos2 - pos1)
                self.debug(image.shape)
                warnings.warn("resizing some images")
                image = skimage.transform.resize(image, pos2 - pos1)
            
            image = image[max(0,-pos1[0]):,max(0,-pos1[1]):]
            #self.debug(pos1, pos2, image.shape)
            pos1 = np.maximum(0, np.minimum(pos1, maxes - mins))
            pos2 = np.maximum(0, np.minimum(pos2, maxes - mins))
            #self.debug(pos1, pos2, image.shape)
            #image = image[max(0,-pos1[0]):,max(0,-pos1[1]):]
            #self.debug(pos1, pos2, image.shape)
            #self.debug(max(0,-pos1[0]),max(0,-pos1[1]))
            image = image[:pos2[0]-pos1[0],:pos2[1]-pos1[1]]
            #self.debug(pos1, pos2, image.shape)
            if image.size == 0: continue
            self.debug(self.boxes[i])
            self.debug((self.boxes[i].pos1[:2] - mins) * self.scale)
            self.debug((self.boxes[i].pos2[:2] - mins) * self.scale)
            self.debug(image.shape, pos1, pos2, mins, maxes)

            position = (slice(pos1[0], pos2[0]), slice(pos1[1], pos2[1]))
            merger.add_image(image, position)

        full_image, mask = merger.final_image()

        if bg_value is not None:
            full_image[mask] = bg_value
        
        if return_bg_mask:
            return full_image, mask
        return full_image


    def plot_scores(self, path, score_func=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches

            

        groups = [list(range(len(self.boxes)))]
        const_groups = self.constraints.keys()
        names = ['']
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
        grid_size = int(np.sqrt(len(groups))) + 1
        fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(axis_size*grid_size,axis_size*grid_size), squeeze=False, sharex=True, sharey=True)

        for indices, const_pairs, axis, name in zip(groups, const_groups, axes.flatten(), names):
            print (name)

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

            points = axis.scatter(poses[:,0], poses[:,1], c=colors, s=sizes, alpha=0.5)
            fig.colorbar(points, ax=axis)

            axis.set_title('Scores of constraints ' + name)

        fig.savefig(path)


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


def align_job(aligner, image1, image2, precalc1=None, precalc2=None):
    return aligner.align(image1=image1, image2=image2, precalc1=precalc1, precalc2=precalc2)

def precalc_job(aligner, image):
    return aligner.precalculate(image)

