import collections
import math
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters
import sklearn.linear_model
import sklearn.mixture
import imageio.v3 as iio
import warnings

from .estimate_translation import calculate_offset, score_offset
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


class CompositeImage:
    def __init__(self, precalculate_fft=False, debug=True, progress=False, executor=None):
        self.images = []
        self.boxes = []
        self.constraints = {}
        self.scale = 1
        self.stage_model = None
        self.set_logging(debug, progress)
        self.executor = executor

        self.precalculate_fft = precalculate_fft
        if precalculate_fft:
            self.ffts = []

    def set_logging(self, debug=True, progress=False):
        self.debug, self.progress = utils.log_env(debug, progress)

    def print_mem_usage(self):
        mem_images = sum((image.nbytes if type(image) == np.ndarray else 0) for image in self.images)
        self.debug("Using {} bytes ({}) for images".format(mem_images, utils.human_readable(mem_images)))
        if self.precalculate_fft:
            mem_ffts = sum(image.nbytes for image in self.ffts)
            self.debug("Using {} bytes ({}) for ffts".format(mem_ffts, utils.human_readable(mem_ffts)))
            mem_images += mem_ffts
        self.debug("Total: {} ({})".format(mem_images, utils.human_readable(mem_images)))
    
    def imagearr(self, image):
        if type(image) == str:
            return iio.imread(image)
        return image

    def imageshape(self, image):
        if type(image) == str:
            return iio.improps(image).shape
        return image.shape

    def add_images(self, images, positions, scale='pixel'):
        """ Adds images to the composite

            images: np.ndarray shape (N, W, H) or list of N np.ndarrays shape (W, H) or list of strings
                The images that will be stitched together. Can pass a list of
                paths that will be opened by imageio.v3.imread when needed.
                Passing paths will require less memory as images are not stored,
                but will increase computation time.

            positions: np.ndarray shape (N, D)
                Specifies the extimated positions of each image. The approx values are
                used to decide which images are overlapping. These values are interpreted
                using the scale argument, default they are pixel values.

            scale: 'pixel', 'tile', float, or tuple length D of any previous values
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
        assert len(self.imageshape(images[0])) == 2, "Only 2d images are supported"
        assert len(self.imageshape(images[0])) == positions.shape[1]

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

        for i in range(len(images)):
            self.boxes.append(BBox(
                positions[i] * scale,
                positions[i] * scale + np.array(self.imageshape(images[i])) * self.scale
            ))
        
        self.images.extend(images)

        if self.precalculate_fft:
            self.debug("Precalculating ffts of", len(images), "images")
            if self.executor is not None:
                self.ffts.extend(self.progress(self.executor.map(fft_job, images), total=len(images)))
            else:
                self.ffts.extend(self.progress(map(fft_job, images), total=len(images)))

    def merge(self, other_composite):
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

        for image, box in zip(other_composite.images, other_composite.boxes):
            self.images.append(image)
            self.boxes.append(BBox(box.pos1 * scale_conversion, box.pos2 * scale_conversion))

        if self.precalculate_fft:
            if other_composite.precalculate_fft:
                self.ffts.extend(other_composite.ffts)
            else:
                if self.executor is not None:
                    self.ffts.extend(self.progress(self.executor.map(np.fft.fft2, other_composite.images), total=len(other_composite.images)))
                else:
                    self.ffts.extend(self.progress(map(np.fft.fft2, other_composite.images), total=len(other_composite.images)))

        for (i,j), constraint in other_composite.constraints.items():
            self.constraints[(i+start_index,j+start_index)] = Constraint(
                    constraint.score, constraint.dx * scale_conversion, constraint.dy * scale_conversion, constraint.modeled)

        return list(range(start_index, len(self.images)))

    def set_scale(self, scale_factor):
        """ Sets the scale factor of the composite. Normally this doesn't need to be changed,
        however if you are trying to stich together images taken at different magnifications you
        may need to modify the scale factor.
            
            scale_factor: float or int
                Scale of images in this composite, as a multiplier. Eg a scale_factor of
                10 will result in each pixel in images corresponding to 10 pixels in the
                output of functions like stitch_images or when merging composites together.
        """
        self.scale = scale_factor

    def find_pairs(self, needs_overlap=False):
        """ Finds all pairs of images that overlap, based on the estimated positions

            Returns: np.ndarray shape (N, 2)
                The sequence of pairs of indices of the images that overlap.
        """
        pairs = []
        if needs_overlap:
            for i in range(len(self.images)):
                for j in range(i+1,len(self.images)):
                    if self.boxes[i].overlaps(self.boxes[j]):
                        pairs.append((i,j))
        else:
            for i in range(len(self.images)):
                for j in range(i+1,len(self.images)):
                    if self.boxes[i].collides(self.boxes[j]):
                        pairs.append((i,j))
        return np.array(pairs)

    def find_unconstrained_pairs(self, needs_overlap=False):
        """ Finds all pairs of images that overlap based on the estimated positions and
        that don't already have a constraint.

        Returns (np.ndarray shape (N, 2)):
            The sequence of pairs of indices of the images that overlap without constraints.
        """
        pairs = self.find_pairs(needs_overlap=needs_overlap)
        mask = [(i,j) not in self.constraints for i,j in pairs]
        return pairs[mask]

    def calc_constraints(self, pairs=None, return_constraints=False, debug=True):
        """ Estimates the pairwise translations of images and add thems as constraints
        in the montage

        Args:
            pairs (sequence of (i,j): optional
                The indices of image pairs to add constraints to. Defaults to
                all images that overlap or are adjacent based on the estimated
                positions that don't already have constraints.
                Important: the pairs given are not checked for overlap, so invalid
                constraints could be generated if specific indices are passed in.

            return_constraints (bool): default False
                If true returns constraints instead of adding them to the montage

        Returns (dict or None):
            if return_constraints is True, a dict of the calculated constraints
            is returned, otherwise nothing.
        """
        if pairs is None:
            pairs = self.find_unconstrained_pairs()

        constraints = {}

        if self.executor is not None:
            futures = []
            if self.precalculate_fft:
                for index1, index2 in pairs:
                    futures.append(self.executor.submit(calculate_offset,
                            self.images[index1], self.images[index2],
                            self.ffts[index1], self.ffts[index2]))
            else:
                for index1, index2 in pairs:
                    futures.append(self.executor.submit(calculate_offset, self.images[index1], self.images[index2]))

            for (index1, index2), future in self.progress(zip(pairs, futures), total=len(futures)):
                score, dx, dy = future.result()
                constraints[(index1,index2)] = Constraint(score, dx, dy)
        else:
            for index1, index2 in self.progress(pairs):
                score, dx, dy = calculate_offset(self.images[index1], self.images[index2])
                constraints[(index1,index2)] = Constraint(score, dx, dy)

        if debug:
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
                The number of fake constraints to be generated, defaults to 0.25*len(images)
                In general the more samples the better the estimate, at the expense of speed

            random_state (int):
                Used as a seed to get reproducible results

        Returns (float):
            threshold for score where all scores lower are likely to be bad constraints
        """
        num_samples = num_samples or max(len(self.images) // 4, 10)
        rng = np.random.default_rng(random_state)
        
        real_consts = rng.choice([const for const in self.constraints.values() if not const.modeled], size=num_samples)
        fake_pairs = []

        for i in rng.permutation(len(self.images)):
            for j in rng.permutation(len(self.images)):
                if ((i,j) not in self.constraints
                        and np.any(np.abs(self.boxes[i].pos1 - self.boxes[j].pos1)
                            > np.abs(self.boxes[i].pos2 - self.boxes[i].pos1) * 1.5)):
                    fake_pairs.append((i,j))

                if len(fake_pairs) == len(real_consts): break
            if len(fake_pairs) == len(real_consts): break

        fake_consts = self.calc_constraints(fake_pairs, return_constraints=True, debug=False).values()
        scores = np.array([const.score for const in real_consts] + [const.score for const in fake_consts])

        #fig, axis = plt.subplots()
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

    def estimate_stage_model(self, model=None, filter_outliers=False, random_state=12345):
        """ Estimages a stage model that translates between estimated
        position differences and pairwise alignment differences.
            
        Args:
            model (sklearn model instance):
                Used as the model to estimate the stage model. fit is called on it
                with the estimated offsets as X and the offset from constraints as y.
                Defaults to LinearRegression if filter_outliers is False and
                RANSACRegressor if filter_outliers is True

            random_state (int):
                Passed to the RANSACRegressor to produce reproducible results
        """
        if filter_outliers:
            model = model or sklearn.linear_model.RANSACRegressor(random_state=random_state)
        else:
            model = model or sklearn.linear_model.LinearRegression()

        est_poses = []
        const_poses = []
        indices = []
        for (i,j), constraint in self.constraints.items():
            if not constraint.modeled:
                #est_poses.append(self.boxes[j].pos1 - self.boxes[i].pos1)
                est_poses.append(np.concatenate([self.boxes[i].pos1, self.boxes[j].pos1]))
                const_poses.append((constraint.dx, constraint.dy))
                indices.append((i,j))

        est_poses, const_poses = np.array(est_poses), np.array(const_poses)
        indices = np.array(indices)

        model.fit(est_poses, const_poses)

        self.debug("Estimated stage model with an r2 score of", model.score(est_poses, const_poses))

        if filter_outliers:
            self.debug('Filtered out', np.sum(~model.inlier_mask_), 'constraints as outliers')
            for (i,j) in indices[~model.inlier_mask_]:
                del self.constraints[(i,j)]

        if (model.predict([[0] * est_poses.shape[1]]).max() > const_poses.max() * 100 or 
                model.predict([[1] * est_poses.shape[1]]).max() > const_poses.max() * 100):
            warnings.warn("Stage model is predicting very large values for simple constraints,"
                " it may have hyperoptimized to the training data.")

        self.stage_model = model

    def model_constraints(self, pairs=None, score_multiplier=0.5, return_constraints=False):
        """ Uses the stored stage model (estimated by estimage_stage_model) to
        fill the specified constraints.
            
        Args:
            pairs (sequence of (i,j)): optional
                The indices of image pairs to add constraints to. Defaults to
                all images that overlap or are adjacent based on the estimated
                positions that don't already have constraints.
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
            assert (max(abs(dx), abs(dy)) <= max(self.images[i].shape[0], self.images[j].shape[1])), (
                "Image offset from stage model does not contain any overlap."
                " The stage model may not have correctly modeled the movement")
            score = score_offset(self.images[i], self.images[j], dx, dy) * score_multiplier
            constraints[(i,j)] = Constraint(score, dx, dy, modeled=True)

        self.debug('Added', len(constraints) - start_size, 'calculated constraints using stage model')

        if return_constraints:
            return constraints

    def solve_constraints(self, apply_positions=True):
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
        solution_mat = np.zeros((len(self.constraints)*2+2, len(self.images)*2))
        solution_vals = np.zeros(len(self.constraints)*2+2)
        
        for index, ((id1, id2), constraint) in enumerate(self.constraints.items()):
            dx, dy = constraint.dx, constraint.dy
            score = max(0.0000001, constraint.score)

            solution_mat[index*2, id1*2] = -score
            solution_mat[index*2, id2*2] = score
            solution_vals[index*2] = score * dx

            solution_mat[index*2+1, id1*2+1] = -score
            solution_mat[index*2+1, id2*2+1] = score
            solution_vals[index*2+1] = score * dy

        # anchor tile 0 to 0,0, otherwise there are inf solutions
        solution_mat[-2, 0] = 1
        solution_mat[-1, 1] = 1

        solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)

        poses = np.round(solution.reshape(-1,2)).astype(int)
        poses -= poses.min(axis=0).reshape(1,2)

        if apply_positions:
            for i, box in enumerate(self.boxes):
                box.pos2 = poses[i] + box.pos2 - box.pos1
                box.pos1 = poses[i]
        return poses


    def stitch_images(self, indices=None, real_images=None, bg_value=None, return_bg_mask=False, keep_zero=False):
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
                multiple images. Otherwise the min of the positions is subtracted

            Returns: np.ndarray
                image stitched together
        """

        if indices is None:
            indices = list(range(len(self.images)))

        mins = np.array(self.boxes[indices[0]].pos1[:2])
        maxes = np.array(self.boxes[indices[0]].pos2[:2])
        for i in indices[1:]:
            mins = np.minimum(mins, self.boxes[i].pos1[:2])
            maxes = np.maximum(maxes, self.boxes[i].pos2[:2])

        if keep_zero:
            mins = np.zeros_like(mins)
        
        if real_images is None:
            real_images = self.images

        example_image = self.imagearr(real_images[0])
        full_shape = tuple((maxes - mins) * self.scale) + example_image.shape[2:]
        full_image = np.zeros(full_shape, dtype=example_image.dtype)
        counts = np.zeros(full_shape[:2] + (1,) * (len(full_shape)-2), dtype=np.uint8)

        if np.issubdtype(example_image.dtype, int):
            div = lambda x,y: x // y
            castdtype = int
        else:
            div = lambda x,y: x / y
            castdtype = example_image.dtype

        for i in indices:
            pos1 = ((self.boxes[i].pos1 - mins) * self.scale).astype(int)
            pos2 = ((self.boxes[i].pos2 - mins) * self.scale).astype(int)
            image = self.imagearr(real_images[i])
            if np.any(pos2 - pos1 != image.shape[:2]):
                warnings.warn("resizing some images")
                image = skimage.transform.resize(image, pos2 - pos1)

            image_counts = counts[pos1[0]:pos2[0],pos1[1]:pos2[1]]
            cur_image = full_image[pos1[0]:pos2[0],pos1[1]:pos2[1]] 
            if np.issubdtype(example_image.dtype, np.integer):
                cur_image[...] = (cur_image.astype(int) * image_counts + image) // (image_counts + 1)
            else:
                cur_image[...] = (cur_image * image_counts + image) / (image_counts + 1)
            image_counts += 1

        if bg_value is not None:
            full_image[counts!=0] = bg_value
        
        if return_bg_mask:
            return full_image, counts!=0
        return full_image

def fft_job(image):
    if type(image) == str:
        image = iio.imread(image)
    return np.fft.fft2(image)

