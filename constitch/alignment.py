import time
import math
import sys
import numpy as np
import skimage.io
import skimage.registration
import itertools

from .constraints import Constraint

import numba


class Aligner:
    """ Abstract class that defines an algorithm for aligning two images onto each other.
    """
    def precalculate(self, image, box):
        """ Performs an arbitrary precalculation step specific to the alignment algorithm.
        This would be a step in the algorithm that only has to be run once per image,
        and can be cached for each calculation done with the same image. The result of
        this function will be passed to the align function as the precalc1 or precalc2 argument
        """
        pass

    def align(self, constraint, precalc1=None, precalc2=None):
        """ Performs the alignment of two images, finding the pixel offset that best aligns the
        two images. The offset should be from image1 to image2. The return value should be a Constraint
        object, with at least the dx, dy fields filled in to represent the offset of image2 needed
        to align the two images. If the function finds the images don't overlap, None should be returned.
        If a constraint is specified, this method should only return a constraint that fits within the error
        of given constraint, meaning within constraint.error pixels from the (constraint.dx, constraint.dy)
        offset.
        """
        pass

    # Helper functions for subclasses:
    def resize_if_needed(self, image, box=None, downscale_factor=None, padding=None):
        if box is not None and image.shape != tuple(box.size[:2]):
            image = skimage.transform.resize(image, box.size[:2])
        if downscale_factor is not None and downscale_factor != 1:
            image = skimage.transform.downscale_local_mean(image, downscale_factor)
        if padding is not None:
            newimage = np.zeros((image.shape[0] + padding[0], image.shape[1] + padding[1], *image.shape[2:]), image.dtype)
            newimage[:image.shape[0],:image.shape[1]] = image
            image = newimage
        return image
    
    def precalculate_if_needed(self, constraint, precalc1, precalc2):
        if precalc1 is None:
            precalc1 = self.precalculate(constraint.image1, constraint.box1.size)
        if precalc2 is None:
            precalc2 = self.precalculate(constraint.image2, constraint.box2.size)
        return precalc1, precalc2


class FFTAligner(Aligner):
    """ An aligner that uses the phase cross correlation algorithm to estimate the proper alignment of two images.
    The algorithm is the same as described in Kuglin, Charles D. "The phase correlation image alignment method."
    http://boutigny.free.fr/Astronomie/AstroSources/Kuglin-Hines.pdf
    """
    def __init__(self, score_func='zncc', num_peaks=2, precalculate_fft=True, downscale_factor=None, upscale_factor=1,
            allow_edges=True, highpass_filter=0):
        """
            score_func: function or str, default zncc
                The score function used to score the overlapping area. Can be an arbitrary function that
                takes two 2d numpy arrays of the same shape and returns a float, or a string out of the two options:
                    'zncc': zero normalized cross correlation, the mean of both images are subtracted, then each is normalized
                        with the L2 norm and the dot product is calculated. This tends to perform better as it
                        can handle different intensity levels that might occur from unequal illumination.
                    'ncc': normalized cross correlation, each image is normalized with the L2 norm and the dot product is calculated.
                        This does not account for constant offsets while the zncc does.
            num_peaks: int default 2
                Sets the number of peaks in the resulting phase cross correlation image to be checked. Sometimes the highest
                peak is not actually the best offset for alignment so checking multiple peaks can help find the one that
                has the highest ncc score. Increasing this will also increase processing time, possibly dramatically.
            precalculate_fft: bool default True
                Whether or not the FFT of the images should be precalculated and cached or be calculated every time.
                The FFT is able to be precalculated however it does take up a large amount of memory, so if memory
                is a limiting factor setting this to False can help
            downscale_factor: int, optional
                This algorithm is rather slow when calculating large numbers of constraints, and one way to improve
                speed is by downscaling the images before running the algorithm. This will improve runtime at the expense
                of precision, the constraints calculated will have a nonzero error value. Also, if the downscale factor
                is large enough the algorithm can begin to fail and not find any overlap, in general the largest recommended
                value is around 32, but it depends on how large the features in your images are.
            upscale_factor: int, optional
            allow_edges: bool, default True
                Whether offsets in only one dimention are allowed, specifically whether peaks found on the edges 
                of the cross power spectrum are accepted. Edge artifacts can cause erroneous constraints to be
                found, and disallowing them can improve the quality of calculated constraints. Of course if the
                offset actually lies on the edge this will prevent it from being found, however it would probably
                find an offset that is off by 1px.
            highpass_filter: int, default 0
                The cutoff frequency for a high pass filter applied to the images. This can be beneficial as
        """
        if type(score_func) == str:
            score_func = score_func.lower()
            if score_func == 'zncc':
                score_func = zncc_fast
            elif score_func == 'ncc':
                score_func = ncc_fast
            else:
                raise ValueError("Value of score_func '{}' not one of ('zncc', 'ncc')".format(score_func[:100]))

        self.score_func = score_func
        self.num_peaks = num_peaks
        self.precalculate_fft = precalculate_fft
        self.downscale_factor = downscale_factor
        self.upscale_factor = upscale_factor
        self.allow_edges = allow_edges
        self.highpass_filter = highpass_filter

    def precalculate(self, image, box=None):
        if len(image.shape) == 3: image = image[:,:,0]
        image = self.resize_if_needed(image, box, downscale_factor=self.downscale_factor)
        fft = None if not self.precalculate_fft else np.fft.fft2(image, axes=(0,1))
        return image, fft

    def align(self, constraint, precalc1=None, precalc2=None):
        return self.align_full(constraint, precalc1, precalc2)
        section1, section2 = constraint.section1, constraint.section2

        orig_section1, orig_section2 = section1, section2
        if section1.shape != section2.shape:
            section1, section2 = image_diff_sizes(section1, section2)

        fft1 = np.fft.fft2(section1, axes=(0,1))
        fft2 = np.fft.fft2(section2, axes=(0,1))

        fft = calc_pcm(fft1.reshape(-1), fft2.reshape(-1)).reshape(fft1.shape)
        fft = np.fft.ifft2(fft, axes=(0,1)).real
        if len(fft.shape) == 3:
            fft = fft.sum(axis=2)

        score, dx, dy, overlap = find_peaks(fft, orig_section1, orig_section2, self.num_peaks)#, self.score_func)

        newconst = constraint.new_section(dx=dx, dy=dy, score=score, error=0)
        return newconst

    def align_full(self, constraint, precalc1=None, precalc2=None):
        image1, image2 = constraint.image1, constraint.image2

        if precalc1 is None:
            image1 = self.resize_if_needed(image1, constraint.box1, downscale_factor=self.downscale_factor)
        else:
            image1 = precalc1[0]

        if precalc2 is None:
            image2 = self.resize_if_needed(image2, constraint.box2, downscale_factor=self.downscale_factor)
        else:
            image2 = precalc2[0]

        orig_image1, orig_image2 = image1, image2
            
        if image1.shape != image2.shape:
            image1, image2 = image_diff_sizes(image1, image2)

        #import tifffile
        #tifffile.imwrite('tmp_prefft1.tif', image1)
        #tifffile.imwrite('tmp_prefft2.tif', image2)

        if image1.shape != orig_image1.shape or precalc1 is None or precalc1[1] is None: #precalc[1] would be none if precalculate_fft is false
            fft1 = np.fft.fft2(image1, axes=(0,1))
        else:
            fft1 = precalc1[1].copy() # fft1 is used for in place computation to save mem

        if image2.shape != orig_image2.shape or precalc2 is None or precalc2[1] is None:
            fft2 = np.fft.fft2(image2, axes=(0,1))
        else:
            fft2 = precalc2[1]

        #apply high pass filter
        fft1[:self.highpass_filter,:self.highpass_filter] = 0
        fft1[:self.highpass_filter,fft1.shape[1]-self.highpass_filter:] = 0
        fft1[fft1.shape[0]-self.highpass_filter:,:self.highpass_filter] = 0
        fft1[fft1.shape[0]-self.highpass_filter:,fft1.shape[1]-self.highpass_filter:] = 0

        fft2[:self.highpass_filter,:self.highpass_filter] = 0
        fft2[:self.highpass_filter,fft2.shape[1]-self.highpass_filter:] = 0
        fft2[fft2.shape[0]-self.highpass_filter:,:self.highpass_filter] = 0
        fft2[fft2.shape[0]-self.highpass_filter:,fft2.shape[1]-self.highpass_filter:] = 0

        #tifffile.imwrite('tmp_postfilter1.tif', np.fft.ifft2(fft1).real)
        #tifffile.imwrite('tmp_postfilter2.tif', np.fft.ifft2(fft2).real)

        fft = calc_pcm(fft1.reshape(-1), fft2.reshape(-1)).reshape(fft1.shape)
        if self.upscale_factor > 1:
            fft_prod = fft
        fft = np.fft.ifft2(fft, axes=(0,1)).real
        if len(fft.shape) == 3:
            fft = fft.sum(axis=2)
            #np.sum(fft, axis=2, 

        if not self.allow_edges:
            fft[0,:] = -np.inf
            fft[-1,:] = -np.inf
            fft[:,0] = -np.inf
            fft[:,-1] = -np.inf

        #import tifffile
        #tifffile.imwrite('tmp_fft.tif', fft.astype(np.float32))

        if constraint is not None and constraint.error is not None and constraint.error != math.inf and constraint.error != np.inf:
            score, dx, dy, overlap = find_peaks_estimate(fft, orig_image1, orig_image2, self.num_peaks,# self.score_func,
                    estimate=(constraint.dx, constraint.dy), search_range=constraint.error)
            if score == -math.inf:
                #print (dx, dy, previous_constraint.dx, previous_constraint.dy, previous_constraint.error, score)
                return constraint
        else:
            score, dx, dy, overlap = find_peaks(fft, orig_image1, orig_image2, self.num_peaks)#, self.score_func)

        if self.upscale_factor > 1:
            dx, dy = refine_peak(fft_prod, self.upscale_factor, np.array([dx, dy]))

        newconst = Constraint(constraint, dx=dx, dy=dy, score=score, error=0)

        if self.downscale_factor:
            newconst.dx *= self.downscale_factor
            newconst.dy *= self.downscale_factor
            newconst.error = self.downscale_factor

        return newconst


class PaddedFFTAligner(FFTAligner):

    def precalculate(self, image, box=None):
        if len(image.shape) == 3: image = image[:,:,0]
        image = self.resize_if_needed(image, box, downscale_factor=self.downscale_factor, padding=box.size[:2])
        fft = None if not self.precalculate_fft else np.fft.fft2(image, axes=(0,1))
        return image, fft

    def align(self, constraint, precalc1=None, precalc2=None):
        image1, image2 = constraint.image1, constraint.image2
        orig_orig_image1, orig_orig_image2 = image1, image2

        if precalc1 is None:
            image1 = self.resize_if_needed(image1, shape1, downscale_factor=self.downscale_factor, padding=shape1)
        else:
            image1 = precalc1[0]

        if precalc2 is None:
            image2 = self.resize_if_needed(image2, shape2, downscale_factor=self.downscale_factor, padding=shape2)
        else:
            image2 = precalc2[0]

        orig_image1, orig_image2 = image1, image2
            
        if image1.shape != image2.shape:
            image1, image2 = image_diff_sizes(image1, image2)

        if image1.shape != orig_image1.shape or precalc1 is None or precalc1[1] is None: #precalc[1] would be none if precalculate_fft is false
            fft1 = np.fft.fft2(image1, axes=(0,1))
        else:
            fft1 = precalc1[1].copy() # fft1 is used for in place computation to save mem

        if image2.shape != orig_image2.shape or precalc2 is None or precalc2[1] is None:
            fft2 = np.fft.fft2(image2, axes=(0,1))
        else:
            fft2 = precalc2[1]

        fft = calc_pcm(fft1.reshape(-1), fft2.reshape(-1)).reshape(fft1.shape)
        fft = np.fft.ifft2(fft, axes=(0,1)).real
        if len(fft.shape) == 3:
            fft = fft.sum(axis=2)
            #np.sum(fft, axis=2, 

        dx, dy = np.unravel_index(np.argmax(fft), fft.shape)
        score = fft[dx,dy]
        dx, dy = dx % constraint.box1.size[0], dy % constraint.box1.size[1]
        return Constraint(previous_constraint, dx=dx, dy=dy, score=score, error=0)


class PCCAligner(Aligner):
    """ An aligner that invokes the skimage.registration.phase_cross_correlation method
    An alternative to the FFTAligner which implements the same algorithm
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def align(self, constraint, precalc1=None, precalc2=None):
        offset, error, phasediff = skimage.registration.phase_cross_correlation(constraint.section1, constraint.section2, **self.kwargs)
        return constraint.new_section(dx=offset[0], dy=offset[1], score=error, error=0)


class FeatureAligner(Aligner):
    def __init__(self, num_features=2000):
        self.num_features = num_features

    def precalculate(self, image, box=None):
        import cv2 as cv

        image = (image / np.percentile(image, 99.9) * 255).astype(np.uint8)
        detector = cv.SIFT_create(nfeatures=self.num_features)
        
        keypoints, features = detector.detectAndCompute(image1, None)

        if shape is not None and image.shape != tuple(shape):
            keypoints[:,0] *= shape[0] / image.shape[0]
            keypoints[:,1] *= shape[1] / image.shape[1]

        return keypoints, features

    def align(self, constraint, precalc1=None, precalc2=None):
        import cv2 as cv

        (keypoints1, features1), (keypoints2, features2) = self.precalculate_if_needed(constraint, precalc1, precalc2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        matcher = cv.FlannBasedMatcher(index_params, search_params)

        matches = matcher.knnMatch(features1, features2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        displacements = []
        for match in good:
            pos1, pos2 = kp1[match.queryIdx].pt, kp2[match.trainIdx].pt
            if shape1 is not None:
                pos1 = pos1[0] * shape1[0] / image1.shape[0], pos1[1] * shape1[1] / image1.shape[1]
            if shape2 is not None:
                pos2 = pos2[0] * shape2[0] / image2.shape[0], pos2[1] * shape2[1] / image2.shape[1]
            displacements.append((pos2[0] - pos1[0], pos2[1] - pos1[1]))
        displacements = np.array(displacements)

        displacements = displacements.astype(int)
        values, counts = np.unique(displacements, axis=0, return_counts=True)
        #print (values, counts)
        best_value = np.argmax(counts)
        #print (values[best_value], counts[best_value])

        score = counts[best_value] / counts.sum()

        print ([end - begin for begin, end in zip(times, times[1:])])





###################################################################
## Code lifted from skimage.registration.phase_cross_correlation ##
###################################################################

def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal "
                "to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input "
                "data's number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * np.fft.fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data

def refine_peak(image_product, upsample_factor, shift):
    float_dtype = np.float32
    # Initial shift estimate in upsampled grid
    upsample_factor = np.array(upsample_factor, dtype=float_dtype)
    shift = np.round(shift * upsample_factor) / upsample_factor
    upsampled_region_size = np.ceil(upsample_factor * 1.5)
    # Center of output array at dftshift + 1
    dftshift = np.fix(upsampled_region_size / 2.0)
    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shift * upsample_factor
    cross_correlation = _upsampled_dft(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    ).conj()
    # Locate maximum and map back to original pixel grid
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    CCmax = cross_correlation[maxima]

    maxima = np.stack(maxima).astype(float_dtype, copy=False)
    maxima -= dftshift

    shift += maxima / upsample_factor

    #src_amp = np.sum(np.real(src_freq * src_freq.conj()))
    #target_amp = np.sum(np.real(target_freq * target_freq.conj()))

    return shift





def pcm(image1, image2):
    """Compute peak correlation matrix for two images.
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    F1 = np.fft.fft2(image1)
    F2 = np.fft.fft2(image2)
    F1 *= np.conjugate(F2)
    FC = F1
    del F1, F2
    FC /= np.abs(FC)
    FC = np.fft.ifft2(FC)
    FC = FC.real.astype(np.float32)
    return FC

def pcm_fft(F1, F2):
    """Compute peak correlation matrix for two images, with the fft already applied
    """
    FC = F1 * np.conjugate(F2)
    FC /= np.abs(FC)
    FC = np.fft.ifft2(FC)
    return FC.real.astype(np.float32)

def multi_peak_max(PCM):
    """Find the first to n th largest peaks in PCM.

    Returns:
    poses: np.ndarray, N by 2
        the indices of the peaks
    values : np.ndarray
        the values of the peaks
    """
    row, col = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
    vals = PCM[row[::-1], col[::-1]]
    return row[::-1], col[::-1], vals

def ncc_slow(image1, image2):
    """Compute the normalized cross correlation for two images.
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    image1 = image1.reshape(-1)
    image2 = image2.reshape(-1)
    n = np.dot(image1, image2)
    d = np.linalg.norm(image1) * np.linalg.norm(image2)
    return n / d

def zncc_slow(image1, image2):
    """Compute the zero-normalized cross correlation for two images.
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    image1 = image1.reshape(-1) - np.mean(image1)
    image2 = image2.reshape(-1) - np.mean(image2)
    n = np.dot(image1, image2)
    d = np.linalg.norm(image1) * np.linalg.norm(image2)
    return n / d


def extract_overlap_subregion(image, y, x):
    """Extract the overlapping subregion of the image.

    Parameters
    ---------
    image : np.ndarray
        the image (the dimension must be 2)
    y : Int
        the y (second last dim.) position
    x : Int
        the x (last dim.) position
    Returns
    -------
    subimage : np.ndarray
        the extracted subimage
    """
    sizeY = image.shape[0]
    sizeX = image.shape[1]
    assert (np.abs(y) < sizeY) and (np.abs(x) < sizeX)
    # clip x to (0, size_Y)
    xstart = max(0, min(int(y), sizeY))
    # clip x+sizeY to (0, size_Y)
    xend = max(0, min(y + sizeY, sizeY))
    ystart = max(0, min(x, sizeX))
    yend = max(0, min(x + sizeX, sizeX))
    return image[xstart:xend, ystart:yend]


def interpret_translation(image1, image2, yins, xins, ymin, ymax, xmin, xmax,
            num_peaks=2, max_peaks=None, ncc_threshold=None):
    """Interpret the translation to find the translation with heighest ncc.
        the first image (the dimension must be 2)
    yins : IntArray
        the y positions estimated by PCM
    xins : IntArray
        the x positions estimated by PCM
    ymin : Int
        the minimum value of y (second last dim.)
    ymax : Int
        the maximum value of y (second last dim.)
    xmin : Int
        the minimum value of x (last dim.)
    xmax : Int
        the maximum value of x (last dim.)
    num_peaks : Int
        the number of peaks to check
    max_peaks : Int
        a limit on the number of peaks to check, only used
        if ncc_threshold is declared.
    ncc_threshold : Float
        a threshold of the score of offsets (the ncc) that is needed to
        return an offset. If this is specified and after checking
        num_peaks peaks the highest score found is below this value,
        it will continue checking more peaks until the score is above
        the value or max_peaks is reached.

    Returns
    -------
    _ncc : Float
        the highest ncc
    x : Int
        the selected x position
    y : Int
        the selected y position
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    #assert np.array_equal(image1.shape, image2.shape)
    sizeY = max(image1.shape[0], image2.shape[0])
    sizeX = max(image1.shape[1], image2.shape[0])
    assert np.all(0 <= yins) and np.all(yins < sizeY)
    assert np.all(0 <= xins) and np.all(xins < sizeX)

    max_peaks = max_peaks or num_peaks
    ncc_threshold = ncc_threshold or 0

    xins = xins[:max_peaks]
    yins = yins[:max_peaks]

    _ncc = -np.infty
    y = 0
    x = 0

    ymagss = [yins, sizeY - yins]
    ymagss[1][ymagss[0] == 0] = 0
    xmagss = [xins, sizeX - xins]
    xmagss[1][xmagss[0] == 0] = 0
    del yins, xins

    # concatenate all the candidates
    _poss = []
    for ymags, xmags, ysign, xsign in itertools.product(
        ymagss, xmagss, [-1, +1], [-1, +1]
    ):
        yvals = ymags * ysign
        xvals = xmags * xsign
        _poss.append([yvals, xvals])
    poss = np.array(_poss)
    """
    valid_ind = (
        (ymin <= poss[:, 0, :])
        & (poss[:, 0, :] <= ymax)
        & (xmin <= poss[:, 1, :])
        & (poss[:, 1, :] <= xmax)
    )
    """
    #assert np.any(valid_ind)
    #print (poss.shape)
    #print (valid_ind.shape)
    #valid_ind = np.any(valid_ind, axis=0)
    #print (valid_ind.shape)
    #print (valid_ind[:100])
    for peakindex, pos in enumerate(np.moveaxis(poss, -1, 0)):
        for yval, xval in pos:
            if (ymin <= yval) and (yval <= ymax) and (xmin <= xval) and (xval <= xmax):
                subI1 = image1[max(0,yval):,max(0,xval):]
                subI2 = image2[max(0,-yval):,max(0,-xval):]
                dim1 = min(subI1.shape[0], subI2.shape[0])
                dim2 = min(subI1.shape[1], subI2.shape[1])
                subI1 = subI1[:dim1,:dim2]
                subI2 = subI2[:dim1,:dim2]
                #subI1 = extract_overlap_subregion(image1, yval, xval)
                #subI2 = extract_overlap_subregion(image2, -yval, -xval)
                if subI1.size > 0:
                    ncc_val = ncc(subI1, subI2)
                    if ncc_val > _ncc:
                        _ncc = float(ncc_val)
                        y = int(yval)
                        x = int(xval)
                del subI1, subI2

        if peakindex+1 >= num_peaks and (peakindex+1 >= max_peaks or _ncc >= ncc_threshold):
            break
    return _ncc, y, x



@numba.jit(nopython=True)
def calc_pcm(fft1, fft2):
    for i in range(fft1.shape[0]):
        val = fft1[i] * np.conjugate(fft2[i])
        fft1[i] = val / (np.abs(val) + 0.0000000000001)
    return fft1

@numba.jit(nopython=True)
def ncc_fast(image1, image2):
    """Compute the normalized cross correlation for two images.
    """
    total = np.float64(0)
    norm1, norm2 = np.float64(0), np.float64(0)
    for val1, val2 in zip(image1.flat, image2.flat):
        total += val1 * val2
        norm1 += val1 * val1
        norm2 += val2 * val2
    denom = np.sqrt(norm1) * np.sqrt(norm2)
    if denom == 0 and total == 0:
        return 0
    return total / denom

@numba.jit(nopython=True)
def zncc_fast(image1, image2):
    """Compute the zero normalized cross correlation for two images.
    """
    mean1, mean2 = image1.mean(), image2.mean()
    total = np.float64(0)
    norm1, norm2 = np.float64(0), np.float64(0)
    for val1, val2 in zip(image1.flat, image2.flat):
        val1 -= mean1
        val2 -= mean2
        total += val1 * val2
        norm1 += val1 * val1
        norm2 += val2 * val2
    denom = np.sqrt(norm1) * np.sqrt(norm2)
    if denom == 0 and total == 0:
        return 0
    return total / denom

@numba.jit(nopython=True)
def find_peaks(fft, image1, image2, num_peaks):#, score_func):
    best_peak = (-math.inf,0,0,0.0)
    shape = (max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1]))
    #print (shape, fft.shape, image1.shape, image2.shape)
    #print ('best fft', fft[shape[0]-241, 102])
    for i in range(num_peaks):
        peak_index = np.argmax(fft, axis=None)
        peak_val = fft.ravel()[peak_index]
        abs_xval, abs_yval = peak_index // fft.shape[1], peak_index % fft.shape[1]
        #print ('trying ', peak_val, abs_xval, abs_yval)
        #print (abs_xval, abs_yval, fft[abs_xval,abs_yval])
        #print (fft.reshape(-1)[peak_index], fft[abs_xval,abs_yval])
        #abs_xval, abs_yval = shape[0]-235, 120
        for abs_xval in (abs_xval, shape[0] - abs_xval):
            for abs_yval in (abs_yval, shape[1] - abs_yval):
                for xval in (abs_xval, -abs_xval):
                    for yval in (abs_yval, -abs_yval):
                        section1 = image1[max(0,xval):,max(0,yval):]
                        section2 = image2[max(0,-xval):,max(0,-yval):]
                        #print (' ', section1.shape, section2.shape)
                        dim1 = min(section1.shape[0], section2.shape[0])
                        dim2 = min(section1.shape[1], section2.shape[1])
                        #print (' ', dim1, dim2)
                        section1 = section1[:dim1,:dim2]
                        section2 = section2[:dim1,:dim2]
                        # only keeping sections with more than 100px overlap as anything lower
                        # has a much higher chance of matching well even if its random noise
                        if section1.size < 100: continue

                        overlap = max(section1.size / image1.size, section2.size / image2.size)
                        peak = (zncc_fast(section1, section2), xval, yval, overlap)

                        if peak[0] > best_peak[0]:
                            #print ('  best ', peak_val, peak, section1.shape)
                            best_peak = peak

                #print ('  ', abs_xval, abs_yval)
                if abs_xval < shape[0] and abs_yval < shape[1]:
                    fft[abs_xval,abs_yval] = -np.inf
    
    return best_peak

@numba.jit(nopython=True)
def find_peaks_estimate(fft, image1, image2,# score_func,
        num_peaks, estimate, search_range):
    best_peak = (-math.inf,0,0,0.0)
    shape = (max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1]))

    for x in range(fft.shape[0]):
        for y in range(fft.shape[1]):
            in_range = False
            for xval in (x, shape[0] - x):
                for yval in (y, shape[1] - y):
                    for xval in (xval, -xval):
                        for yval in (yval, -yval):
                            in_range = in_range or max(abs(estimate[0] - xval), abs(estimate[1] - yval)) <= search_range
            if not in_range:
                fft[x,y] = -np.inf

    for i in range(num_peaks):
        peak_index = np.argmax(fft, axis=None)
        abs_xval, abs_yval = peak_index // fft.shape[1], peak_index % fft.shape[1]
        for abs_xval in (abs_xval, shape[0] - abs_xval):
            for abs_yval in (abs_yval, shape[1] - abs_yval):
                for xval in (abs_xval, -abs_xval):
                    for yval in (abs_yval, -abs_yval):
                        if max(abs(estimate[0] - xval), abs(estimate[1] - yval)) > search_range:
                            continue

                        section1 = image1[max(0,xval):,max(0,yval):]
                        section2 = image2[max(0,-xval):,max(0,-yval):]
                        dim1 = min(section1.shape[0], section2.shape[0])
                        dim2 = min(section1.shape[1], section2.shape[1])
                        section1 = section1[:dim1,:dim2]
                        section2 = section2[:dim1,:dim2]
                        # only keeping sections with more than 100px overlap as anything lower
                        # has a much higher chance of matching well even if its random noise
                        if section1.size < 100: continue

                        overlap = max(section1.size / image1.size, section2.size / image2.size)
                        peak = (zncc_fast(section1, section2), xval, yval, overlap)

                        if peak[0] > best_peak[0]:
                            best_peak = peak

                if abs_xval < shape[0] and abs_yval < shape[1]:
                    fft[abs_xval,abs_yval] = -np.inf

    return best_peak


def image_diff_sizes(image1, image2):
    #new_shape = (max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1]))
    new_shape = tuple(np.maximum(image1.shape, image2.shape))

    if image1.shape != new_shape:
        newimg = np.zeros(new_shape, dtype=image1.dtype)
        #newimg[:] = image1.mean()
        image1 = np.pad(image1, (0, 50), mode='linear_ramp', end_values=0)
        newimg[:image1.shape[0],:image1.shape[1]] = image1
        image1 = newimg

    if image2.shape != new_shape:
        newimg = np.zeros(new_shape, dtype=image2.dtype)
        #newimg[:] = image2.mean()
        image2 = np.pad(image2, (0, 50), mode='linear_ramp', end_values=0)
        newimg[:image2.shape[0],:image2.shape[1]] = image2
        image2 = newimg

    return image1, image2


def calculate_offset_fast(image1, image2, shape1=None, shape2=None, fft1=None, fft2=None, num_peaks=2):
    if type(image1) == str:
        image1 = skimage.io.imread(image1)
    if type(image2) == str:
        image2 = skimage.io.imread(image2)

    while len(image1.shape) > 2:
        image1 = image1[:,:,0]
    while len(image2.shape) > 2:
        image2 = image2[:,:,0]

    if shape1 is not None and tuple(shape1) != image1.shape:
        image1 = skimage.transform.resize(image1, shape1)
    if shape2 is not None and tuple(shape2) != image2.shape:
        image2 = skimage.transform.resize(image2, shape2)

    orig_image1, orig_image2 = image1, image2
    if image1.shape != image2.shape:
        image1, image2 = image_diff_sizes(image1, image2)

    if fft1 is None:
        fft1 = np.fft.fft2(image1)
    else:
        fft1 = fft1.copy() #fft1 is used for inplace computation to save mem

    if fft2 is None:
        fft2 = np.fft.fft2(image2)

    fft = calc_pcm(fft1.reshape(-1), fft2.reshape(-1)).reshape(fft1.shape)
    fft = np.fft.ifft2(fft).real

    best_peak = find_peaks(fft, orig_image1, orig_image2, num_peaks, ncc_fast)
    return best_peak


def calculate_offset_slow(image1, image2, shape1=None, shape2=None, fft1=None, fft2=None, num_peaks=2, max_peaks=2, score_threshold=0.1):
    if type(image1) == str:
        image1 = skimage.io.imread(image1)
    if type(image2) == str:
        image2 = skimage.io.imread(image2)

    while len(image1.shape) > 2:
        image1 = image1[:,:,0]
    while len(image2.shape) > 2:
        image2 = image2[:,:,0]

    if shape1 is not None and tuple(shape1) != image1.shape:
        image1 = skimage.transform.resize(image1, shape1)
    if shape2 is not None and tuple(shape2) != image2.shape:
        image2 = skimage.transform.resize(image2, shape2)

    orig_image1, orig_image2 = image1, image2
    if image1.shape != image2.shape:
        image1, image2 = image_diff_sizes(image1, image2)

    sizeY, sizeX = max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1])

    times = []
    times.append(time.time())
    if fft1 is None or fft2 is None:
        PCM = pcm(image1, image2).real
    else:
        PCM = pcm_fft(fft1, fft2).real
    times.append(time.time())
    yins, xins, vals = multi_peak_max(PCM)
    del PCM
    #print (vals[:5])
    times.append(time.time())
    max_peak = interpret_translation(orig_image1, orig_image2, yins, xins, -sizeY, sizeY, -sizeX, sizeX,
                    num_peaks=num_peaks, max_peaks=max_peaks, ncc_threshold=score_threshold)
    times.append(time.time())
    #print ([end - start for start, end in zip(times, times[1:])])
    return max_peak

ncc = ncc_slow
zncc = zncc_slow
calculate_offset = calculate_offset_slow

def score_offset(image1, image2, dx, dy):
    if type(image1) == str:
        image1 = skimage.io.imread(image1)
    if type(image2) == str:
        image2 = skimage.io.imread(image2)

    subI1 = extract_overlap_subregion(image1, dx, dy)
    subI2 = extract_overlap_subregion(image2, -dx, -dy)
    del image1, image2
    ncc_val = ncc(subI1, subI2)
    return ncc_val
    

