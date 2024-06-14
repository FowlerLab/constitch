import time
import numpy as np
import skimage.io
import itertools

from .constraints import Constraint

import cv2 as cv

import numba


class Aligner:
    """ Abstract class that defines an algorithm for aligning two images onto each other.
    """
    def precalculate(self, image, shape1=None):
        """ Performs an arbitrary precalculation step specific to the alignment algorithm.
        This would be a step in the algorithm that only has to be run once per image,
        and can be cached for each calculation done with the same image. The result of
        this function will be passed to the align function as the precalc1 or precalc2 argument
        """
        pass

    def align(self, image1, image2, shape1=None, shape2=None, precalc1=None, precalc2=None):
        """ Performs the alignment of two images, finding the pixel offset that best aligns the
        two images. The offset should be from image1 to image2. The return value should be a Constraint
        object, with at least the dx, dy fields filled in to represent the offset of image2 needed
        to align the two images. If the function finds images don't overlap, None should be returned.
        """
        pass

    def resize_if_needed(self, image, shape=None, downscale_factor=None):
        if shape is not None and image.shape != tuple(shape):
            image = skimage.transform.resize(image, shape)
        if downscale_factor is not None and downscale_factor != 1:
            image = skimage.transform.downscale_local_mean(image, downscale_factor)
        return image
    
    def precalculate_if_needed(self, image1, image2, shape1=None, shape2=None, precalc1=None, precalc2=None):
        if precalc1 is None:
            precalc1 = self.precalculate(image1, shape1)
        if precalc2 is None:
            precalc2 = self.precalculate(image2, shape2)
        return precalc1, precalc2


class FFTAligner:
    def __init__(self, num_peaks=2, downscale_factor=None):
        self.num_peaks = num_peaks
        self.downscale_factor = downscale_factor

    def precalculate(self, image, shape=None):
        image = self.resize_if_needed(image, shape, self.downscale_factor)
        fft = np.fft.fft2(image)
        return image, fft

    def align(self, image1, image2, shape1=None, shape2=None, precalc1=None, precalc2=None, previous_constraint=None):
        if precalc1 is None:
            precalc1 = self.precalculate(image1, shape1)
        else:
            precalc1[1] = precalc1[1].copy() #fft1 is used for inplace computation to save mem
        if precalc2 is None:
            precalc2 = self.precalculate(image2, shape2)

        image1, fft1 = precalc1
        image2, fft2 = precalc2

        orig_image1, orig_image2 = image1, image2
        if image1.shape != image2.shape:
            image1, image2 = image_diff_sizes(image1, image2)

        fft = calc_pcm(fft1.reshape(-1), fft2.reshape(-1)).reshape(fft1.shape)
        fft = np.fft.ifft2(fft).real

        score, dx, dy = find_peaks(fft, orig_image1, orig_image2, num_peaks)
        constraint = Constraint(score, dx, dy)

        if self.downscale_factor:
            constraint.dx *= self.downscale_factor
            constraint.dy *= self.downscale_factor
            constraint.error = self.downscale_factor

        return constraint


class FeatureAligner:
    def __init__(self, num_features=2000):
        self.num_features = num_features

    def precalculate(self, image):
        image = (image / np.percentile(image, 99.9) * 255).astype(np.uint8)
        detector = cv.SIFT_create(nfeatures=self.num_features)
        
        keypoints, features = detector.detectAndCompute(image1, None)

        return keypoints, features

    def align(self, image1, image2, shape1=None, shape2=None, precalc1=None, precalc2=None):
        (keypoints1, features1), (keypoints2, features2) = self.precalculate_if_needed(image1, image2, shape1, shape2, precalc1, precalc2)

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
    n = np.dot(image1 - np.mean(image1), image2 - np.mean(image2))
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
        fft1[i] = val / np.abs(val)
    return fft1

@numba.jit(nopython=True)
def ncc_fast(image1, image2):
    """Compute the normalized cross correlation for two images.
    """
    mean1, mean2 = image1.mean(), image2.mean()
    total = np.float64(0)
    norm1, norm2 = np.float64(0), np.float64(0)
    for val1, val2 in zip(image1.flat, image2.flat):
        total += (val1 - mean1) * (val2 - mean2)
        norm1 += val1 * val1
        norm2 += val2 * val2
    denom = np.sqrt(norm1) * np.sqrt(norm2)
    if denom == 0 and total == 0:
        return 0
    return total / denom

@numba.jit(nopython=True)
def find_peaks(fft, image1, image2, num_peaks):
    best_peak = (0,0,0)
    shape = (max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1]))
    for i in range(num_peaks):
        peak_index = np.argmax(fft, axis=None)
        xval, yval = peak_index // fft.shape[1], peak_index % fft.shape[1]
        for xval in (xval, shape[0] - xval):
            for yval in (yval, shape[1] - yval):
                for xval in (xval, -xval):
                    for yval in (yval, -yval):
                        section1 = image1[max(0,xval):,max(0,yval):]
                        section2 = image2[max(0,-xval):,max(0,-yval):]
                        dim1 = min(section1.shape[0], section2.shape[0])
                        dim2 = min(section1.shape[1], section2.shape[1])
                        section1 = section1[:dim1,:dim2]
                        section2 = section2[:dim1,:dim2]
                        if section1.size == 0: continue

                        peak = (ncc_fast(section1, section2), xval, yval)

                        if peak[0] > best_peak[0]:
                            best_peak = peak

                if xval < shape[0] and yval < shape[1]:
                    fft[xval,yval] = 0

    return best_peak


def image_diff_sizes(image1, image2):
    new_shape = max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1])

    if image1.shape != new_shape:
        newimg = np.zeros(new_shape, dtype=image1.dtype)
        newimg[:image1.shape[0],:image1.shape[1]] = image1
        image1 = newimg

    if image2.shape != new_shape:
        newimg = np.zeros(new_shape, dtype=image2.dtype)
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

    best_peak = find_peaks(fft, orig_image1, orig_image2, num_peaks)
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
    

