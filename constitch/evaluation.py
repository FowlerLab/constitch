import numpy as np
import itertools

from .estimate_translation import pcm, multi_peak_max, extract_overlap_subregion, ncc


def sort_peaks(image1, image2, yins, xins, ymin, ymax, xmin, xmax, n = 25):
    """Interpret the translation to find the translation with heighest ncc.
        the first image (the dimension must be 2)
    yins : IntArray
        the y positions estimated by PCM
    xins : IntArray
        the x positions estimated by PCM
    n : Int
        the number of peaks to check, default is 2.
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    sizeY = image1.shape[0]
    sizeX = image1.shape[1]

    ymagss = [yins, sizeY - yins]
    ymagss[1][ymagss[0] == 0] = 0
    xmagss = [xins, sizeX - xins]
    xmagss[1][xmagss[0] == 0] = 0

    # concatenate all the candidates
    _poss = []
    for ymags, xmags, ysign, xsign in itertools.product(
        ymagss, xmagss, [-1, +1], [-1, +1]
    ):
        yvals = ymags * ysign
        xvals = xmags * xsign
        _poss.append([yvals, xvals])
    poss = np.array(_poss)
    valid_ind = (
        (ymin <= poss[:, 0, :])
        & (poss[:, 0, :] <= ymax)
        & (xmin <= poss[:, 1, :])
        & (poss[:, 1, :] <= xmax)
    )
    assert np.any(valid_ind)
    valid_ind = np.any(valid_ind, axis=0)

    poses = []
    scores = []

    for pos in np.moveaxis(poss[:, :, valid_ind], -1, 0)[: int(n)]:
        pos = np.unique(pos, axis=0)
        for xval, yval in pos:
            subI1 = extract_overlap_subregion(image1, xval, yval)
            subI2 = extract_overlap_subregion(image2, -xval, -yval)
            ncc_val = ncc(subI1, subI2)
            if not np.isnan(ncc_val):
                poses.append((xval, yval))
                scores.append(ncc_val)
    
    poses, scores = np.array(poses), np.array(scores)

    _, indices = np.unique(np.abs(poses), axis=0, return_index=True)
    poses = poses[indices]
    scores = scores[indices]

    order = np.argsort(scores)[::-1]
    return poses[order], scores[order]

import skimage.io

def evaluate_stitching1(image):
    print (ncc(image, image))
    sizeY, sizeX = image.shape[0], image.shape[1]
    PCM = pcm(image, image)
    yins, xins, _ = multi_peak_max(PCM)
    skimage.io.imsave('plots/pcm.tif', PCM.real)
    poses, scores = sort_peaks(image, image, yins, xins, -sizeY, sizeY, -sizeX, sizeX)
    print (poses[:10])
    print (scores[:10])
    return scores[0] / scores[1]


import sklearn.linear_model
import matplotlib.pyplot as plt

def evaluate_stitching2(image, radius=10):
    #shift1, shift2 = (4,6), (6,2)
    #image = ((image.astype(int) + np.roll(image, shift1, axis=(0,1)) // 2 + np.roll(image, shift2, axis=(0,1))) // 3).astype(image.dtype)
    mask = np.ones((radius*2 + 1, radius*2 + 1), dtype=bool)
    mask[radius-2:radius+3,radius-2:radius+3] = False
    mask[radius+1:,:] = False
    mask[radius,radius:] = False
    print (mask.astype(int))

    regions = []
    values = []
    for x in range(radius, image.shape[0]-radius):
        for y in range(radius, image.shape[1]-radius):
            region = image[x-radius:x+radius+1,y-radius:y+radius+1].astype(float)
            #region[radius,radius] = 0
            region = np.array([region, region[::-1,::-1]]).sum(axis=0)[mask]
            #section1 = region[:radius+1,:]
            #section2 = region[radius:,:][::-1,::-1]
            #region = np.array([section1, section2]).sum(axis=0)[mask]
            #region = np.array([section1, section2]).sum(axis=0).flatten()[:-(radius+1)]
            regions.append(region)
            values.append(image[x,y])

    regions = np.array(regions)
    values = np.array(values)
    print (regions)
    print (regions.shape)
    
    #model = sklearn.linear_model.RANSACRegressor(random_state=1245)
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(regions, values)

    error = np.mean((model.predict(regions) - values) ** 2)
    print (error)

    coefs = np.zeros((radius*2 + 1, radius*2 + 1))
    coefs[mask] = model.coef_
    coefs[mask[::-1,::-1]] = model.coef_[::-1]

    #coefs = np.concatenate([model.coef_, [0], model.coef_[::-1]]).reshape(radius*2 + 1, radius*2 + 1)

    fig, axis = plt.subplots()
    axis.imshow(coefs)
    fig.savefig('plots/eval_stitch_coefs.png')

evaluate_stitching = evaluate_stitching2
