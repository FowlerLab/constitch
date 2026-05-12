import sys
import concurrent.futures
import constitch
import tifffile
import numpy as np


def stitch(images, positions, merging_method='efficient_nearest', stitching_cycle=0, threads=8):
    """ Stitches a single imaging cycle of tiles into a full well image.
    images should be a numpy array of the tiles, of shape (num_tiles, num_channels, width, height).
    positions should be a numpy array of shape (num_tiles, 2) with the tile positions, eg (0,0), (0,1), ...
    Returns the full well image, shape (num_channels, full_width, full_height). Areas with no tiles
    are filled black
    """
    images, positions = np.asarray(images), np.asarray(positions)

    single_channel = images.ndim == 3
    if single_channel:
        images = images[:,None,:,:]

    # Create executor for multithreading. Has to be ThreadPool not ProcessPool so the numpy arrays
    # can be shared and not copied
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, threads))
    # Creates the composite with the images and positions. If tiles are not in an exact grid layout
    # the positions can be specified in pixels, for that set scale='pixel'
    composite = constitch.CompositeImage(images=images[:,stitching_cycle], positions=positions, scale='tile',
                                         debug=True, progress=True, executor=executor)
    composite.plot_scores('plots/initial_poses.png')

    # Alignment is calculated to 1/16 of a pixel with interpolation. More than this is usually unnecessary
    calculate_params = dict(upscale_factor=16)

    # Touching means any pairs of images that are overlapping or sharing an edge
    overlapping = composite.constraints(touching=True)
    constraints = overlapping.calculate(**calculate_params)
    composite.plot_scores('plots/prefilter_constraints.png', constraints)

    # To filter constraints we calculate a random set of non overlapping images,
    # as examples of what erroneous constraint scores are like
    nonoverlapping = composite.constraints(lambda const: const.overlap_x < -3000 and const.overlap_y < -3000, limit=100, random=True)
    erroneous_constraints = nonoverlapping.calculate(**calculate_params)

    score_threshold = np.percentile([const.score for const in erroneous_constraints], 95) if len(erroneous_constraints) else 0.5
    constraints = constraints.filter(min_score=score_threshold)

    # To further filter we fit an outlier resistent model to the constraints,
    # which removes any that don't match the estimated microscope params
    stage_model = constitch.SimpleOffsetModel()
    stage_model = constraints.fit_model(stage_model, outliers=True)
    constraints = stage_model.inliers
    # The model is used to fill in any constraints that were removed
    modeled = overlapping.calculate(stage_model)

    composite.plot_scores('plots/postfilter_constraints.png', constraints)

    solving_constraints = constraints.merge(modeled)

    composite.plot_scores('plots/solving_constraints.png', solving_constraints)

    # Finding global positions for each tile that minimize the error on all constraints.
    # 'pulp' uses integer linear programming and is the most precise, but can be very expensive
    # as the number of constraints grows. The best alternative is 'mae'
    solution = solving_constraints.solve(solver='pulp', threads=threads*2)

    composite.setpositions(solution)
    composite.plot_scores('plots/solved.png', solving_constraints)
    composite.plot_scores('plots/solved_accuracy.png', solving_constraints, score_func='accuracy')

    # Combines the tiles with the global positions to get the final image.
    # merging_method decides how to merge overlapping regions. Common methods are 'efficient_nearest'
    # keeps the image that is closest to the pixel; basically crops tiles until they are not overlapping
    # 'efficient_mean' calculates the mean of overlapping regions
    full_image = composite.stitch(real_images=images.transpose(0,2,3,1), merger=merging_method, prevent_resize=True)

    if single_channel:
        full_image = full_image[:,0]

    return full_image


if __name__ == '__main__':
    images = tifffile.imread(sys.argv[1])
    positions = np.loadtxt(sys.argv[2], delimiter=',', skiprows=1, dtype=int)

    full_image = stitch(images, positions)
    tifffile.imwrite(sys.argv[3], full_image)

