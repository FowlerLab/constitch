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

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, threads))
    composite = constitch.CompositeImage(images=images[:,stitching_cycle], positions=positions, scale='tile',
                                         debug=True, progress=True, executor=executor)
    composite.plot_scores('plots/initial_poses.png')

    calculate_params = dict(upscale_factor=16)

    overlapping = composite.constraints(touching=True)
    constraints = overlapping.calculate(**calculate_params)
    composite.plot_scores('plots/prefilter_constraints.png', constraints)

    nonoverlapping = composite.constraints(lambda const: const.overlap_x < -3000 and const.overlap_y < -3000, limit=100, random=True)
    erroneous_constraints = nonoverlapping.calculate(**calculate_params)

    score_threshold = np.percentile([const.score for const in erroneous_constraints], 95) if len(erroneous_constraints) else 0.5
    constraints = constraints.filter(min_score=score_threshold)

    stage_model = constitch.SimpleOffsetModel()
    stage_model = constraints.fit_model(stage_model, outliers=True)
    constraints = stage_model.inliers
    modeled = overlapping.calculate(stage_model)

    composite.plot_scores('plots/postfilter_constraints.png', constraints)

    solving_constraints = constraints.merge(modeled)

    composite.plot_scores('plots/solving_constraints.png', solving_constraints)

    solution = solving_constraints.solve(solver='pulp', threads=threads*2)

    composite.setpositions(solution)
    composite.plot_scores('plots/solved.png', solving_constraints)
    composite.plot_scores('plots/solved_accuracy.png', solving_constraints, score_func='accuracy')

    full_image = composite.stitch(real_images=images.transpose(0,2,3,1), merger=merging_method, prevent_resize=True)

    return full_image


if __name__ == '__main__':
    images = tifffile.imread(sys.argv[1])
    positions = np.loadtxt(sys.argv[2], delimiter=',', skiprows=1, dtype=int)

    full_image = stitch(images, positions)
    tifffile.imwrite(sys.argv[3], full_image)

