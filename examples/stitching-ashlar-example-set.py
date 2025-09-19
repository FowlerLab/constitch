""" This is an example of what a multi-cycle stitching script would look like.

Specifically this script was written to stitch the alignment test set
presented in the ASHLAR paper
"""

import sys
import re
import glob
import tifffile
import constitch
import numpy as np
import concurrent.futures

def stitch(cycle_images, cycle_poses, outpath):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    full_composite = constitch.CompositeImage(executor=executor, debug=True, progress=True)

    all_constraints = constitch.ConstraintSet()
    all_modeled = constitch.ConstraintSet()

    for cycle in range(len(cycle_images)):
        # Get composite for this cycle, images added to this composite will also
        # be added to full_composite with a layer of cycle
        composite = full_composite.layer(cycle)
        print (cycle_images[cycle].shape)
        composite.add_images(cycle_images[cycle], cycle_poses[cycle])

        #overlapping = composite.constraints(touching=True)
        # We define a custom function to only calculate alignment for adjacent tiles, not tiles
        # overlapping diagonally
        def const_filter(const):
            return const.touching and max(const.overlap_ratio_x, const.overlap_ratio_y) > 0.5

        overlapping = composite.constraints(const_filter)
        # Calculate the constraints, the pairwise alignment between each image pair
        constraints = overlapping.calculate(constitch.FFTAligner(upscale_factor=16, num_peaks=10))
        full_composite.plot_scores('{}_cycle{}_scores.png'.format(outpath.replace('/', '_'), cycle), constraints)
        constraints = constraints.filter(min_score=0.5)
        full_composite.plot_scores('{}_cycle{}_scores_filter.png'.format(outpath.replace('/', '_'), cycle), constraints)

        # Fit a linear model to the filtered constraints, then use it to estimate
        # missing constraints
        stage_model = constraints.fit_model(outliers=True)
        constraints = stage_model.inliers
        full_composite.plot_scores('{}_cycle{}_scores_outliers.png'.format(outpath.replace('/', '_'), cycle), constraints)
        modeled = overlapping.calculate(stage_model)

        # Combine constraints together
        all_constraints.add(constraints)
        all_modeled.add(modeled)

    # Find overlapping images between cycles and calculate their alignment
    overlapping = full_composite.constraints(lambda const: const.overlap_ratio > 0.5 and const.box1.position[2] != const.box2.position[2])
    constraints = overlapping.calculate(constitch.FFTAligner(upscale_factor=16, num_peaks=10))
    full_composite.plot_scores('{}_scores.png'.format(outpath.replace('/', '_')), constraints)
    constraints = constraints.filter(min_score=0.5)
    full_composite.plot_scores('{}_scores_filter.png'.format(outpath.replace('/', '_')), constraints)

    #stage_model = constraints.fit_model(outliers=True)
    #constraints = stage_model.inliers
    #full_composite.plot_scores('{}_scores_outliers.png'.format(outpath.replace('/', '_'), cycle), constraints)
    #modeled = overlapping.calculate(stage_model)

    all_constraints.add(constraints)
    #all_modeled.add(modeled)

    #for const in all_constraints:
        #const.score *= const.overlap_ratio
    #for const in all_modeled:
        #const.score *= const.overlap_ratio

    constitch.save(outpath.replace('.tif', '') + 'composite.json', full_composite, all_constraints, all_modeled)

    full_composite.plot_scores('{}_scores_presolved.png'.format(outpath.replace('/', '_')), all_constraints.merge(all_modeled))

    # Solve all the pairwise constraints, while minimizing the mean absolute error of them
    # meaning, find a set of global positions for each image tile that minimizes the error
    # on the calculated constraints
    solution = all_constraints.merge(all_modeled).solve(solver='mae')
    full_composite.setpositions(solution)

    # Plot the resulting tile locations
    full_composite.plot_scores('{}_scores_solved.png'.format(outpath.replace('/', '_')), all_constraints.merge(all_modeled))
    full_composite.plot_scores('{}_scores_solved2.png'.format(outpath.replace('/', '_')), all_constraints)
    full_composite.plot_scores('{}_scores_solved_accuracy.png'.format(outpath.replace('/', '_')), all_constraints.merge(all_modeled), score_func='accuracy')

    # Stitch each cycle together
    mins, maxes = full_composite.boxes.points1[:,:2].min(axis=0), full_composite.boxes.points2[:,:2].max(axis=0)
    for cycle in range(len(cycle_images)):
        composite = full_composite.layer(cycle)
        full_image = composite.stitch(merger=constitch.NearestMerger(), mins=mins, maxes=maxes)
        tifffile.imwrite(outpath + 'cycle{}-ch{}.tif'.format(cycle, 0), full_image)
        #for chan in range(cycle_images[cycle].shape[1]):
            #full_image = composite.stitch(cycle_images[cycle][chan], merger=constitch.NearestMerger())
            #tifffile.imwrite(outpath + 'cycle{}-ch{}.tif'.format(cycle, chan), full_image)


if __name__ == '__main__':
    paths = sys.argv[1:-1]
    outpath = sys.argv[-1]
    # As input we expect a list of tif files, one for each cycle.
    # for each tif file there is also a csv file with the same
    # name, which contains the positions of each tile.
    images = [tifffile.imread(path) for path in paths]
    poses = [np.loadtxt(path.replace('.tif', '') + '.csv', delimiter=',', skiprows=1).astype(int) for path in paths]
    for i in range(len(poses)):
        if poses[i].shape[1] == 4:
            poses[i] = np.array([poses[i][:,3], poses[i][:,2]]).T
            #poses[i] = poses[i][:,2:]
        poses[i] -= poses[i].min(axis=0).reshape(1,-1)

    stitch(images, poses, outpath)


