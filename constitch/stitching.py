import collections
import math
import dataclasses
import itertools
import numpy as np
import glob
import os
import sys
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import skimage.filters

def stitch_cycles(images, positions, debug=True, progress=False, **kwargs):
    """ Stitches a time sequence of composite images together, aligning both between tiles as well as
        between cycles.

    """
    progress_arg = progress
    if debug: debug = print
    else: debug = lambda *args: None
    if progress:
        import tqdm
        progress = tqdm.tqdm
    else:
        progress = lambda x, **kwargs: x

    import m2stitch
    constraints = []

    for cycle in range(positions[:,0].max() + 1):
        debug ('Stitching cycle', cycle)
        mask = positions[:,0] == cycle
        indices = np.arange(len(positions))[mask]

        result, stats = m2stitch.stitch_images(images[mask], position_indices=positions[mask][:,1:], full_output=True, **kwargs)#, silent=not progress_arg, **kwargs)
        for index, row in result.iterrows():
            for j, direction in enumerate(['left', 'top']):
                if not pd.isna(row[direction]):
                    constraints.append(Constraint(indices[index], indices[row[direction]], row[direction+'_ncc'], (-row[direction+'_y'], -row[direction+'_x'])))
        debug('  done')


    debug('Aligning images across cycles')
    position_indices = {}
    for i in range(len(images)):
        pos = (positions[i,1], positions[i,2])
        position_indices[pos] = position_indices.get(pos, []) + [i]

    for pos, indices in progress(position_indices.items()):
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                score, dx, dy = calculate_offset(images[indices[i]], images[indices[j]])
                constraints.append(Constraint(indices[i], indices[j], score, (dx, dy)))
    debug('  done')

    solution_mat = np.zeros((len(constraints)*2+2, len(images)*2))
    solution_vals = np.zeros(len(constraints)*2+2)
    
    for index in range(len(constraints)):
        id1, id2, score, (dx, dy) = constraints[index]

        solution_mat[index*2, id1*2] = -score
        solution_mat[index*2, id2*2] = score
        solution_vals[index*2] = score * dx

        solution_mat[index*2+1, id1*2+1] = -score
        solution_mat[index*2+1, id2*2+1] = score
        solution_vals[index*2+1] = score * dy

    # anchor tile 0 to 0,0, otherwise there are inf solutions
    solution_mat[-2, 0] = 1
    solution_mat[-1, 1] = 1

    debug('Solving for final tile positions')
    solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
    debug('  done')

    poses = np.round(solution.reshape(-1,2)).astype(int)
    poses -= poses.min(axis=0).reshape(1,2)
    return poses


def make_test_image(image, grid_size, window_size=50):
    rows = []
    x_size, y_size = window_size, window_size
    for x in np.linspace(0, image.shape[0]-1, grid_size+1).astype(int):
        if x == 0 or x == image.shape[0]-1: x_size *= 2
        tiles = []
        for y in np.linspace(0, image.shape[1]-1, grid_size+1).astype(int):
            if y == 0 or y == image.shape[1]-1: y_size *= 2
            tiles.append(image[x-x_size:x+x_size+1,y-y_size:y+y_size+1])
        rows.append(np.concatenate(tiles, axis=1))
    test_image = np.concatenate(rows, axis=0)
    return test_image



