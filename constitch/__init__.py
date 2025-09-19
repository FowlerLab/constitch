""" ConStitch - A microscopy stitching library made for complex imaging experiments

ConStitch performs robust, high quality stitching of microscopy images. It is
specifically designed to stitch and align microscopy image sets with multiple
rounds of imaging, where each round needs to be aligned to each other as well
as stitched together.

Also see ASHLAR (<github.com/labsyspharm/ashlar>) for another package built to
stitch multi-cycle image sets, which also heavily inspired the design of this
package.

## Installation

ConStitch can be installed by cloning the repository, then installing with pip.

	git clone https://github.com/FowlerLab/constitch
	cd constitch
	pip3 install ./

## Usage

Example scripts are provided in examples/ that can help you get started. More
examples are coming, as well as an example dataset that can be downloaded and run easily.
Additionally an example of a snakemake pipeline using constitch for stitching
is available at <github.com/FowlerLab/starcall-workflow>.

A simple example of stitching a single cycle of imaging consists of these steps:

	import constitch
	import numpy as np
	import tifffile

	# load in images and tile positions
	images = tifffile.imread('images.tif')
	positions = np.loadtxt('tile_positions.csv', delimiter=',', dtype=int)

	# create composite image
	composite = constitch.CompositeImage()
	composite.add_images(images, positions, scale='tile')

	# find all overlapping regions between images
	overlapping = composite.constraints(touching=True)

	# align each image pair
	constraints = overlapping.calculate()

	# filter out erroneous constraints
	constraints = constraints.filter(min_score=0.5)
	# train a linear model on remaining constraints
	stage_model = constraints.fit_model(outliers=True)
	constraints = stage_model.inliers
	# and use it to estimate missing constraints
	modeled = overlapping.calculate(stage_model)

	# solve for global positions of each tile
	solution = modeled.merge(constraints).solve()
	composite.setpositions(solution)

	# (optional) plot locations of tiles to ensure correct stitching
	composite.plot_scores('plot.png', constraints)

	# stitch images together
	full_image = composite.stitch()

	tifffile.write('stitched.tif', full_image)

In this example it is assumed all the images being stitched are saved in the file images.tif of shape (num_images, width, height).
The positions of each image are stored in 'positions.csv', as the (row, col) position of each tile in the grid. You can
load in your data however you are able to, as long as your images are in the form of a numpy array of
shape (num_images, width, height), and you have positions for each image as a numpy array of shape (num_images, 2).
If your images don't follow a regular grid or you have more exact image positions, you can specify them
as pixel values, see the documentation for the function [CompositeImage.add_images](https://fowlerlab.github.io/starcall-docs/constitch.composite.html#CompositeImage-add_images)
for more information.

This example goes through the different steps needed to stitch a 2d grid of images. Each of the functions
used has reference documentation available at <fowlerlab.github.io/starcall-docs/constitch.html>
"""

from .composite import CompositeImage, BBox, BBoxList
from .constraints import Constraint, ConstraintSet, ConstraintFilter
from .alignment import calculate_offset, ncc, score_offset, Aligner, FFTAligner, FeatureAligner, PCCAligner
from .stage_model import SimpleOffsetModel, GlobalStageModel
from .stitching import stitch_cycles, make_test_image
from .evaluation import evaluate_stitching, evaluate_grid_stitching
from .merging import Merger, MeanMerger, EfficientMeanMerger, NearestMerger, MaskMerger, LastMerger, EfficientNearestMerger
from .solving import Solver, LinearSolver, OptimalSolver, OutlierSolver, MAESolver, SpanningTreeSolver, LPSolver
from .utils import save, load


__all__ = [
    "composite",
    "constraints",
    "alignment",
    "stitching",
    "evaluation",
    "merging",

    "CompositeImage",
    "BBox",
    "BBoxList",
    "Constraint",
    "ConstraintSet",
    "ConstraintFilter",

    "Aligner",
    "FFTAligner",
    "PCCAligner",
    "FeatureAligner",

    "Merger",
    "MeanMerger",
    "EfficientMeanMerger",
    "NearestMerger",
    "MaskMerger",
    "LastMerger",
    "EfficientNearestMerger",

    "Solver",
    "LinearSolver",
    "OptimalSolver",
    "OutlierSolver",
    "MAESolver",
    "SpanningTreeSolver",
    "LPSolver",
    "MAESolver",
	"HuberSolver",

    "SimpleOffsetModel",
    "GlobalStageModel",

    "save",
    "load",
]
