"""
ConStitch: A stitching library that solves the global alignment of images using
a graph of pairwise constraints between images.

Stitching is a vital step in the fisseq pipeline, as it is in any microscopy
data pipeline, however fisseq has some requirements that make stitching even
more important than in other experimental procedures. This is mainly because
the features we need to detect in cells are quite small, and they need to line
up with each other between cycles in order to detect and read them. To
accomplish this, the stitching package is basically a full stitching library,
capable of stitching any group of images into one contiguous image or
stack of images.

This stitching library is based on building up a collection of pairwise
offsets between images, represented by the Constraint class. Using
different algorithms these constraints can be calculated between all
overlapping images, then filtered and processed to improve the accuracy.
Finally, we can consider all constraints and globally solve the positions
of all the images.

The library is meant to be simple to use for simple use cases, but allow for customization
and fine tuning when you need more control. The whole stitching process is contained in the
CompositeImage class, and the simplest working example is shown below,
it stitches together the provided images and creates a full composite image of them combined
together:

    composite = constitch.CompositeImage(images, positions)
    constrants = composite.constraints(touching=True).calculate().filter(min_score=0.5)
	composite.setpositions(constraints.solve())
    full_image = composite.stitch()

This will work with most smaller stitching problems, where images is a list of the images
in the form of numpy arrays, and positions is a numpy array of initial positions for each
image. A more in depth run through of the stitching process can be found in the
documentation of the CompositeImage.

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
