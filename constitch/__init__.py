"""
The stitching submodule for the fisseq pipeline.

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

The general steps for the stitching of a group images are as follows:


Creating the composite

The stitching process is encapsulated in the CompositeImage class,
and when creating it we can set various parameters that control
its behaviour. The full method signature can be found at
CompositeImage.__init__() but some important parameters are described below:

The executor is what the composite uses to perform intensive computation
tasks, namely calculating the alignment of all the images. If provided
it should be a concurrent.futures.Executor object, for example
concurrent.futures.ThreadPoolExecutor. Importantly, concurrent.futures.ProcessPoolExecutor
does not work very well as the images need to be passed to the executor
and in the case of ProcessPoolExecutor this means they need to be pickled
and unpickled to get to the other process. ThreadPoolExecutor doesn't need
this as the threads can share memory, but it doesn't take full advantage of
multithreading as the python GIL prevents python code from running in parallel.
Luckily most of the intensive computation happens in numpy functions which don't
hold the GIL, so ThreadPoolExecutor is usually the best choice.

The arguments debug and process define how logging should happen with the composite.
If debug is True logging messages summarizing the results of different operations
will be printed out to stderr. Setting it to False will disable these messages.
If process is True a progress bar will be printed out during long running steps.
The default progress bar is a simple ascii bar that works whether the output is
a tty or a file, but if you want you can pass a replacement in instead of setting
it to True, such as tqdm.

An example of setting up the composite would be something similar to this:

import fisseq
import concurrent.futures
import tqdm

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    composite = fisseq.stitching.CompositeImage(executor=executor, debug=True, progress=tqdm.tqdm)


Setting the aligner

The aligner of the composite is a class that encapsulates the algorithm used to align two images onto
each other. The base class Aligner has more information on the specifics but basically
an aligner has a function that takes two images and returns the offset from one image to another
that has the best overlap. This is normally measured using the normalized cross correlation
of the overlapping regions, using the function alignment.ncc, but different aligner classes
can use different ways to find and score overlapping regions.

The main aligner provided is the FFTAligner, which uses the phase cross correlation algorithm
to find the best overlapping region. By default this one is used, however if you want to modify
the parameters you can set the aligner to a new instance of it. For example:
    composite.set_aligner(fisseq.stitching.FFTAligner(num_peaks=5, precalculate_fft=False))

The other aligner provided is the FeatureAligner, which uses feature based methods to align the images.
This is quite a bit faster than the FFTAligner, but it is not as accurate or reliable, so be aware that
results might not be as good.

As with many of the classes used for stitching, the aligner is meant to be customized and you are
encouraged to subclass the Aligner class and make your own method. The two methods needed for an
Aligner class are described in its docs.


Adding the images

Once the composite is set up we can add the images, and this is done through the CompositeImage.add_images()
method. There are a couple ways of adding images, depending on how much information you have on the images:

First of all, you can just add the images with no positions, meaning they will all default to being at 0,0.
This will work out, as when you calculate constraints between images it will calculate constraints for all
possible images and filter out constraints that have no overlap. However the number of constraints that have
to be calculated grows exponentially with the number of images, and if you have positional information on your
images it is best to pass that in to help with the alignment. If you would like to use this method but are running
into computational limits, the section on pruning constraints below can be helpful.
    composite.add_images(images)

If your images are taken on a grid you can pass in their positions as grid positions, by setting the scale
parameter to 'tile'. For example:
    positions=[(0,0), (0,1), (1,0), (1,1)]
    composite.add_images(images, positions=positions, scale='tile')
Now when constraints are calculated only nearby images will be checked, speeding up computation a lot.

If your images are not on a grid or you have the exact position they were taken in, you can also specify
positions in pixels instead of grid positions, to do this simply set the scale parameter to 'pixel'
and the positions passed in will be interpreted as pixel coordinates.

When specifying positions, you can also specify more than two dimensions. The first two are the x and y
dimensions of the images, but a z dimension can be added if you are doing 3 dimensional stitching or in our case
if you are doing fisseq and want to make sure all the cycles line up perfectly. In the case of fisseq,
you can add the cycle index as the z coordinate for the image.


Calculating constraints

Once images have been added we need to calculate the constraints between overlapping images. This is done
with the CompositeImage.calc_constraints() function. For usual usage you can run it with no parameters,
but if you want to be specific about which images are overlapping it takes in an argument called pairs,
which should be a sequence of tuples, each being a pair of image indices that should be checked for overlap.
If pairs is not specified it defaults to CompositeImage.find_unconstrained_pairs(), which finds all image
pairs that have overlap but don't have a constraint between them. The two lines below both work to calculate
constraints:
    composite.calc_constraints()
    composite.calc_constraints(composite.find_unconstrained_pairs())

One important parameter is precalculate, if True the results of the precalculation step will be cached
for each image, saving on computation time but increasing memory usage.

The find_unconstrained_pairs function also has some parameters that affect how it looks for overlap,
the most useful one being overlap_threshold. This is a value in pixels that specifies how far images
have to overlap to be considered overlapping. This can also be negative, meaning images that are within
a certain pixel distance to touching will be considered overlapping. This is useful if you want to make sure
that you find all overlapping images, even if your original positions are not very accurate. For example,
the line below will expand the search by 2000 pixels and hopefully find more overlap
    composite.calc_constraints(composite.find_unconstrained_pairs(overlap_threshold=(-2000, -2000)))

Another use would be for fisseq, where you want to calculate constraints across all cycles and not just
adjacent cycles. As described before fisseq cycles can be represented using the z axis, so we can expand
the search in that z axis and calculate constraints between all cycles:
    composite.calc_constraints(composite.find_unconstrained_pairs(overlap_threshold=(0, 0, -12))


Filtering constraints

After calculating constraints for all overlapping images, we have to filter out constraints that aren't accurate.
Unfortunately the alignment algorithms are not perfect and sometimes they miss overlap, so we need to filter out
constraints that have low scores or don't fit in with the other constraints. There are a couple steps of this,
the first being filtering based on scores:
    composite.filter_constraints()
This method, CompositeImage.filter_constraints(), looks at the scores of all constraints and calculates a random
set of bad constraints to find a good score threshold to eliminate any constraints that are not accurate. This
is necessary because the scores returned by the alignment algorithms depend on the features present in the images,
and using a fixed threshold would not work for all image types.

"""

from .composite import CompositeImage, BBox, BBoxList
from .constraints import Constraint
from .alignment import calculate_offset, ncc, score_offset, Aligner, FFTAligner, FeatureAligner
#from .stage_model import StageModel
from .stitching import stitch_cycles, make_test_image
from .evaluation import evaluate_stitching, evaluate_grid_stitching
from .merging import Merger, MeanMerger, EfficientMeanMerger, NearestMerger, MaskMerger, LastMerger, EfficientNearestMerger


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
    "Aligner",
    "FFTAligner",
    "FeatureAligner",

    "Merger",
    "MeanMerger",
    "EfficientMeanMerger",
    "NearestMerger",
    "MaskMerger",
    "LastMerger",
    "EfficientNearestMerger",
]
