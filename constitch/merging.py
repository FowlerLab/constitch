import sys
import numpy as np
import skimage.io

from .. import utils

class Merger:
    """ Abstract class that specifies the methods required for an image merger.
    At its heart an image merger is a method for handling multiple values at one
    pixel location, either taking one, combining them all, or some other method.
    Using this the merger is able to combine many images at different locations
    into one final image.
    The simplest merger is LastMerger, which doesn't do any extra processing, it
    simply places each image on top of each other, meaning overlapping areas will
    be the last image added. Other mergers attempt to combine images better but
    this requires using more memory and processing.
    """
    def create_image(self, image_shape, image_dtype):
        """ Init is called with the shape of the final image (may be
        more than 2d) and the dtype. Normally a merger would create
        an image using this information.
        """
        pass

    def add_image(self, image, location):
        """ Called to add an image. location is a tuple of slices used
        to select the area for the image, it is guaranteed to be the
        same size as image.
        """
        pass

    def final_image(self):
        """ Called once to get the final image. It will only be called
        once when stitching, so final processing can take place and modify
        the stored image. Returns the final image and a mask for pixels that
        had no images.
        """
        pass


class LastMerger(Merger):
    """ This is the simplest merger, overlap is decided just by which image was added last.
    This also means it has no extra memory requirements, if your images have no overlap or
    you don't need any merging this is the best choice.
    """
    def create_image(self, image_shape, image_dtype):
        self.image = np.zeros(image_shape, image_dtype)

    def add_image(self, image, location):
        self.image[location] = image

    def final_image(self):
        return self.image, np.ones(self.image.shape, dtype=bool)

class MeanMerger(Merger):
    """ A merger that calculates the mean of any overlapping areas. This requires
    storing the whole image using a larger dtype, eg when merging uint16 images
    the whole image will be stored as a uint32 image to account for possible overflow.
    This will double the memory requirements compared to other mergers such as LastMerger
    or EfficientMeanMerger.
    """
    def create_image(self, image_shape, image_dtype):
        self.image = np.zeros(image_shape, self.promote_dtype(image_dtype))
        self.counts = np.zeros(image_shape[:2], dtype=np.uint8)
        self.og_dtype = image_dtype

    def promote_dtype(self, dtype):
        if np.issubdtype(dtype, np.unsignedinteger):
            if dtype == np.uint8:
                return np.uint16
            elif dtype == np.uint16:
                return np.uint32
            return np.uint64
        elif np.issubdtype(dtype, np.signedinteger):
            if dtype == np.int8:
                return np.int16
            elif dtype == np.int16:
                return np.int32
            return np.int64
        return dtype

    def add_image(self, image, location):
        self.image[location] += image
        self.counts[location] += 1

    def final_image(self):
        mask = self.counts != 0
        if np.issubdtype(self.image.dtype, np.integer):
            self.image[mask] //= self.counts.reshape(self.image.shape[:2] + (1,) * (len(self.image.shape)-2))[mask]
        else:
            self.image[mask] /= self.counts.reshape(self.image.shape[:2] + (1,) * (len(self.image.shape)-2))[mask]
        return self.image.astype(self.og_dtype), self.counts != 0


class EfficientMeanMerger(Merger):
    """ A version of MeanMerger that does not require double the memory, however there is a small loss of 
    precision as calculation errors can add up as images are merged. It is recommended to start with
    MeanMerger and if memory becomes an issue switch to this one.
    """
    def create_image(self, image_shape, image_dtype):
        self.image = np.zeros(image_shape, image_dtype)
        self.counts = np.zeros(image_shape[:2] + (1,) * (len(image_shape)-2), dtype=np.uint8)

    def add_image(self, image, location):
        image_counts = self.counts[location]
        cur_image = self.image[location]
        if np.issubdtype(self.image.dtype, np.integer):
            cur_image[...] = (cur_image.astype(int) * image_counts + image) // (image_counts + 1)
        else:
            cur_image[...] = (cur_image * image_counts + image) / (image_counts + 1)
        image_counts += 1

    def final_image(self):
        return self.image, self.counts != 0


class NearestMerger(Merger):
    """ This merger keeps the pixel that is closer to the center of its image when there is overlap.
    This normally means that the edges of images are removed and central pixels are prioritized.
    This does require more memory, an extra array of uint16 the size of the final image. If memory
    is an issue EfficientNearestMerger uses a similar algorithm but only needs an array of uint8.
    """
    def create_image(self, image_shape, image_dtype):
        self.image = np.zeros(image_shape, image_dtype)
        self.dists = np.zeros(image_shape[:2] + (1,) * (len(image_shape)-2), dtype=np.uint16)

    def add_image(self, image, location):
        xdists, ydists = np.arange(image.shape[0]), np.arange(image.shape[1])
        xdists = np.minimum(xdists, xdists[::-1])
        ydists = np.minimum(ydists, ydists[::-1])
        dists = np.minimum(*np.meshgrid(xdists, ydists)).T + 1

        cur_image, cur_dists = self.image[location], self.dists[location]
        print (cur_dists.shape, dists.shape, image.shape, file=sys.stderr)
        mask = dists > cur_dists
        cur_image[mask] = image[mask]
        cur_dists[mask] = dists[mask]

    def final_image(self):
        return self.image, self.dists != 0


class EfficientNearestMerger(Merger):
    """ This merger does the same as the NearestMerger but it does it with only an extra array of
    uint8, halfing the extra memory requirements. The downside is that it will only trim up to 255
    pixels away from the edge of images, so if the overlap between your images is much larger than
    this, this merger will not give the same results.
    """
    def create_image(self, image_shape, image_dtype):
        self.image = np.zeros(image_shape, image_dtype)
        self.dists = np.zeros(image_shape[:2], dtype=np.uint8)
        print (self.image.shape, self.dists.shape, file=sys.stderr)
        import matplotlib.pyplot as plt
        self.fig, self.axis = plt.subplots()

    def add_image(self, image, location):
        x1, y1, x2, y2 = location[0].start, location[1].start, location[0].stop, location[1].stop
        self.axis.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])
        xdists, ydists = np.arange(image.shape[0]), np.arange(image.shape[1])
        xdists = np.clip(xdists, None, 254).astype(np.uint8)
        ydists = np.clip(ydists, None, 254).astype(np.uint8)
        xdists = np.minimum(xdists, xdists[::-1])
        ydists = np.minimum(ydists, ydists[::-1])
        dists = np.minimum(*np.meshgrid(xdists, ydists)).T + 1
        
        cur_image, cur_dists = self.image[location], self.dists[location]
        mask = dists >= cur_dists
        cur_image[mask] = image[mask]
        cur_dists[mask] = dists[mask]

    def final_image(self):
        self.fig.savefig('plots/merger_locations.png')
        return self.image, self.dists != 0




class MaskMerger(NearestMerger):
    """ This merger is specifically for merging an image mask, where each integer represents
    an independent class, for example cell segmentation masks. Other merging techniques won't
    work on these image masks. This merger first makes sure that every mask in each image
    has a unique integer value, then merges overlapping sections by going through the
    masks and combining masks that are mostly overlapping in both images. The amount
    of overlap needed to combine masks is the overlap_threshold.
    """
    def __init__(self, overlap_threshold=0.75):
        self.overlap_threshold = overlap_threshold

    def create_image(self, image_shape, image_dtype):
        super().create_image(image_shape, int)

    def add_image(self, image, location):
        print ("Adding image", location, file=sys.stderr)
        image = image.astype(int)

        #nextlabel = 1
        #while nextlabel in self.image:
            #nextlabel += 1

        image[image!=0] += self.image.max()

        overlapping_mask = self.dists[location] != 0
        overlapping_labels = np.unique(image[overlapping_mask])

        print ("Merging {} overlapping labels".format(len(overlapping_labels)), file=sys.stderr)
        for label in utils.simple_progress(overlapping_labels):
            if label == 0: continue

            mask = image == label
            mask_count = np.count_nonzero(mask)

            for otherlabel in np.unique(self.image[location][mask]):
                if otherlabel == 0: continue

                othermask = self.image[location] == otherlabel
                if min(mask_count, np.count_nonzero(othermask)) * self.overlap_threshold <= np.count_nonzero(mask & othermask):
                    image[mask] = otherlabel
                    break
            #else:
                #image[mask] = nextlabel
                #nextlabel += 1
                #while nextlabel in self.image:
                    #nextlabel += 1
        
        super().add_image(image, location)

