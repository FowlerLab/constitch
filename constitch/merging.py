import sys
import numpy as np
import skimage.io

from .. import utils

class Merger:
    """Abstract class that specifies the methods required for an image merger.
    """
    def __init__(self, image_shape, image_dtype):
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


class MeanMerger:
    def __init__(self, image_shape, image_dtype):
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


class EfficientMeanMerger:
    def __init__(self, image_shape, image_dtype):
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


class NearestMerger:
    def __init__(self, image_shape, image_dtype):
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

class LastMerger:
    def __init__(self, image_shape, image_dtype):
        self.image = np.zeros(image_shape, image_dtype)

    def add_image(self, image, location):
        self.image[location] = image

    def final_image(self):
        return self.image, np.zeros(self.image, dtype=bool)


class MaskMerger(NearestMerger):
    def __init__(self, image_shape, image_dtype, overlap_threshold=0.75):
        super().__init__(image_shape, int)
        self.overlap_threshold = overlap_threshold

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

