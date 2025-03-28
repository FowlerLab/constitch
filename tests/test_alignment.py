import unittest
import constitch
import numpy as np
import skimage.transform
import tifffile

class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.composite = constitch.CompositeImage()

        self.rng = np.random.default_rng(12345)
        self.image = (
            skimage.transform.rescale(self.rng.random(size=(10,10)), 50)
            + skimage.transform.rescale(self.rng.random(size=(50,50)), 10) * 0.2
            + self.rng.random(size=(500,500)) * 0.05)
        tifffile.imwrite('tmp_image.tif', self.image.astype(np.float32))

        #self.offsets = [(0,0), (5,0), (0,5), (10,6), (25, 0), (0, 25), (75, 0), (0, 75)]
        self.offsets = [(0,0), (48, 48), (48, 55), (55, 48), (68, 48), (48, 68), (100, 48)]
        self.composite.add_images([self.image[pos[0]:pos[0]+200,pos[1]:pos[1]+200] for pos in self.offsets])

        self.downscaled_composite = constitch.CompositeImage()
        self.image_upscaled = skimage.transform.rescale(self.image, 16)
        self.subpixel_offsets = [(0,0), (4, 0), (8, 0), (16, 0), (8, 8), (4, 7), (13, 7)]

        images = [skimage.transform.rescale(self.image_upscaled[:200*16,:200*16], 1/16)]
        offset = 48 * 16
        images.extend(
            skimage.transform.rescale(
                self.image_upscaled[offset+pos[0]:offset+pos[0]+200*16,offset+pos[1]:offset+pos[1]+200*16],
                1/16) for pos in self.subpixel_offsets[1:])

        print ([image.shape for image in images])

        self.downscaled_composite.add_images(images)
        tifffile.imwrite('tmp_image_downscaled.tif', self.downscaled_composite.images[:5])

    def test_aligner(self):
        for i in range(len(self.offsets)):
            const = self.composite.constraints[0,i]
            const = const.calculate()
            self.assertEqual((const.dx, const.dy), self.offsets[i])

    def test_aligner_upscale(self):
        for i in range(len(self.subpixel_offsets)):
            const = self.downscaled_composite.constraints[0,i]
            const = const.calculate(constitch.FFTAligner(upscale_factor=16))
            offset = self.subpixel_offsets[i]
            offset = 48 + offset[0] / 16, 48 + offset[1] / 16
            print (const.dx, const.dy, offset[0], offset[1])
            #self.assertEqual((const.dx, const.dy), self.offsets[i])

if __name__ == '__main__':
    unittest.main()

