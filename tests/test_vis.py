import unittest
import constitch
import constitch.visualization
import numpy as np
import io
import os
import tifffile
import PIL.Image
import matplotlib.pyplot as plt

class TestVis(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.images = [
            np.arange(100).reshape(10,10).astype(np.uint16),
            self.rng.integers(65535, size=(10,10), dtype=np.uint16),
            np.arange(100).reshape(10,10).astype(float),
            self.rng.normal(size=(10,10)),
            np.arange(400).reshape(4,10,10).astype(np.uint16),
            self.rng.integers(65535, size=(4,10,10), dtype=np.uint16),
            np.arange(400).reshape(4,10,10).astype(float),
            self.rng.normal(size=(4,10,10)),
            np.arange(4000).reshape(10,4,10,10).astype(np.uint16),
            self.rng.integers(65535, size=(10,4,10,10), dtype=np.uint16),
            np.arange(4000).reshape(10,4,10,10).astype(float),
            self.rng.normal(size=(10,4,10,10)),
        ]

    def test_encode_image(self):
        for image in self.images:
            encoded = constitch.visualization.encode_image(image)
            decoded = constitch.visualization.decode_image(encoded, image.shape, image.dtype)
            if np.abs(image - decoded).max() > 0.0001:
                print (image)
                print (decoded)
                print (image.shape, decoded.shape)
                print (np.abs(image - decoded).max())
            self.assertTrue(np.abs(image - decoded).max() <= 0.0001)

    def test_encode_image2(self):
        for image in self.images:
            while image.ndim > 2:
                image = image[0]
                encoded = constitch.visualization.encode_image(image)
                decoded = constitch.visualization.decode_image(encoded, image.shape, image.dtype)
                self.assertTrue(np.abs(image - decoded).max() <= 0.0001)

    def test_encode_quality(self):
        if not os.path.exists('viewer/tmp_input.tif'):
            return

        image = tifffile.imread('viewer/tmp_input.tif')[0]
        encoded = constitch.visualization.encode_image2(image)
        pilimage = PIL.Image.fromarray(encoded)

        parameters = []

        for quality in [25, 50, 75, 80, 90, 100]:
            for subsample in range(3):
                parameters.append(dict(quality=quality, subsampling=subsample))

        names = [''.join(key[0] + str(val) for key, val in params.items()) for params in parameters]

        filesizes = []
        errors = []

        for params in parameters:
            imageparams = dict(format='jpeg', quality=90, subsampling=2)
            imageparams.update(params)

            fileobj = io.BytesIO()
            pilimage.save(fileobj, **imageparams)

            fileobj.seek(0, 2)
            size = fileobj.tell()
            fileobj.seek(0)

            encoded2 = np.asarray(PIL.Image.open(fileobj)).copy()
            decoded = constitch.visualization.decode_image2(encoded2)

            filesizes.append(size)
            errors.append(image.astype(int) - decoded)

            #print (np.abs(errors[-1]).min(), np.abs(errors[-1]).mean(), np.abs(errors[-1]).max())

        fig, axes = plt.subplots(figsize=(15,15))

        #mse = [(errs * errs).mean() for errs in errors]
        mse = [np.abs(errs).mean() for errs in errors]
        axes.scatter(filesizes, mse)
        for name, size, err in zip(names, filesizes, mse):
            axes.annotate(name, (size, err))

        fig.savefig('plot_encoded_error.png')



if __name__ == '__main__':
    unittest.main()

