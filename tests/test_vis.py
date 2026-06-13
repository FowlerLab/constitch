import unittest
import constitch
import constitch.visualization
import numpy as np
import io

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

if __name__ == '__main__':
    unittest.main()

