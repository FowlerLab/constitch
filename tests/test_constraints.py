import unittest
import constitch
import numpy as np

class TestConstraint(unittest.TestCase):
    def setUp(self):
        self.composite = constitch.CompositeImage()

        images = np.zeros((4, 100, 100), dtype=np.uint16)
        xposes, yposes = np.meshgrid(np.arange(2), np.arange(2))
        poses = np.array([xposes.reshape(-1), yposes.reshape(-1)]).T * 75
        self.composite.add_images(images, poses)

    def test_overlap(self):
        constraint = self.composite.constraints[0,1]
        self.assertEqual(constraint.overlap_x, 25)
        self.assertEqual(constraint.overlap_y, 100)
        self.assertEqual(constraint.overlap_ratio_x, 0.25)
        self.assertEqual(constraint.overlap_ratio_y, 1.0)
        self.assertEqual(constraint.overlap, 2500)
        self.assertEqual(constraint.overlap_ratio, 0.25)

    def test_filters(self):
        constraints = self.composite.constraints(min_overlap=0)
        for const in constraints:
            self.assertTrue(const.overlap >= 0)

        constraints = constraints.filter(min_overlap_x=0, max_overlap_x=50)
        for const in constraints:
            self.assertTrue(const.overlap >= 0)
            self.assertTrue(const.overlap_x >= 0)
            self.assertTrue(const.overlap_x <= 50)

if __name__ == '__main__':
    print ('running')
    unittest.main()

