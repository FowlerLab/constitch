import unittest
import constitch
import numpy as np

class TestConstraint(unittest.TestCase):
    def setUp(self):
        self.composite = constitch.CompositeImage()

        images = np.zeros((16, 100, 100), dtype=np.uint16)
        xposes, yposes = np.meshgrid(np.arange(4), np.arange(4))
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

    def test_neighboring(self):
        constraints = self.composite.constraints(touching=True)
        self.assertTrue(np.all(constraints.difference == 0))

        constraints[0,1].dx += 5
        constraints[1,2].dx += 10
        self.assertEqual(constraints[0,1].difference.tolist(), [5, 0])
        self.assertEqual(constraints[1,2].difference.tolist(), [10, 0])

        self.assertEqual(constraints.neighborhood_difference(constraints[0,1]), 0)
        self.assertEqual(constraints.neighborhood_difference(constraints[1,2]), 10)

        #self.composite.plot_scores('tmp_consts.png', constraints, 'accuracy')
        neighboring = constraints.neighboring(constraints[0,1], 2)
        #self.composite.plot_scores('tmp_consts2.png', neighboring, 'accuracy')
        expected = {(0, 4), (0, 5), (1, 2), (1, 4), (1, 5), (1, 6), (2, 3), (2, 5), (2, 6), (2, 7), (3, 6), (4, 5), (4, 8), (4, 9), (5, 6), (5, 8), (5, 9), (5, 10), (6, 7), (6, 9), (6, 10), (6, 11)}
        self.assertEqual(set(neighboring.constraints.keys()), expected)

        neighboring = constraints.neighboring([0, 15], 1)
        expected = {(0, 1), (0, 4), (0, 5), (14, 15), (11, 15), (10, 15)}
        self.assertEqual(set(neighboring.constraints.keys()), expected)

if __name__ == '__main__':
    print ('running')
    unittest.main()

