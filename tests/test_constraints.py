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

        self.small_composite = constitch.CompositeImage()
        images = np.arange(4 * 4 * 4).reshape(4, 4, 4)
        poses = np.array([xposes.reshape(-1), yposes.reshape(-1)]).T * 3
        self.small_composite.add_images(images, poses[:4])

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

    def test_solving(self):
        constraints = self.composite.constraints(touching=True)
        for const in constraints:
            const.score = 0.5
        result = constraints.solve()
        for index, pos in result.positions.items():
            self.assertEqual(tuple(pos), tuple(self.composite.boxes[index].position))

        prevposes = self.composite.boxes.positions.tolist()
        self.composite.setpositions(result)
        self.assertEqual(prevposes, self.composite.boxes.positions.tolist())

    def test_solving_outlier(self):
        constraints = self.composite.constraints(touching=True)
        for const in constraints:
            const.score = 0.5

        constraints[0,1].dx += 500
        result = constraints.solve()
        tmp = self.composite.copy()
        #tmp.plot_scores('tmp_consts.png', constraints, 'accuracy')
        tmp.setpositions(result)
        #tmp.plot_scores('tmp_consts.png', constraints, 'accuracy')

        #self.assertNotEqual(tuple(self.composite.boxes[0].position), tuple(result.positions[0]))
        self.assertNotEqual(tuple(self.composite.boxes[1].position), tuple(result.positions[1]))

        for const in constraints:
            neigh_score = constraints.neighborhood_difference(const)
            if const.pair == (0,1):
                self.assertTrue(neigh_score > 5)
            else:
                self.assertEqual(neigh_score, 0)

        constraints = constraints.filter(lambda const: constraints.neighborhood_difference(const) < 5)
        self.assertTrue((0,1) not in constraints)
        result = constraints.solve()

        for index, pos in result.positions.items():
            self.assertEqual(tuple(pos), tuple(self.composite.boxes[index].position))

    def test_solving_optimal(self):
        constraints = self.composite.constraints(touching=True)
        for const in constraints:
            const.score = 0.5

        constraints[0,1].dx += 500
        result = constraints.solve(constitch.QuantileSolver())
        tmp = self.composite.copy()
        #tmp.plot_scores('tmp_consts_outlier.png', constraints, 'accuracy')
        tmp.setpositions(result)
        #tmp.plot_scores('tmp_consts_outlier2.png', constraints, 'accuracy')

        self.assertEqual(tuple(self.composite.boxes[1].position), tuple(result.positions[1]))

    def test_random(self):
        for bounds in [5, 200, 3498, 339, 11]:
            values = set()
            value = 1
            while value not in values:
                values.add(value)
                value = constitch.utils.lfsr(value, bounds)
                self.assertTrue(value <= bounds and value > 0)
            self.assertEqual(len(values), bounds)

        constraints = self.composite.constraints()
        constraints2 = self.composite.constraints(random=True)
        self.assertEqual(set(const.pair for const in constraints), set(const.pair for const in constraints2))

    def test_section(self):
        images = [
            np.arange(4).reshape(2,2),
            np.arange(6).reshape(2,3),
            np.arange(9).reshape(3,3),
        ]
        boxes = [constitch.BBox([0,0], [2,2]), constitch.BBox([1,1], [2,3]), constitch.BBox([0,0], [2,2])]
        composite = constitch.CompositeImage(images=images, boxes=boxes)

        constraints = composite.constraints()

        for const in constraints:
            print (const.box1, const.box2)
            print (const.image1)
            print (const.image2)
            print (const.resized_image1)
            print (const.resized_image2)
            print (const.section1)
            print (const.section2)
            print ()

if __name__ == '__main__':
    unittest.main()

