import unittest
import constitch
import numpy as np
import io

class TestComposite(unittest.TestCase):
    def setUp(self):
        self.composite = constitch.CompositeImage()

        images = np.zeros((10, 100, 100), dtype=np.uint16)
        xposes, yposes = np.meshgrid(np.arange(4), np.arange(4))
        poses = np.array([xposes.reshape(-1), yposes.reshape(-1)]).T * 75
        self.composite.add_images(images, poses)

    def test_subcomposite(self):
        mapping = [1,2,5,6]
        subcomposite = self.composite.subcomposite(mapping)

        for i in range(len(subcomposite.images)):
            self.assertIs(subcomposite.images[i], self.composite.images[mapping[i]])
        self.assertTrue(np.all(subcomposite.boxes.positions == self.composite.boxes.positions[mapping]))

        subcomposite.add_image(self.composite.images[0], (20,20))
        self.assertEqual(len(mapping), 5)
        self.assertEqual(mapping[-1], len(self.composite.images) - 1)
        self.assertIs(self.composite.images[0], self.composite.images[-1])
        self.assertEqual(subcomposite.boxes[-1].position.tolist(), [20, 20])

        for i in range(len(subcomposite.images)):
            self.assertIs(subcomposite.images[i], self.composite.images[mapping[i]])
        self.assertTrue(np.all(subcomposite.boxes.positions == self.composite.boxes.positions[mapping]))

        constraints = subcomposite.constraints(min_overlap=0)

        for const in constraints:
            self.assertIn(const.index1, mapping)
            self.assertIn(const.index2, mapping)

        constraint = subcomposite.constraints[0,1]
        self.assertEqual(constraint.pair, (mapping[0], mapping[1]))
        self.assertIs(constraint.image1, self.composite.images[mapping[0]])
        self.assertIs(constraint.image2, self.composite.images[mapping[1]])

    def test_layers(self):
        composite = constitch.CompositeImage(self.composite.images, self.composite.positions)
        newcomposite = constitch.CompositeImage(self.composite.images[:4], self.composite.positions[:4])
        newsub = composite.merge(newcomposite, new_layer=True)

        self.assertEqual(composite.boxes.positions.shape[1], 3)
        self.assertEqual(newsub.mapping, composite.layer(1).mapping)

    def test_saving(self):
        file = io.BytesIO()

        constraints = self.composite.constraints(min_overlap=0)

        constitch.save(file, self.composite, constraints)
        file.seek(0)
        newcomposite, newconsts = constitch.load(file)

        self.assertEqual(len(newcomposite.images), len(self.composite.images))
        self.assertEqual(newcomposite.boxes.positions.tolist(), self.composite.boxes.positions.tolist())

        self.assertEqual(list(constraints.keys()), list(newconsts.keys()))

        for const in newconsts:
            self.assertIs(const.composite, newcomposite)

    def test_boxes(self):
        def assertBox(box, **kwargs):
            for name, val in kwargs.items():
                attrval = getattr(box, name)
                if isinstance(attrval, np.ndarray): attrval = attrval.tolist()
                self.assertEqual((name, attrval), (name, val))

        box = fisseq.BBox([0,0], [5,3])

        # test different types setting
        assertBox(box, position=[0,0])
        box.position = 0
        assertBox(box, position=[0,0])
        box.position = (0,0)
        assertBox(box, position=[0,0])
        box.position = [0,0]
        assertBox(box, position=[0,0])
        box.position = np.array([0,0])
        assertBox(box, position=[0,0])
        box.position = np.array(0)
        assertBox(box, position=[0,0])
        box.position = np.array([0])
        assertBox(box, position=[0,0])

        # test relationships between attrs when setting
        assertBox(box, position=[0,0], size=[5,3], point1=[0,0], point2=[5,3])
        box.position = (4,6)
        assertBox(box, position=[4,6], size=[5,3], point1=[4,6], point2=[11,9])
        box.size = (2,1)
        assertBox(box, position=[4,6], size=[2,1], point1=[4,6], point2=[6,7])
        box.point1 = (-1,3)
        assertBox(box, position=[-1,3], size=[7,4], point1=[-1,3], point2=[6,7])
        box.point2 = (7,8)
        assertBox(box, position=[-1,3], size=[8,5], point1=[-1,3], point2=[6,7])



if __name__ == '__main__':
    unittest.main()

