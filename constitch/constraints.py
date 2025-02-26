from typing import Optional
import dataclasses
import itertools
import enum
import math
import warnings

import numpy as np
import sklearn.linear_model

class ConstraintType(enum.Enum):
    NORMAL = 'normal'
    MODELED = 'modeled'
    IMPLICIT = 'implicit'

@dataclasses.dataclass
class OldConstraint:
    dx: int
    dy: int
    score: Optional[float] = None
    overlap: Optional[float] = None
    modeled: bool = False
    error: int = 0


class Constraint:
    """ class to represent the pixel offset between two images in a composite.

    """

    def __init__(self, composite, index1=None, index2=None, dx=None, dy=None, score=None, error=None, type=ConstraintType.NORMAL):
        if isinstance(composite, Constraint):
            composite, index1, index2 = composite.composite, composite.index1, composite.index2
        else:
            assert index1 is not None and index2 is not None, "index1 and index2 are only optional when constructing from another constraint"

        self.composite = composite
        self.index1 = index1
        self.index2 = index2

        if dx is None or dy is None:
            type = ConstraintType.IMPLICIT
            dx, dy = self.box2.pos1[:2] - self.box1.pos1[:2]
            if error is None: error = math.inf

        self.dx = dx
        self.dy = dy
        self.score = score
        self.error = error
        self.type = type

    def to_obj(self):
        return dict(
            index1=self.index1, index2=self.index2,
            dx=self.dx, dy=self.dy,
            score=self.score,
            error=self.error,
            type=self.type,
        )

    def __str__(self):
        params = dict(dx=self.dx, dy=self.dy, score=self.score, error=self.error)
        return "Constraint({}, {}{})".format(self.index1, self.index2,
            ", ".join([''] + ["{}={}".format(name, val) for name, val in params.items() if val is not None]))

    def __repr__(self):
        return self.__str__()

    @property
    def modeled(self):
        return self.type == ConstraintType.MODELED
    @property
    def implicit(self):
        return self.type == ConstraintType.IMPLICIT

    @property
    def pair(self):
        return (self.index1, self.index2)

    @property
    def box1(self):
        return self.composite.boxes[self.index1]
    @property
    def box2(self):
        return self.composite.boxes[self.index2]

    @property
    def image1(self):
        return self.composite.images[self.index1]
    @property
    def image2(self):
        return self.composite.images[self.index2]

    @property
    def section1(self):
        """ Returns the section of image1 relevant for alignment, that is the
        section of image1 that overlaps with image2, expanded to include self.error.
        If error is infinite or None this will be the same as self.image1.
        """
        x1, x2, y1, y2 = self.section1_bounds
        return self.image1[x1:x2,y1:y2]

    @property
    def section2(self):
        """ Returns the section of image2 relevant for alignment, that is the
        section of image2 that overlaps with image1, expanded to include self.error.
        If error is infinite or None this will be the same as self.image1.
        """
        x1, x2, y1, y2 = self.section2_bounds
        return self.image2[x1:x2,y1:y2]

    @property
    def section1_bounds(self):
        if self.error is None or self.error == math.inf: return 0, self.box1.size[0], 0, self.box1.size[1]
        expand = self.error

        x1 = max(0, self.dx - expand)
        x2 = min(self.box1.size[0], self.box2.size[0] + self.dx + expand)
        y1 = max(0, self.dy - expand)
        y2 = min(self.box1.size[1], self.box2.size[1] + self.dy + expand)

        return x1, x2, y1, y2

    @property
    def section2_bounds(self):
        if self.error is None or self.error == math.inf: return 0, self.box2.size[0], 0, self.box2.size[1]
        expand = self.error

        x1 = max(0, -self.dx - expand)
        x2 = min(self.box2.size[0], self.box1.size[0] - self.dx + expand)
        y1 = max(0, -self.dy - expand)
        y2 = min(self.box2.size[1], self.box1.size[1] - self.dy + expand)

        return x1, x2, y1, y2

    @property
    def overlap_x(self):
        return min(self.box1.size[0] - self.dx, self.box2.size[0], self.box2.size[0] + self.dx, self.box1.size[0])
    @property
    def overlap_y(self):
        return min(self.box1.size[1] - self.dy, self.box2.size[1], self.box2.size[1] + self.dy, self.box1.size[1])
    @property
    def overlap(self):
        overlaps = self.overlap_x, self.overlap_y
        if overlaps[0] < 0 and overlaps[1] < 0: return -overlaps[0] * overlaps[1]
        return overlaps[0] * overlaps[1]

    @property
    def overlap_ratio_x(self):
        return self.overlap_x / min(self.box1.size[0], self.box2.size[0])
    @property
    def overlap_ratio_y(self):
        return self.overlap_y / min(self.box1.size[1], self.box2.size[1])
    @property
    def overlap_ratio(self):
        return self.overlap / min(self.box1.size.prod(), self.box2.size.prod())

    @property
    def touching(self):
        return self.overlap >= 0 and (self.overlap_x > 0 or self.overlap_y > 0)

    @property
    def length(self):
        return (self.dx ** 2 + self.dy ** 2) ** 0.5

    @property
    def difference(self):
        return (self.dx, self.dy) - (self.box2.pos1[:2] - self.box1.pos1[:2])

    def calculate(self, aligner=None, executor=None):
        return self.calculate_future(aligner, executor).result()

    def calculate_future(self, aligner=None, executor=None):
        aligner = aligner or self.composite.aligner
        executor = executor or self.composite.executor
        #newconst = aligner.align(image1=self.image1, image2=self.image2, shape1=self.box1.size, shape2=self.box2.size, previous_constraint=self)
        future = executor.submit(align_job, aligner, constraint=self)
        #future = executor.submit(align_job, aligner, image1=self.image1, image2=self.image2, shape1=self.box1.size, shape2=self.box2.size, previous_constraint=self)
        #self.composite.add_constraint(newconst)
        return future

    def expand_overlap(self, amount):
        newconst = Constraint(self, dx=self.dx, dy=self.dy, score=self.score, error=self.error, type=self.type)

        if type(amount) in (int, float):
            amount = (amount, amount)

        amount = min(amount[0], abs(self.dx)), min(amount[1], abs(self.dy))

        if newconst.dx < 0: newconst.dx += amount[0]
        else: newconst.dx -= amount[0]

        if newconst.dy < 0: newconst.dy += amount[1]
        else: newconst.dy -= amount[1]

        return newconst

    def new(self, dx=None, dy=None, section_dx=None, section_dy=None, score=None, error=None, type=None):
        if section_dx is not None:
            dx = self.section1_bounds[0] - self.section2_bounds[0] + section_dx
            dy = self.section1_bounds[2] - self.section2_bounds[2] + section_dy

        return Constraint(self, dx=dx, dy=dy, score=score, error=error, type=type)

def align_job(aligner, **kwargs):
    return aligner.align(**kwargs)

class ConstraintFilter:
    def __init__(self, func=None, mins=None, maxes=None, equals=None):
        self.func = func
        self.mins = mins or {}
        self.maxes = maxes or {}
        self.equals = equals or {}

    def __str__(self):
        parts = []
        parts.extend('{} >= {}'.format(name, val) for name, val in self.mins.items())
        parts.extend('{} == {}'.format(name, val) for name, val in self.equals.items())
        parts.extend('{} <= {}'.format(name, val) for name, val in self.maxes.items())
        return 'ConstraintFilter({})'.format(', '.join(parts))

    @classmethod
    def fromdict(cls, params):
        mins, maxes, equals = {}, {}, {}

        for name, val in params.items():
            if name[:4] == 'min_':
                mins[name[4:]] = val
            elif name[:4] == 'max_':
                maxes[name[4:]] = val
            else:
                equals[name] = val

        return cls(mins=mins, maxes=maxes, equals=equals)

    @classmethod
    def asfilter(cls, obj):
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, dict):
            return cls.fromdict(obj)

        if callable(obj):
            return cls(obj)

        raise "Unexpected type in ConstraintFilter.asfilter"

    def __call__(self, constraint):
        #print (' running filter', self, constraint, constraint.overlap_ratio)
        if self.func is not None and not self.func(constraint):
            return False

        for name, val in self.mins.items():
            if getattr(constraint, name) < val: return False

        for name, val in self.maxes.items():
            if getattr(constraint, name) > val: return False

        for name, val in self.equals.items():
            if getattr(constraint, name) != val: return False

        #print ('    True')
        return True

    def __and__(self, other):
        mins = self.mins.copy()
        for name, val in other.mins.items():
            mins[name] = max(mins.get(name, val), val)

        maxes = self.maxes.copy()
        for name, val in other.maxes.items():
            maxes[name] = max(maxes.get(name, val), val)

        equals = self.equals.copy()
        for name, val in other.equals.items():
            if name in equals and val != equals[name]: #short circuit to always false
                return ConstraintFilter(lambda constraint: False)
            equals[name] = val

        func = self.func
        if other.func is not None:
            if func is None:
                func = other.func
            else:
                func = lambda constraint, filter1=func, filter2=other.func: filter1(constraint) and filter2(constraint)

        return ConstraintFilter(func=func, mins=mins, maxes=maxes, equals=equals)

    def __or__(self, other):
        return ConstraintFilter(lambda constraint, filter1=self, filter2=other: filter1(constraint) or filter2(constraint))

    def alwaystrue(self):
        return self.func is None and len(self.mins) == 0 and len(self.maxes) == 0 and len(self.equals) == 0

#"""
class ConstraintSet:
    """ A class that stores a set of constraints, and can perform 
    """
    def __init__(self, constraints=None):
        self.constraints = {}
        if constraints:
            self.add(constraints)

    @property
    def composite(self):
        if len(self.constraints):
            return next(iter(self.constraints.values()))[0].composite

    def debug(self, *args, **kwargs):
        if self.composite:
            self.composite.debug(*args, **kwargs)

    def progress(self, iter, **kwargs):
        if self.composite:
            return self.composite.progress(iter, **kwargs)
        return iter

    def add(self, other):
        if isinstance(other, Constraint):
            self._add_single(other)
        else:
            for const in self._constraint_iter(other):
                self._add_single(const)

    def remove(self, other):
        if isinstance(other, Constraint) or (type(other) == tuple and len(other) == 2):
            self._remove_single(other)
        else:
            for const in self._constraint_iter(other):
                self._remove_single(const)

    def merge(self, other):
        new_set = ConstraintSet()
        new_set.add(self)
        new_set.add(other)
        return new_set

    def find(self, obj=None, **kwargs):
        return next(iter(self.filter(obj, limit=1, **kwargs)))

    def filter(self, obj=None, limit=None, **kwargs):
        if isinstance(obj, dict):
            newset = ConstraintSet()
            for pair, val in obj.items():
                for const in self.constraints[pair][int(not val):]:
                    newset.add(const)
            return newset

        if obj is None:
            obj = kwargs

        filters = ConstraintFilter.asfilter(obj)
        if limit is None:
            newset = ConstraintSet(filter(filters, self._constraint_iter()))
        else:
            newset = ConstraintSet(filter(filters, itertools.islice(self._constraint_iter(), limit)))

        return newset

    def __iter__(self):
        return iter(const_list[0] for const_list in self.constraints.values())

    def __len__(self):
        return len(self.constraints)

    def __getitem__(self, pair):
        return self.constraints[pair][0]

    def __contains__(self, obj):
        if isinstance(obj, Constraint):
            return obj.pair in self.constraints and obj in self.constraints[obj.pair]
        else:
            return obj in self.constraints

    def keys(self):
        return self.constraints.keys()

    def values(self):
        return (const_list[0] for const_list in self.constraints.values())

    def items(self):
        return ((pair, const_list[0]) for pair, const_list in self.constraints.items())


    ATTRS = ['dx', 'dy', 'score', 'error', 'overlap', 'overlap_x', 'overlap_y', 'overlap_ratio', 'overlap_ratio_x', 'overlap_ratio_y', 'size', 'difference']
    def __getattr__(self, name):
        if name not in self.ATTRS:
            return getattr(super(), name)
        runfilters()
        return np.array([getattr(const, name) for const in self.constraints])

    def neighborhood_difference(self, constraint):
        touching_constraints = self.filter(lambda const: const.index1 in constraint.pair or const.index2 in constraint.pair)
        diffs = touching_constraints.difference
        max_diff = np.max(np.linalg.norm(diffs))
        curdiff = np.linalg.norm(constraint.difference)
        if max_diff > curdiff:
            curdiff = 0
        return curdiff

    def calculate(self, aligner=None, executor=None):
        futures = [const.calculate_future(aligner=aligner, executor=executor) for const in self]
        newset = ConstraintSet(future.result() for future in self.progress(futures))
        self.debug("Calculated", len(futures), "new constraints")
        #newset = ConstraintSet(const.calculate(aligner=aligner) for const in self)
        return newset

    def fit_model(self, model=None, outliers=False, random_state=12345):
        from . import stage_model
        model = model or stage_model.SimpleOffsetModel()

        if outliers:
            model = sklearn.linear_model.RANSACRegressor(model,
                    min_samples=4,
                    max_trials=1000,
                    random_state=random_state)

        est_poses = []
        const_poses = []
        indices = []
        for constraint in self:
            est_poses.append(np.concatenate([constraint.box1.pos1, constraint.box2.pos1]))
            const_poses.append((constraint.dx, constraint.dy))
            indices.append(constraint.pair)

        est_poses, const_poses = np.array(est_poses), np.array(const_poses)
        indices = np.array(indices)

        model.fit(est_poses, const_poses)
        #print (model.estimator.model.coef_)

        aligner = stage_model.StageModelAligner(model.estimator_ if outliers else model)

        if outliers:
            self.debug ('Filtered out', np.sum(~model.inlier_mask_), 'constraints as outliers')

            if np.mean(model.inlier_mask_.astype(int)) < 0.8:
                warnings.warn("Stage model filtered out over 20% of constraints as outilers."
                        " It may have hyperoptimized to the data, make sure all are actually outliers")

            est_poses, const_poses = est_poses[model.inlier_mask_], const_poses[model.inlier_mask_]
            self.debug ("Estimated stage model", model, "with an r2 score of", model.score(est_poses, const_poses),
                    ", classifying {}/{} constraints as outliers".format(np.sum(~model.inlier_mask_), len(self.constraints)))

            aligner.inliers = dict(zip(self.constraints.keys(), model.inlier_mask_))
            aligner.outlier = dict(zip(self.constraints.keys(), ~model.inlier_mask_))

        else:
            self.debug ("Estimated stage model", model, "with an r2 score of", model.score(est_poses, const_poses))

        if (aligner.model.predict([[0] * est_poses.shape[1]]).max() > const_poses.max() * 100 or 
                aligner.model.predict([[1] * est_poses.shape[1]]).max() > const_poses.max() * 100):
            warnings.warn("Stage model is predicting very large values for simple constraints,"
                " it may have hyperoptimized to the training data.")

        # calculate variance
        error = aligner.model.predict(est_poses) - const_poses
        error_thresh = np.percentile(np.abs(error), 99)
        self.debug ("Stage model error", np.percentile(np.abs(error), [0,5,50,75,95,100]).tolist(), error_thresh)
        aligner.error = error_thresh

        return aligner

    def solve(self, solver=None):
        from . import solving
        solver = solver or solving.LinearSolver()
        constraints = {}
        poses = {}
        for const in self:
            poses[const.index1] = const.box1.pos1
            poses[const.index2] = const.box2.pos1
            constraints[const.pair] = const

        newposes = solver.solve(constraints, poses)
        if type(newposes) == tuple:
            newposes, constraints = newposes
            pairs = set(constraints.keys())
            for pair in list(self.constraints.keys()):
                if pair not in pairs:
                    self._remove_single(pair)

        return newposes


    def _add_single(self, constraint):
        assert len(self) == 0 or constraint.composite is self.composite, "Adding constraints from different composite"

        const_list = self.constraints.setdefault(constraint.pair, [])
        index = 0
        while index < len(const_list) and const_list[index].error < constraint.error:
            index += 1
        if index >= len(const_list) or const_list[index] != constraint:
            const_list.insert(index, constraint)
        return const_list[0]

    def _remove_single(self, constraint):
        pair = constraint
        if isinstance(constraint, Constraint):
            pair = constraint.pair

        const_list = self.constraints[pair]

        index = 0
        if isinstance(constraint, Constraint):
            index = const_list.index(constraint)
        const_list.pop(index)

        if len(const_list) == 0:
            del self.constraints[pair]

    def _constraint_iter(self, constraints=None):
        constraints = constraints or self.constraints
        if isinstance(constraints, ConstraintSet):
            constraints = constraints.constraints
        if isinstance(constraints, dict):
            for pair, constraint in constraints.items():
                if type(constraint) == list:
                    for const in constraint:
                        yield const
                else:
                    yield constraint
        else:
            for constraint in constraints:
                yield constraint



class ImplicitConstraintDict(dict):
    def __init__(self, composite, pairs_func):
        self.composite = composite
        self.pairs_func = pairs_func

    def keys(self):
        for pair in self.pairs_func():
            yield pair

    def values(self):
        for pair in self.pairs_func():
            yield self[pair]

    def items(self):
        for pair in self.pairs_func():
            yield pair, self[pair]

    def __getitem__(self, pair):
        return [Constraint(self.composite, index1=pair[0], index2=pair[1])]



"""
composite = constitch.CompositeImage()
composite.add_images(images, poses)

overlapping = composite.newconstraints(min_overlap=0.1)
constraints = overlapping.calculate()
constraints.remove(min_score=0.5, min_overlap=0.1)
modeled_constraints = overlapping.calculate(constraints.fit_model(remove_outliers=True))

constraints = composite.newconstraints(min_overlap=0.1).calculate(constitch.FFTAligner())
constraints.remove(min_score=0.5, min_overlap=0.1)
constraints.model(

constraints.calculate(constitch.FFTAligner())
constraints.filter(constraints.score < 0.5)
constraints.solve(constitch.OutlierSolver())

composite.apply(constraints.solve())
composite.stitch()




composite = constitch.CompositeImage(images, poses)

overlapping = composite.constraints(min_overlap=0.1)
constraints = overlapping.calculate(constitch.FFTAligner())
constraints = constraints.filter(min_score=0.5, min_overlap=0.1, max_length=max(images.shape))

stage_model = constraints.fit_model()
constraints = constraints.filter(stage_model.inliers)
modeled = overlapping.calculate(stage_model)

constraints = constraints.merge(modeled)
composite.apply(constraints.solve())

solver = constitch.LinearSolver()
composite.apply(solver.solve(constraints, modeled, overlapping))

composite.apply(constraints.solve(modeled, overlapping))
constraints = overlapping.merge(modeled).merge(constraints)
composite.apply(constraints.solve())

constraints.remove(min_score=0.5, min_overlap=0.1)
modeled_constraints = overlapping.calculate(constraints.fit_model(remove_outliers=True))

constraints = composite.newconstraints(min_overlap=0.1).calculate(constitch.FFTAligner())
constraints.remove(min_score=0.5, min_overlap=0.1)
constraints.model(

constraints.calculate(constitch.FFTAligner())
constraints.filter(constraints.score < 0.5)
constraints.solve(constitch.OutlierSolver())

composite.apply(constraints.solve())
composite.stitch()

#"""
