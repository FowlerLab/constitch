from typing import Optional
import dataclasses
import enum

import numpy as np

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

        self.dx = dx
        self.dy = dy
        self.score = score
        self.error = error
        self.type = type

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
    def overlap_x(self):
        return min(self.box1.size[0] - self.dx, self.box2.size[0], self.box2.size[0] + self.dx, self.box1.size[0])
    @property
    def overlap_y(self):
        return min(self.box1.size[1] - self.dy, self.box2.size[1], self.box2.size[1] + self.dx, self.box1.size[1])
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
    def length(self):
        return (self.dx ** 2 + self.dy ** 2) ** 0.5


    def calculate(self, aligner=None):
        aligner = aligner or self.composite.aligner
        newconst = aligner.align(image1=self.image1, image2=self.image2, shape1=self.box1.size, shape2=self.box2.size, previous_constraint=self)
        #self.composite.add_constraint(newconst)
        return newconst

    def remove(self):
        return self.composite.remove_constraint(self)

    def add(self):
        return self.composite.add_constraint(self)


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
            elif getattr(constraint, name) != val:
                return False

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


class ConstraintSet:
    def __init__(self, composite, constraints=None, filters=None, ran_filters=None, last_update=0, never_update=False, pairs=None):
        self.composite = composite
        self.constraints = constraints
        self.filters = filters or ConstraintFilter()
        self.ran_filters = ran_filters or ConstraintFilter()
        self.last_update = None if never_update else last_update
        self.pairs = None

    def __str__(self):
        parts = []
        if self.constraints is not None and self.constraints_updated():
            parts.append('{} constraints'.format(len(self.constraints)))
        if not self.ran_filters.alwaystrue():
            parts.append(str(self.ran_filters)[17:-1]) # remove ConstraintFilter(...)
        if not self.filters.alwaystrue():
            parts.append(str(self.filters)[17:-1]) # remove ConstraintFilter(...)
        return 'ConstraintSet({})'.format(', '.join(parts))

    def constraints_updated(self):
        return self.last_update is None or self.composite.constraints_update_count == self.last_update

    def __and__(self, other):
        if isinstance(other, ConstraintSet):
            assert self.composite is other.composite, 'Cannot merge ConstraintSets that are from different composites'
            return ConstraintSet(self.composite,
                    constraints = list(set(self.constraints) & set(self.constraints)),
                    filters = self.filters & other.filters,
                    ran_filters = self.ran_filters & other.ran_filters,
                    last_update = min(self.last_update, other.last_update))

        elif isinstance(other, ConstraintFilter):
            return ConstraintSet(self.composite,
                    constraints = self.constraints,
                    filters = self.filters & other,
                    ran_filters = self.ran_filters,
                    last_update = self.last_update)

    def __or__(self, other):
        if isinstance(other, ConstraintSet):
            assert self.composite is other.composite, 'Cannot merge ConstraintSets that are from different composites'
            return ConstraintSet(self.composite,
                    constraints = list(set(self.constraints) | set(self.constraints)),
                    filters=self.filters | other.filters,
                    ran_filter = self.ran_filters | other.ran_filters,
                    last_update = min(self.last_update, other.last_update))

        elif isinstance(other, ConstraintFilter):
            return ConstraintSet(self.composite,
                    constraints = self.constraints,
                    filters = self.filters | other,
                    ran_filters = self.ran_filters,
                    last_update = self.last_update,
            )

    def filter(self, obj=None, **kwargs):
        if obj is None:
            obj = kwargs

        if isinstance(obj, np.ndarray) and obj.dtype == bool:
            self.runfilters()
            constraints = [const for const, valid in zip(self.constraints, obj) if valid]
            return ConstraintSet(self.composite, constraints=constraints)

        filters = ConstraintFilter.asfilter(obj)
        return self & filters

    def merge(self, obj=None, **kwargs):
        if obj is None:
            obj = kwargs
        filters = ConstraintFilter.asfilter(obj)
        return self | filters

    def runfilters(self):
        #if self.filters.alwaystrue() and (self.ran_filters.alwaystrue() or self.composite.constraints_update_count == self.last_update):
            #return
        if self.filters.alwaystrue() and self.constraints_updated():
            return

        newconstraints = []
        if self.constraints is not None and self.constraints_updated():
            newconstraints.extend(filter(self.filters, self.constraints))
        else:
            if 'type' in self.filters.equals:
                iterable = self.composite.constraintiter(type=self.filters.equals['type'], pairs=self.pairs)
            else:
                iterable = self.composite.constraintiter(pairs=self.pairs)
            newconstraints.extend(filter(self.ran_filters & self.filters, iterable))

        self.last_update = self.composite.constraints_update_count
        self.constraints = newconstraints
        self.ran_filters = self.ran_filters & self.filters
        self.filters = ConstraintFilter()

    def __iter__(self):
        self.runfilters()
        return iter(self.constraints)

    ATTRS = ['dx', 'dy', 'score', 'error', 'overlap', 'overlap_x', 'overlap_y', 'overlap_ratio', 'overlap_ratio_x', 'overlap_ratio_y', 'size']
    def __getattr__(self, name):
        if name not in self.ATTRS:
            return getattr(super(), name)
        runfilters()
        return np.array([getattr(const, name) for const in self.constraints])

    def calculate(self, aligner=None):
        self.runfilters()
        newconsts = [const.calculate(aligner=aligner) for const in self.constraints]
        return ConstraintSet(self.composite, constraints=newconsts, never_update=True)

    def remove(self):
        self.runfilters()
        newconsts = [const.remove() for const in self.constraints]
        return ConstraintSet(self.composite, constraints=newconsts, never_update=True)

    def solve(self, solver=None):
        self.runfilters()
        solver = solver or composite.solver
        constraints = {}
        poses = {}
        for const in self.constraints:
            poses[const.index1] = const.box1.pos1
            poses[const.index2] = const.box2.pos1
            constraints[const.index1,const.index2] = const

        return solver.solve(constraints, poses)

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

#"""
