from typing import Optional
import dataclasses
import enum

class ConstraintType(enum.Enum):
    NORMAL = 'normal'
    MODELED = 'modeled'
    IMPLICIT = 'implicit'

@dataclasses.dataclass
class Constraint:
    dx: int
    dy: int
    score: Optional[float] = None
    overlap: Optional[float] = None
    modeled: bool = False
    error: int = 0


class NewConstraint:
    def __init__(self, composite, index1, index2, dx=None, dy=None, score=None, error=None, type=ConstraintType.NORMAL):
        self.composite = composite
        self.index1 = index1
        self.index2 = index2
        self.dx = dx
        self.dy = dy
        self.score = score
        self.error = error
        self.type = type

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
    def overlap_x(self):
        return max(self.box1.pos2[0] - self.box2.pos1[0], self.box2.pos2[0] - self.box1.pos1[0])
    @property
    def overlap_y(self):
        return max(self.box1.pos2[1] - self.box2.pos1[1], self.box2.pos2[1] - self.box1.pos1[1])
    @property
    def overlap(self):
        return self.overlap_x * self.overlap_y

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

    def remove(self):
        self.composite.remove(self)

    def calculate(self, aligner=None):
        aligner = aligner or self.composite.aligner
        return 

class ConstraintFilter:
    def __init__(self, func=None, mins=None, maxes=None, equals=None):
        self.func = func
        self.mins = mins or {}
        self.maxes = maxes or {}
        self.equals = equals or {}

    @classmethod
    def fromdict(cls, params):
        mins, maxes, equals = {}, {}, {}

        for name, val in params.items():
            if name[:4] == 'min_':
                mins[name[4:]] = val
            elif name[:4] == 'max_':
                maxes[name[:4]] = val
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
        if self.func is not None and not self.func(constraint):
            return False

        for name, val in self.mins.items():
            if getattr(constraint, name) < val: return False

        for name, val in self.maxes.items():
            if getattr(constraint, name) > val: return False

        for name, val in self.equals.items():
            if getattr(constraint, name) != val: return False

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
        equals.update(other.equals)

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
        return self.func is None and self.mins is None and self.maxes is None and self.equals is None


class ConstraintSet:
    def __init__(self, composite, constraints=None, filters=None):
        self.composite = composite
        self.constraints = constraints or []
        self.filters = filters or ConstraintFilter()

    def __and__(self, other):
        if isinstance(other, ConstraintSet):
            assert self.composite is other.composite, 'Cannot merge ConstraintSets that are from different composites'
            return ConstraintSet(self.composite, constraints=list(set(self.constraints) & set(self.constraints)), filters=self.filters & other.filters)
        elif isinstance(other, ConstraintFilter):
            return ConstraintSet(self.composite, constraints=self.constraints, filters=self.filters & other)

    def __or__(self, other):
        if isinstance(other, ConstraintSet):
            assert self.composite is other.composite, 'Cannot merge ConstraintSets that are from different composites'
            return ConstraintSet(self.composite, constraints=list(set(self.constraints) | set(self.constraints)), filters=self.filters | other.filters)
        elif isinstance(other, ConstraintFilter):
            return ConstraintSet(self.composite, constraints=self.constraints, filters=self.filters | other)

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
        if self.filters.alwaystrue():
            return

        if 'type' in self.filters.equals:
            iterable = self.composite.constraintiter(type=self.filters.equals['type'])
        else:
            iterable = self.composite.constraintiter()

        self.constraints.extend(filter(self.filters, iterable))
        self.filters = ConstraintFilter()

    def __iter__(self):
        self.runfilters()
        return iter(self.constraints)

    ATTRS = ['dx', 'dy', 'score', 'error', 'overlap', 'overlap_x', 'overlap_y', 'overlap_ratio', 'overlap_ratio_x', 'overlap_ratio_y', 'size']
    def __getattr__(self, name):
        if name not in self.ATTRS:
            super().__getattr__(name)
        runfilters()
        return np.array([getattr(const, name) for const in self.constraints])


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
