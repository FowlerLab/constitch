import os
import sys
import time
import numpy as np
import sklearn.linear_model
import skimage.io
import scipy.optimize

from .constraints import Constraint, ConstraintSet


def calc_box_score(scores, boxes, overlapping_indices, box, index):
    area = box.area()
    overlapping_indices.append(index)

    for i in range(len(scores)):
        intersection = box.intersection(boxes[i])
        intersection_area = intersection.area()
        if intersection_area <= 0:
            continue

        if i != index:
            area -= intersection_area

        if i > index:
            calc_box_score(scores, boxes, overlapping_indices, intersection, i)

    for i in overlapping_indices:
        scores[i] += area / len(overlapping_indices)

    overlapping_indices.pop(-1)

def constraint_scores(constraints, index):
    indices = []
    scores = []
    boxes = []

    for const in constraints.filter(index2=index):
        indices.append(const.index1)
        scores.append(0)
        boxes.append(const.intersection())

    for const in constraints.filter(index1=index):
        indices.append(const.index2)
        scores.append(0)
        boxes.append(const.intersection())

    for i in range(len(indices)):
        calc_box_score(scores, boxes, [], boxes[i], i)

    return dict(zip(indices, scores))


class Solver:
    """ Base class that takes in all the constraints of a composite
    and solves them into global positions
    """

    def solve(self, constraints, initial_poses):
        """ Solve the global positions for images given the constraints and estimated positions.
        returns the xy position for each image, mapping the image index to the position
        with a dictionary.
        Additionally the constraints dictionary can be returned as the second return value, which
        will replace the constraints in the composite with the ones returned, filtering outliers.
        """
        pass


class LinearSolver(Solver):
    """ Solver that represents the constraints as an overconstrained system of equations, and
    solves it using least squares.
    """

    def __init__(self, model=None):
        self.model = model or sklearn.linear_model.LinearRegression(fit_intercept=False)

    def make_constraint_matrix(self, constraints, initial_poses):
        image_indices = sorted(list(initial_poses.keys()))
        solution_mat = np.zeros((len(constraints)*2+2, len(image_indices)*2))
        solution_vals = np.zeros(len(constraints)*2+2)
        
        for index, ((id1, id2), constraint) in enumerate(constraints.items()):
            id1, id2 = image_indices.index(id1), image_indices.index(id2)
            dx, dy = constraint.dx, constraint.dy
            score = self.score_func(constraint)

            solution_mat[index*2, id1*2] = -score
            solution_mat[index*2, id2*2] = score
            solution_vals[index*2] = score * dx

            solution_mat[index*2+1, id1*2+1] = -score
            solution_mat[index*2+1, id2*2+1] = score
            solution_vals[index*2+1] = score * dy

        # anchor tile 0 to 0,0, otherwise there are inf solutions
        solution_mat[-2, 0] = 1
        solution_mat[-1, 1] = 1

        initial_values = np.array(list(initial_poses.values()))

        return solution_mat, solution_vals, initial_values

    def make_positions(self, initial_poses, poses):
        return dict(zip(sorted(list(initial_poses.keys())), poses))

    def solve(self, constraints, initial_poses):
        orig_constraints = constraints.copy()
        #image_indices = sorted(list(set(pair[0] for pair in constraints) | set(pair[1] for pair in constraints)))
        for i in range(1):
            solution_mat, solution_vals, initial_values = self.make_constraint_matrix(constraints, initial_poses)

            solution = self.solve_matrix(solution_mat, solution_vals, initial_values)

            poses = solution.reshape(-1,2)

            residuals = np.matmul(solution_mat, solution) - solution_vals
            residuals = residuals.reshape(-1,2)
            print (np.mean(np.abs(residuals)), file=sys.stderr)
            self.constraints_accuracy = dict(zip(constraints.keys(), residuals))

            newconsts = {}
            for i, pair in enumerate(constraints.keys()):
                const = constraints[pair]
                offset = poses[pair[1]] - poses[pair[0]]
                #offset = np.round(offset).astype(int)
                newconst = Constraint(const.composite, pair[0], pair[1], offset[0], offset[1], const.score, const.error)
                newconsts[pair] = newconst

            constraints = newconsts

        # find offset that minimizes error from rounding
        #for i in range(2):
            #frac_part = np.mod(poses[:,i], 1)
            #frac_mean = (np.sum(np.sin(frac_part * 2 * np.pi)), np.sum(np.cos(frac_part * 2 * np.pi)))
            #if frac_mean != (0,0):
                #print ('frac_mean', frac_mean, file=sys.stderr)
                #frac_mean = np.arctan2(*frac_mean) / (2 * np.pi)
                #print (frac_mean, file=sys.stderr)
                #poses[:,i] -= frac_mean

        poses = np.round(poses).astype(int)
        poses -= poses.min(axis=0).reshape(1,2)

        residuals = np.matmul(solution_mat, poses.reshape(-1)) - solution_vals
        residuals = residuals.reshape(-1,2)
        print ('after round', np.mean(np.abs(residuals)), file=sys.stderr)

        return self.make_positions(initial_poses, poses)

        rounded_poses = SpanningTreeSolver(score_func=lambda const: const.overlap_ratio).solve(constraints, initial_poses)
        errors = np.array([np.linalg.norm(np.array(rounded_poses[const.index2]) - rounded_poses[const.index1] - (const.dx, const.dy)) for const in constraints.values()])
        print (np.mean(np.abs(errors)), file=sys.stderr)
        errors = np.array([np.linalg.norm(np.array(rounded_poses[const.index2]) - rounded_poses[const.index1] - (const.dx, const.dy)) for const in orig_constraints.values()])
        print (np.mean(np.abs(errors)), file=sys.stderr)

        poses = np.array([rounded_poses[i] for i in initial_poses.keys()])
        #poses = np.array(list(rounded_poses.values()))

        residuals = np.matmul(solution_mat, poses.reshape(-1)) - solution_vals
        residuals = residuals.reshape(-1,2)
        print ('after round', np.mean(np.abs(residuals)), file=sys.stderr)
        return rounded_poses

    def solve_matrix(self, solution_mat, solution_vals, initial_values):
        #solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
        model = self.model.fit(solution_mat, solution_vals)
        solution = model.coef_
        return solution

    def score_func(self, constraint):
        return max(0, constraint.score) * max(0, constraint.overlap_ratio)


class MAESolver(LinearSolver):
    """ Solver that performs quantile regression instead of ordinary least squares
    to solve the system of equations described by constraints. Equivalent to minimizing
    the mean absolute error, this is much more outlier resistant that minimizing MSE
    and should provide better results when there are erroneous constraints present.
    This is identical to LinearSolver except that the linear model used is QuantileRegressor.
    Any parameters passed to the constructor are forwared to the constructor of
    sklearn.linear_model.QuantileRegressor. By default the L1 regularization constant
    alpha=0 and fit_intercept=False, unless specified in the constructor.
    """
    def __init__(self, **kwargs):
        params = dict(alpha=0, fit_intercept=False, solver='highs')
        params.update(kwargs)
        super().__init__(model=sklearn.linear_model.QuantileRegressor(**params))

    def score_func(self, constraint):
        return max(0, constraint.score) * max(0, constraint.overlap_ratio)

class LPSolver(LinearSolver):
    """ Solver that uses (integer) linear programming to find a solution minimizing the mean
    absolute error of the system of equations specified by the constraints. This
    differs from MAESolver as by using integer linear programming we can constrain
    the resulting values to be integers, which removes any errors that might come from
    rounding the solution values.
    """
    def __init__(self, integral=True):
        super().__init__(model=None)
        self.integral = integral

    def solve_matrix(self, solution_mat, solution_vals, initial_values):
        ### Below code mostly from sklearn.linear_model.QuantileRegressor:
        ### https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/linear_model/_quantile.py#L20

        # After rescaling alpha, the minimization problem is
        #     min sum(pinball loss) + alpha * L1
        # Use linear programming formulation of quantile regression
        #     min_x c x
        #           A_eq x = b_eq
        #                0 <= x
        # x = (s0, s, t0, t, u, v) = slack variables >= 0
        # intercept = s0 - t0
        # coef = s - t
        # c = (0, alpha * 1_p, 0, alpha * 1_p, quantile * 1_n, (1-quantile) * 1_n)
        # residual = y - X@coef - intercept = u - v
        # A_eq = (1_n, X, -1_n, -X, diag(1_n), -diag(1_n))
        # b_eq = y
        # p = n_features
        # n = n_samples
        # 1_n = vector of length n with entries equal one

        n_features = solution_mat.shape[1]
        n_params = n_features
        n_indices = solution_mat.shape[0]
        quantile = 0.5

        c = np.concatenate(
            [
                np.zeros(2 * n_params),
                np.full(n_indices, quantile),
                np.full(n_indices, 1 - quantile),
            ]
        )

        #eye = np.eye(n_indices, dtype=solution_mat.dtype)
        #print (solution_mat.shape, eye.shape, file=sys.stderr)
        #A_eq = np.concatenate([solution_mat, -solution_mat, eye, -eye], axis=1)
        eye = scipy.sparse.eye(n_indices, dtype=solution_mat.dtype, format="csc")
        print (solution_mat.shape, eye.shape, file=sys.stderr)
        A_eq = scipy.sparse.hstack([solution_mat, -solution_mat, eye, -eye], format="csc")
        b_eq = solution_vals

        print (c.shape, A_eq.shape, b_eq.shape, file=sys.stderr)

        if self.integral:
            """
            result = scipy.optimize.linprog(
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                method='highs',
                #options=solver_options,
            )

            solution = result.x
            print ('Solved first problem', file=sys.stderr)
            tmp_params = solution[:n_params] - solution[n_params:2*n_params]
            print (np.histogram(np.mod(tmp_params, 1)), file=sys.stderr)
            """

            integrality = np.concatenate([np.ones(2 * n_params), np.zeros(2 * n_indices)])
            lower_bounds = np.concatenate([np.zeros(2 * n_params), np.zeros(2 * n_indices)])
            upper_bounds = np.concatenate([np.full(2 * n_params, np.inf), np.full(2 * n_indices, np.inf)])
            #lower_bounds = np.concatenate([np.floor(solution[:2*n_params]), np.zeros(2 * n_indices)])
            #upper_bounds = np.concatenate([np.ceil(solution[:2*n_params]), np.full(2 * n_indices, np.inf)])
            print ('bounds', lower_bounds, upper_bounds, file=sys.stderr)

            result = scipy.optimize.milp(
                c=c,
                integrality=integrality,
                constraints=(A_eq, b_eq, b_eq),
                bounds=scipy.optimize.Bounds(lower_bounds, upper_bounds),
                options=dict(disp=True),
            )
            print ('solved integer problem', file=sys.stderr)

        else:
            result = scipy.optimize.linprog(
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                method='highs',
                #options=solver_options,
            )

        solution = result.x
        solution = solution[:n_params] - solution[n_params:2*n_params]

        print (solution, file=sys.stderr)
        print (np.histogram(np.mod(solution, 1)), file=sys.stderr)

        return solution

class PULPSolver(Solver):
    """ Solver that uses the pulp library to solve a integer programming problem
    representing the set of constraints.
    """
    def __init__(self, integral=True, threads=None):
        self.integral = integral
        self.threads = threads

    def solve(self, constraints, initial_poses):
        import pulp

        prob = pulp.LpProblem("Constraint_set_problem", pulp.LpMinimize)

        xposes, yposes = {}, {}
        anchored_index = None
        for i in initial_poses.keys():
            if anchored_index is None:
                anchored_index = i
                continue
            if self.integral:
                xposes[i] = pulp.LpVariable('xpos{}'.format(i), None, None, pulp.LpInteger)
                yposes[i] = pulp.LpVariable('ypos{}'.format(i), None, None, pulp.LpInteger)
            else:
                xposes[i] = pulp.LpVariable('xpos{}'.format(i), None, None)
                yposes[i] = pulp.LpVariable('ypos{}'.format(i), None, None)

        error_terms = []
        for const in constraints.values():
            for i in range(4):
                error_terms.append(pulp.LpVariable('error{}_{}_{}'.format(const.index1, const.index2, i), 0, None))

        prob += pulp.lpSum(error_terms), 'total_alignment_error'

        xposes[anchored_index] = 0
        yposes[anchored_index] = 0
        print ('set positions to zero for anchored index', file=sys.stderr)

        for i, const in enumerate(constraints.values()):
            score = self.score_func(const)
            error1, error2, error3, error4 = error_terms[i*4:i*4+4]
            prob += (score * xposes[const.index2] - score * xposes[const.index1] == score * const.dx + error1 - error2,
                    "constraint_x_{}_{}".format(const.index1, const.index2))
            prob += (score * yposes[const.index2] - score * yposes[const.index1] == score * const.dy + error3 - error4,
                    "constraint_y_{}_{}".format(const.index1, const.index2))

        #prob += next(iter(xposes)) == 0, 'anchor_x'
        #prob += next(iter(yposes)) == 0, 'anchor_y'
        #print ('added anchors to zero', file=sys.stderr)

        prob.writeLP("constraints.lp")
        if self.threads:
            prob.solve(pulp.COIN_CMD(msg=True, threads=self.threads))
        else:
            prob.solve()

        print (pulp.LpStatus[prob.status], file=sys.stderr)

        for v in prob.variables():
            print(v.name, "=", v.varValue, file=sys.stderr)

        for i in initial_poses.keys():
            if i == anchored_index: continue
            xposes[i] = round(xposes[i].varValue)
            yposes[i] = round(yposes[i].varValue)

        poses = {i: (xposes[i], yposes[i]) for i in initial_poses.keys()}
        return poses

    def score_func(self, constraint):
        return max(0, constraint.score) * max(0, constraint.overlap_ratio)

class HuberSolver(LinearSolver):
    """ Solver that performs quantile regression instead of ordinary least squares
    to solve the system of equations described by constraints. Equivalent to minimizing
    the mean absolute error, this is much more outlier resistant that minimizing MSE
    and should provide better results when there are erroneous constraints present.
    This is identical to LinearSolver except that the linear model used is QuantileRegressor.
    Any parameters passed to the constructor are forwared to the constructor of
    sklearn.linear_model.QuantileRegressor. By default the L1 regularization constant
    alpha=0 and fit_intercept=False, unless specified in the constructor.
    """
    def __init__(self, **kwargs):
        params = dict(alpha=0, fit_intercept=False, epsilon=1)
        params.update(kwargs)
        super().__init__(model=sklearn.linear_model.SGDRegressor(loss='huber', **params))

class OptimalSolver(LinearSolver):
    """ Solver that solves the system of equations generated by LinearSolver by
    minimizing an arbitrary loss function.
    This solver uses the provided linear model, default LinearRegression,
    to get initial positions then uses scipy.optimize.minimize to find the final
    solution that minimizes the loss function.
    By default the loss function used is mean absolute error, however if you are
    looking to minimize on MAE you should use MAESolver as it is equivalent
    and more efficient. Only use this model when there is not a dedicated
    sklearn.linear_model or similar regressor that can be utilized with LinearSolver
    for your loss function
    """

    def __init__(self, model=None, loss_func=None):
        super().__init__(model)
        if loss_func is None:
            loss_func = self.loss
        self.loss_func = loss_func

    def loss(self, values, solution_mat, solution_vals):
        error = np.matmul(solution_mat, values.T) - solution_vals
        #sigmoid_error = 1 / (1 + np.exp(-np.abs(error)))
        return np.sum(np.abs(error))

    def solve_matrix(self, solution_mat, solution_vals, initial_values):
        values = super().solve_matrix(solution_mat, solution_vals, initial_values)
        values = initial_values
        values = values.reshape(-1)

        result = scipy.optimize.minimize(self.loss_func, values, args=(solution_mat, solution_vals))#, options=dict(maxiter=10))
        values = result.x

        return values


class OutlierSolver(Solver):
    def __init__(self, solver=None, outlier_threshold=1.5):
        self.solver = solver or LinearSolver()
        self.outlier_threshold = outlier_threshold

    def solve(self, constraints, initial_poses):
        fully_solved = False

        while not fully_solved:
            poses = self.solver.solve(constraints, initial_poses)

            diffs = []
            errors = []
            for (id1, id2), constraint in constraints.items():
                new_offset = poses[id2] - poses[id1]
                diffs.append((new_offset[0] - constraint.dx, new_offset[1] - constraint.dy))
                errors.append(constraint.error)
            diffs = np.array(diffs)
            diffs = np.linalg.norm(diffs, axis=1)
            errors = np.array(errors)
            min_error = errors.min()

            print ("Solved", len(constraints), "constraints, with error: min {} max".format(
                    np.percentile(diffs, (0,1,5,50,95,99,100)).astype(int)), file=sys.stderr)

            max_diffs = {}
            for pair, diff, error in zip(constraints.keys(), diffs, errors):
                if error == min_error:
                    max_diffs[pair[0]] = max(max_diffs.get(pair[0], 0), diff)
                    max_diffs[pair[1]] = max(max_diffs.get(pair[1], 0), diff)

            fully_solved = True

            for pair, diff, error in zip(list(constraints.keys()), diffs, errors):
                if (diff > self.outlier_threshold and error == min_error
                        and max_diffs[pair[0]] == diff and max_diffs[pair[1]] == diff
                        and not constraints[pair].modeled):
                    fully_solved = False
                    del constraints[pair]

            print ('now', len(constraints), 'constraints', file=sys.stderr)
            #break

        #return poses
        return poses, constraints



class NeighborOutlierSolver(Solver):
    def __init__(self, solver=None, testing_radius=3):
        self.solver = solver or LinearSolver()
        self.outlier_threshold = 5
        self.testing_radius = testing_radius

    def solve(self, constraints, initial_poses):
        #new_constraints = self.get_touching(constraints, (264, 265), self.testing_radius)
        #poses = self.solver.solve(new_constraints)
        #image_indices = sorted(list(set(pair[0] for pair in constraints) | set(pair[1] for pair in constraints)))
        #for i in image_indices:
            #if i not in poses:
                #poses[i] = np.array([0,0])
        #return poses
        constraints = constraints.copy()

        while True:
            poses = self.solver.solve(constraints, initial_poses)
            #for pos in poses.values():
                #pos += 1500

            diffs = []
            for (id1, id2), constraint in constraints.items():
                new_offset = poses[id2] - poses[id1]
                diffs.append((new_offset[0] - constraint.dx, new_offset[1] - constraint.dy))

            diffs = np.abs(np.array(diffs))

            print ("Solved", len(constraints), "constraints, with error: min {} max".format(
                    np.percentile(diffs, (0,1,5,50,95,99,100)).astype(int)))

            if diffs.max() < self.outlier_threshold:
                return poses, constraints

            removal_scores = {}
            for index, (pair, constraint) in enumerate(constraints.items()):
                offset = diffs[index]
                if constraint.modeled or np.linalg.norm(offset) < self.outlier_threshold:
                    continue

                new_constraints = self.get_touching(constraints, pair, self.testing_radius)
                new_poses = self.solver.solve(new_constraints, initial_poses)
                del new_constraints[pair]

                before_diffs = []
                for (id1, id2), const in new_constraints.items():
                    new_offset = new_poses[id2] - new_poses[id1]
                    before_diffs.append((new_offset[0] - const.dx, new_offset[1] - const.dy))
                before_diffs = np.array(before_diffs)

                new_poses = self.solver.solve(new_constraints, initial_poses)

                after_diffs = []
                for (id1, id2), const in new_constraints.items():
                    new_offset = new_poses[id2] - new_poses[id1]
                    after_diffs.append((new_offset[0] - const.dx, new_offset[1] - const.dy))
                after_diffs = np.array(after_diffs)
                print (pair, np.sum(np.abs(before_diffs) - np.abs(after_diffs)), constraint)

                removal_scores[pair] = np.sum(np.abs(before_diffs) - np.abs(after_diffs))
                #print (before_diffs.sum(), after_diffs.sum())
                #print (np.percentile(np.abs(before_diffs), [0,1,5,50,95,99,100]), np.percentile(np.abs(after_diffs), [0,1,5,50,95,99,100]))

            if len(removal_scores) == 0:
                return poses, constraints

            max_pair = max(removal_scores.keys(), key=lambda pair: removal_scores[pair])
            del constraints[max_pair]


    def get_touching(self, constraints, start_pair, max_dist):
        """ returns a new constraints dict with only constraints within max_dist to the start pair
        basically bfs
        """
        pairs = {start_pair}
        pairs_left = set(constraints.keys())
        pairs_left.remove(start_pair)
        frontier = {start_pair[0], start_pair[1]}

        while len(frontier) > 0 and max_dist > 0:
            new_frontier = set()
            for pair in pairs_left:
                if pair[0] in frontier:
                    new_frontier.add(pair[1])
                    pairs.add(pair)
                if pair[1] in frontier:
                    new_frontier.add(pair[0])
                    pairs.add(pair)
            frontier = new_frontier
            pairs_left = pairs_left - pairs
            max_dist -= 1

        return {pair: constraints[pair] for pair in pairs}


class SelectionSolver(Solver):
    """ Solver that finds the maximal set of constraints that all align with each other
    """

    def __init__(self):
        pass

    def solve(self, constraints, initial_poses):
        pass

    def find_cycles(self, nodes, edges):
        cycles = set()
        for node in nodes:
            for cycle in find_cycles_node(node, edges):
                max_elem = cycle.index(max(cycle))
                cycle = cycle[max_elem:] + cycle[:max_elem]
                if cycle[1] < cycle[-1]:
                    cycle = cycle[:1] + cycle[1:][::-1]
                cycles.add(cycle)

    def find_cycles_node(self, node, edges):
        pass

class SpanningTreeSolver(Solver):
    """ Solver that finds the maximum spanning tree, and uses it to solve for
    global positions of all images
    """

    def __init__(self, score_func=None):
        self.score_func = score_func
        if score_func is None:
            self.score_func = lambda const: const.score

    def solve(self, constraints, initial_poses):
        """ Constructs a maximum spanning tree with Kruskals algorithm
        """
        parents = {i: i for i in initial_poses.keys()}
        sizes = {i: 1 for i in initial_poses.keys()}

        def find(i):
            while parents[i] != i:
                i, parents[i] = parents[i], parents[parents[i]]
            return i

        def union(i, j):
            i, j = find(i), find(j)
            if i == j:
                return

            if sizes[i] < sizes[j]:
                i, j = j, i

            parents[j] = i
            sizes[i] += sizes[j]

        spanning_tree = {}

        for constraint in sorted(constraints.values(), key=self.score_func, reverse=True):
            i, j = constraint.pair
            i, j = find(i), find(j)
            if i != j:
                union(i, j)
                spanning_tree.setdefault(constraint.pair[0], {})[constraint.pair[1]] = constraint
                spanning_tree.setdefault(constraint.pair[1], {})[constraint.pair[0]] = constraint

        final_poses = {}

        consts = ConstraintSet()
        def propagate(i, last_i, pos):
            final_poses[i] = pos
            for child, constraint in spanning_tree[i].items():
                if child != last_i:
                    if constraint.index1 == i:
                        newpos = pos[0] + round(constraint.dx), pos[1] + round(constraint.dy)
                    else:
                        newpos = pos[0] - round(constraint.dx), pos[1] - round(constraint.dy)
                    #print (i, child, constraint, pos, pos[0] + constraint.dx, pos[1] + constraint.dy, file=sys.stderr)
                    consts.add(constraint)
                    print (constraint, file=sys.stderr)
                    print (constraint.composite, file=sys.stderr)
                    propagate(child, i, newpos)

        propagate(next(iter(initial_poses.keys())), -1, (0,0))

        print (consts, file=sys.stderr)
        print (len(consts), file=sys.stderr)
        print (consts.composite, file=sys.stderr)
        consts.composite.plot_scores('spantree_scores.png', consts)

        assert set(final_poses.keys()) == set(initial_poses.keys())

        return final_poses

