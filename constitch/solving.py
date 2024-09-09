import os
import time
import numpy as np
import sklearn.linear_model
import skimage.io
import scipy.optimize

from .constraints import Constraint

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


class LinearSolver:
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

    def solve(self, constraints, initial_poses):
        #image_indices = sorted(list(set(pair[0] for pair in constraints) | set(pair[1] for pair in constraints)))
        solution_mat, solution_vals, initial_values = self.make_constraint_matrix(constraints, initial_poses)

        solution = self.solve_matrix(solution_mat, solution_vals, initial_values)

        poses = np.round(solution.reshape(-1,2)).astype(int)
        poses -= poses.min(axis=0).reshape(1,2)

        return dict(zip(list(initial_poses.keys()), poses))

    def solve_matrix(self, solution_mat, solution_vals, initial_values):
        #solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
        model = self.model.fit(solution_mat, solution_vals)
        solution = model.coef_
        return solution

    def score_func(self, constraint):
        return max(0, constraint.score) + 0.1

class OptimalSolver(LinearSolver):
    def __init__(self, **kwargs):
        super().__init__()

    def solve_matrix(self, solution_mat, solution_vals, initial_values):
        values = initial_values
        initial_poses = values.reshape(-1, 2)

        def simple_loss(values):
            error = np.matmul(solution_mat, values.T) - solution_vals
            error = 1 / (1 + np.exp(-error / 25)) - 0.5
            return np.sum(np.abs(error))
            #return np.sum(error * error)

        result = scipy.optimize.minimize(simple_loss, values)#, options=dict(maxiter=10))
        values = result.x
        np.save('tmp_values.npy', values)

        """
        import matplotlib.pyplot as plt
        #fig, axes = plt.subplots(ncol=1, nrow=20, figsize=(6, 4*20))
        fig, axis = plt.subplots(figsize=(15, 15))
        all_poses = [initial_poses]
        for i in range(1):
            print ('optimizing')
            result = scipy.optimize.minimize(simple_loss, values)#, options=dict(maxiter=10))
            values = result.x
            poses = values.reshape(-1, 2)
            all_poses.append(poses)
            #axis.scatter(poses[:,0], poses[:,1], s=1)
        all_poses = np.array(all_poses)

        axis.scatter(all_poses[0,:,0], all_poses[0,:,1])
        for i in range(all_poses.shape[1]):
            axis.plot(all_poses[:,i,0], all_poses[:,i,1])
        fig.savefig('plots/huber_optim.png')
        """

        return values

class OutlierSolver:
    def __init__(self, solver=None, testing_radius=3):
        self.solver = solver or LinearSolver()
        self.outlier_threshold = 5
        self.testing_radius = testing_radius

    def solve(self, constraints):
        #new_constraints = self.get_touching(constraints, (264, 265), self.testing_radius)
        #poses = self.solver.solve(new_constraints)
        #image_indices = sorted(list(set(pair[0] for pair in constraints) | set(pair[1] for pair in constraints)))
        #for i in image_indices:
            #if i not in poses:
                #poses[i] = np.array([0,0])
        #return poses
        constraints = constraints.copy()

        while True:
            poses = self.solver.solve(constraints)
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
                return poses

            removal_scores = {}
            for index, (pair, constraint) in enumerate(constraints.items()):
                offset = diffs[index]
                if constraint.modeled or np.linalg.norm(offset) < self.outlier_threshold:
                    continue

                new_constraints = self.get_touching(constraints, pair, self.testing_radius)
                new_poses = self.solver.solve(new_constraints)
                del new_constraints[pair]

                before_diffs = []
                for (id1, id2), const in new_constraints.items():
                    new_offset = new_poses[id2] - new_poses[id1]
                    before_diffs.append((new_offset[0] - const.dx, new_offset[1] - const.dy))
                before_diffs = np.array(before_diffs)

                new_poses = self.solver.solve(new_constraints)

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
                return poses

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
