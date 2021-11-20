import numpy as np
import scipy.optimize


def run_learned_weights(weights, dataset):
    correct = 0
    for row in dataset:
        output = 0
        for idx, value in enumerate(row[:-1]):
            output += value * weights[idx]
        # output += weights[-1]
        result = 1 if output > 0 else 0
        if result == int(row[-1]):
            correct += 1
    return (len(dataset) - correct) / len(dataset)


class SVMLearner:
    dataset = []
    label_col = 0

    def __init__(self, dataset):
        self.dataset = np.array(dataset.copy())
        self.dataset[:, -1][self.dataset[:, -1] == 0] = -1

    def primal_svm(self, T, initial_gamma, alpha, C):
        # extra weight at the end of weights is the bias
        weights = np.zeros(self.dataset.shape[1] - 1)
        initial_weights = np.zeros(self.dataset.shape[1] - 1)
        for i in range(T):
            gamma = initial_gamma / (1 + (initial_gamma / alpha) * i)
            np.random.shuffle(self.dataset)
            for row in self.dataset:
                y = row[-1]
                # b = weights[-1]
                n = len(self.dataset)
                # hinge loss sub gradient calculations
                hinge = y * np.matmul(weights.transpose(), row[:-1])
                if hinge <= 1:
                    weights = weights - (gamma * initial_weights) + gamma * C * n * y * row[:-1]
                else:
                    initial_weights = (1 - gamma) * initial_weights
        return weights

    def dual_svm(self, initial_gamma, C, kernel):
        weights = np.zeros(self.dataset.shape[1] - 1)

        def objective_func(alphas: np.array, y: np.array, x: np.array):
            reshaped_alphas = alphas.reshape(len(alphas),1)
            a_mult = np.matmul(reshaped_alphas, reshaped_alphas.transpose())
            reshaped_y = y.reshape(len(y), 1)
            y_mult = np.matmul(reshaped_y, reshaped_y.transpose())
            ay = np.matmul(a_mult, y_mult)
            result = np.matmul(ay, kernel(x, x.transpose())).sum()
            result = (0.5 * result) - alphas.sum()
            return result

        def constraint(alphas: np.array, y: np.array):
            if np.matmul(alphas.reshape(len(alphas), 1), y.reshape(len(y), 1).transpose()).sum() == 0:
                return 0
            else:
                return 1

        x = np.array([row[:-1] for row in self.dataset])
        y = np.array([row[-1] for row in self.dataset])
        initial_guess = np.full(self.dataset.shape[0], C/2)
        bound = [(0, C)] * self.dataset.shape[0]
        result = scipy.optimize.minimize(fun=objective_func, x0=initial_guess, args=(y, x), method='SLSQP',
                                         bounds=bound, constraints={'type': 'eq', 'fun': constraint, 'args': (y,)})
        if result.success:
            final_alphas = result.x
            print(final_alphas)
        else:
            print('Optimizer did not return success')
            print(result)
