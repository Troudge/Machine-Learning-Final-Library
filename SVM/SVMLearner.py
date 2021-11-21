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
            # gamma = initial_gamma/(1+i)
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

    def dual_svm(self, C, kernel):
        def objective_func(alphas, y, x):
            a_mult = np.dot(alphas, alphas.transpose())
            y_mult = np.dot(y, y.transpose())
            ay = np.dot(a_mult, y_mult)
            total = np.dot(ay, kernel(x, x.transpose())).sum()
            total = (0.5 * total) - alphas.sum()
            return total

        def constraint(alphas):
            y = np.array([row[-1] for row in self.dataset])
            if np.dot(alphas, y).sum() == 0:
                return 0
            else:
                return 1

        x = np.array([row[:-1] for row in self.dataset])
        y = np.array([row[-1] for row in self.dataset])
        #x0 = np.full(self.dataset.shape[0], C / 2)
        x0 = np.full(self.dataset.shape[0], C/2)
        bound = scipy.optimize.Bounds(0, C)
        result = scipy.optimize.minimize(fun=objective_func, x0=x0, args=(y, x), method='SLSQP',
                                         bounds=bound, constraints={'type': 'eq', 'fun': constraint})
        if result.success:
            final_alphas = result.x
            print(final_alphas)
            weights = np.dot(np.dot(final_alphas, y), kernel(x, x.transpose()))
            b = y - np.dot(weights.transpose(), x)
            return weights, b
        else:
            print('Optimizer did not return success')
            print(result)
