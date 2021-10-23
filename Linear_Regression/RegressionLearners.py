import random


def run_weight_vec_cost_function(weights, dataset, label_col):
    cost = [0] * len(weights)
    for row in dataset:
        for i, w in enumerate(weights):
            cost[i] += ((float(row[label_col]) - weights[i]*float(row[i]))**2)/2
    return min(cost)


class RegressionLearners:
    dataset = []
    attributes = {}
    label_col = 0
    attribute_names = []
    '''dataset is a set of numeric data
        The attributes are simply a set of tuples of ints and strings containing the the column 
        index and column name for each attribute for the input dataset'''
    def __init__(self, dataset, attributes):
        self.dataset = dataset.copy()
        self.attributes = attributes

    def batch_gradient_descent(self, learning_rate, max_num_steps, min_error_threshold):
        # set initial weights here
        label_col = len(self.attributes)
        weights = [0] * len(self.attributes)
        total_error = None
        for j in range(max_num_steps):
            print(run_weight_vec_cost_function(weights, self.dataset, label_col))
            if total_error is not None and total_error < min_error_threshold:
                return weights
            # construct the gradient vector
            grad_vec = []
            # add the gradient of each sample to the gradient vector
            for i in range(len(self.attributes)):
                grad = 0
                for idx, row in enumerate(self.dataset):
                    total_error = float(row[label_col]) - weights[i]*float(row[i])
                    grad += (total_error * float(row[i]))
                grad = grad * -1.0
                grad_vec.append(grad)
            for i in range(len(self.attributes)):
                temp = weights[i]
                weights[i] = weights[i] - (learning_rate * grad_vec[i])
                # print((temp - weights[i])/max(weights))
        return weights

    def stochastic_gradient_descent(self, learning_rate, max_num_steps, min_error_threshold):
        # set initial weights here
        label_col = len(self.attributes)
        weights = [0] * len(self.attributes)
        total_error = None
        for j in range(max_num_steps):
            print(run_weight_vec_cost_function(weights, self.dataset, label_col))
            if total_error is not None and total_error < min_error_threshold:
                print("returning. Total error was: ", total_error)
                return weights
            # update the weights based of a randomly selected row
            total_error = 0
            row = random.choice(self.dataset)
            for i in range(len(self.attributes)):
                total_error += abs(float(row[label_col]) - weights[i]*float(row[i]))
                weights[i] = weights[i] + learning_rate * ((float(row[label_col]) - weights[i]*float(row[i])) * float(row[i]))
        return weights
