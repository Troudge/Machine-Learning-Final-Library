from SVM import SVMLearner
import numpy as np

dataset_banknote = []
with open('../Data/bank-note/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        float_terms = [float(t) for t in terms]
        dataset_banknote.append(float_terms)

dataset_banknote_test = []
with open('../Data/bank-note/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        float_terms = [float(t) for t in terms]
        dataset_banknote_test.append(float_terms)

learner = SVMLearner.SVMLearner(dataset_banknote)
final_weights = learner.primal_svm(100, 0.01, 0.05, 100 / 873)
# training error
print('train and test for C = 100/873')
print(SVMLearner.run_learned_weights(final_weights, dataset_banknote))
# test error
print(SVMLearner.run_learned_weights(final_weights, dataset_banknote_test))

final_weights = learner.primal_svm(100, 0.01, 0.05, 500 / 873)
# training error
print('train and test for C = 500/873')
print(SVMLearner.run_learned_weights(final_weights, dataset_banknote))
# test error
print(SVMLearner.run_learned_weights(final_weights, dataset_banknote_test))

final_weights = learner.primal_svm(100, 0.01, 0.05, 700 / 873)
# training error
print('train and test for C = 700/873')
print(SVMLearner.run_learned_weights(final_weights, dataset_banknote))
# test error
print(SVMLearner.run_learned_weights(final_weights, dataset_banknote_test))


def basic_kernel(x: np.array, xt: np.array):
    return np.dot(x, xt)


learner.dual_svm(100/873, basic_kernel)
