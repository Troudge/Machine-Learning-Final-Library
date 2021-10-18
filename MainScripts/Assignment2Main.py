from Decision_Trees.Id3 import Id3Tree
from Decision_Trees.tree import Node
import networkx as nx
import matplotlib.pyplot as plt
from Ensemble_Learning import EnsembleLearners

dataset2 = [['S', 'H', 'H', 'W', 0],
            ['S', 'H', 'H', 'S', 0],
            ['O', 'H', 'H', 'W', 1],
            ['R', 'M', 'H', 'W', 1],
            ['R', 'C', 'N', 'W', 1],
            ['R', 'C', 'N', 'S', 0],
            ['O', 'C', 'N', 'S', 1],
            ['S', 'M', 'H', 'W', 0],
            ['S', 'C', 'N', 'W', 1],
            ['R', 'M', 'N', 'W', 1],
            ['S', 'M', 'N', 'S', 1],
            ['O', 'M', 'H', 'S', 1],
            ['O', 'H', 'N', 'W', 1],
            ['R', 'M', 'H', 'S', 0]]

dataset_bank = []
with open('../Data/bank/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_bank.append(terms)

dataset_bank_test = []
with open('../Data/bank/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_bank_test.append(terms)

atr = {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}
atr2 = {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'), (5, 'safety')}
# tests of the adaboost algorithm
learner = EnsembleLearners.EnsembleLearner(dataset2, atr, 4)
forest = learner.adaboost(1)
print(EnsembleLearners.run_learned_forest(forest, ['R', 'C', 'N', 'W', 1], atr))

bank_learner = EnsembleLearners.EnsembleLearner(dataset_bank, atr2, 6)
for i in range(500):
    bank_forest = bank_learner.adaboost(i)
    print(EnsembleLearners.run_forest_on_set(bank_forest, dataset_bank, atr2, 6))

# An example of a decision tree stump
# decision_tree = Id3Tree(dataset2, atr, 4, 'information_gain')
# nx.draw(decision_tree.generate_id3_tree_stump().to_graph(),
#         with_labels=True, arrows=True)
# plt.show()

