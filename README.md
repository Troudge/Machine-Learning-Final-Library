# Machine-Learning-Final-Library
This is a machine learning library developed by Nathan Lockwood forCS5350/6350 in University of Utah


Some notes on the classes that have been created:
all classes have a basic initializer that must be declared first. This stores the dataset and attribute information.
An example of this is shown below:

`` bank_learner = EnsembleLearners.EnsembleLearner(dataset_bank, atr2, 16)``

The input attributes  marked "atr2" is a set of tuples inn the form (column index, attribute name), which is an int and a string. 
After declaring a new instance of the learner, you can call any of the machine learning algorithms associcated with it.

The Regression Learners class contains the gradient descent algorithms

The Ensemble Learners class contains the Boosting and bagging methods

The Id3 class contains Id3 and the stumps method (the generate id3 stump is used for the some of the ensemble learners).

Additionally, there are some other methods not part of the classes in each module/file. the most useful of which start with run, and take in a dataset and some other parameters to run the learned algorithms.

Lastly any of the algorithms that use trees utilize my node class. you can call print on a node to print out the tree structure to the console. 
Additionally, you can call the to_graph method to convert a node tree to a network x graph, and then you can plot it as you see fit.