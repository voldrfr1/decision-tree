import numpy as np
from tree import DecisionTree
from tqdm import tqdm

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, num_of_samples=2):
        """
        Initialize the Random Forest model.

        Parameters:
        - n_trees: Number of decision trees in the forest.
        - max_depth: Maximum depth of each decision tree.
        - num_of_samples: Number of samples to consider for each split during tree construction.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.num_of_samples = num_of_samples
        self.trees = [DecisionTree(self.max_depth, self.num_of_samples) for _ in range(self.n_trees)]

    def fit(self, X, y):
        """
        Fit the Random Forest model on the training data.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        dataset = np.concatenate((X, y.reshape((-1, 1))), axis=1)

        # Create and fit individual decision trees
        for i in tqdm(range(self.n_trees), desc="Fitting Trees"):
            #bootstrap dataset for each tree
            bootstrapped_data = self.__bootstrap(dataset)
            #fit on the bootstrapped dataset, creating different weak classifiers
            X_boot, y_boot = bootstrapped_data[:, :-1], bootstrapped_data[:, -1]
            self.trees[i].fit(X_boot, y_boot)

    def predict(self, X):
        """
        Make predictions using the Random Forest model.

        Parameters:
        - X: Input features for prediction.

        Returns:
        - majority_pred: Predicted labels based on a majority voting scheme.
        """
        #we dont need to loop over the individual vectors to predict,
        #this is done by DecisionTree.predict method
        #of course this is not ideal approach in terms of performance
        preds = np.array([tree.predict(X) for tree in self.trees]).T
        majority_pred = np.array([self.__majority_pred(pred) for pred in preds])
        return majority_pred

    def __majority_pred(self, pred):
        """
        Determine the majority prediction for a given set of predictions.

        Parameters:
        - pred: List of predictions.

        Returns:
        - majority_prediction: Majority prediction.
        """
        pred_list = list(pred)
        return max(pred_list, key=pred_list.count)

    def __bootstrap(self, dataset):
        """
        Create a bootstrapped dataset by sampling with replacement.

        Parameters:
        - dataset: Original dataset.

        Returns:
        - bootstrapped_data: Bootstrapped dataset.
        """
        rows, _ = dataset.shape
        indices = np.random.choice(rows, rows, replace=True)
        return dataset[indices]
