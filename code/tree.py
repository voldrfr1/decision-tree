import numpy as np

class Node:
    """
    Node class for Decision Tree representation.

    Parameters:
    - left: Left child node.
    - right: Right child node.
    - feature_idx: Index of the feature used for the split.
    - value: Value at which the split occurs.
    - class_id: Class label for leaf nodes.
    """
    def __init__(self, left=None, right=None, feature_idx=None, value=None, class_id=None):
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.value = value
        self.class_id = class_id


class DecisionTree:
    """
    Decision Tree classifier.

    Parameters:
    - max_depth: Maximum depth of the tree.
    - num_of_samples: Minimum number of samples required to perform a split.

    Methods:
    - fit: Build the decision tree model.
    - predict: Make predictions on new data.
    - evaluate: Evaluate the accuracy of the model on a test set.
    """

    def __init__(self, max_depth=5, num_of_samples=2):
        self.max_depth = max_depth
        self.num_of_samples = num_of_samples
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        dataset = np.concatenate((X, y.reshape((-1, 1))), axis=1)
        self.root = self.__build_tree(dataset)

    def predict(self, X):
        """
        Make predictions using the decision tree.

        Parameters:
        - X: Input features for prediction.

        Returns:
        - predictions: Predicted labels.
        """
        #traverses the tree recursively to find a leaf value
        predictions = [self.__make_prediction(self.root, x) for x in X]
        return predictions

    def evaluate(self, X, y):
        """
        Evaluate the accuracy of the decision tree on a test set.

        Parameters:
        - X: Input features of the test set.
        - y: True labels of the test set.

        Returns:
        - accuracy: Accuracy of the model on the test set.
        """
        predictions = self.predict(X)
        return np.sum(predictions == y) / len(y)

    def __make_prediction(self, node, x):
        """
        Make a prediction for a single input using the decision tree.

        Parameters:
        - node: Current node in the tree.
        - x: Input feature.

        Returns:
        - prediction: Predicted label.
        """
        if node.class_id is not None:
            return node.class_id
        feature = x[node.feature_idx]
        if self.__compare_one(feature, node.value):
            return self.__make_prediction(node.left, x)
        return self.__make_prediction(node.right, x)

    def __compare(self, feature, value):
        """
        Compare feature values with a given value.

        Parameters:
        - feature: Feature values.
        - value: Value for comparison.

        Returns:
        - left: Boolean array indicating if feature is less than or equal to the value.
        - right: Boolean array indicating if feature is greater than the value.
        """
        #for categorical data, the only change in the code required is in this comparison
        left = feature <= value
        right = ~left
        return left, right

    def __compare_one(self, feature, value):
        """
        Compare a single feature value with a given value. A bit
        cumbersome due to our architecture.

        Parameters:
        - feature: Feature value.
        - value: Value for comparison.

        Returns:
        - result: Boolean indicating if feature is equal to the value.
        """
        return self.__compare(np.array([feature]), value)[0][0]

    def __entropy(self, column):
        """
        Compute the entropy of a column.

        Parameters:
        - column: Column of labels.

        Returns:
        - entropy: Entropy of the column.
        """
        #we want unique values. Not optimal way of computing entropy of a column!!!
        values = set(column)
        entropy = 0
        for val in values:
            #extract valuus with the given  label and get their relative frequency
            p_val = len(column[column == val]) / len(column)
            entropy += -p_val * np.log2(p_val)
        return entropy

    def __information_gain(self, parent, left, right):
        """
        Compute information gain for a split, i.e. H(x)-H(x|a)

        Parameters:
        - parent: Parent column.
        - left: Left child column.
        - right: Right child column.

        Returns:
        - information_gain: Information gain for the split.
        """
        parent_entropy = self.__entropy(parent)# entropy of the full column
         #compute entropy of the column when its splitted into two sets
        left_entr, right_entr = self.__entropy(left), self.__entropy(right)
        weights = len(left) / len(parent), len(right) / len(parent)
        return parent_entropy - weights[0] * left_entr - weights[1] * right_entr

    def __split_data(self, X, feature_idx, value):
        """
        Split the dataset based on a feature and a value.
        Uses compare function so that we can easily change it if needed.

        Parameters:
        - X: Input features.
        - feature_idx: Index of the feature to split on.
        - value: Value for the split.

        Returns:
        - left_X: Left subset of features.
        - right_X: Right subset of features.
        """
        left_mask, right_mask = self.__compare(X[:, feature_idx], value)
        return X[left_mask, :], X[right_mask, :]

    def __best_split(self, X):
        """
        Find the best split for the given dataset.

        Parameters:
        - X: Input features.

        Returns:
        - result: Dictionary containing information about the best split.
        """
        result = {'IG': 0, 'feature_idx': 0, 'value': 0, 'left_X': None, 'right_X': None}

        for feature_idx in range(len(X[0, :-1])): #dont count labels
            features = X[:, feature_idx] #extract the column of features
            unique_values = set(features) #get possible values, use them for splitting
            for value in unique_values:
                #split the training dataset using this feature and value
                left_X, right_X = self.__split_data(X, feature_idx, value)
                 #check if either is empty (if it is, then we have a leaf)
                if len(left_X) and len(right_X):
                    #we compute this on labels, because we want to know
                    #how well we managed to split the sets into sets of different classes
                    IG = self.__information_gain(X[:, -1], left_X[:, -1], right_X[:, -1])
                    if result['IG'] < IG:
                        result['IG'] = IG
                        result['feature_idx'] = feature_idx
                        result['value'] = value
                        result['left_X'] = left_X
                        result['right_X'] = right_X
        return result

    def __build_tree(self, X, current_depth=1):
        """
        Recursively build the decision tree by creating the best splits.

        Parameters:
        - X: Input features.
        - current_depth: Current depth of the tree.

        Returns:
        - decision_node: Root node of the decision tree.
        """
        #if we have no more data to split, max depth exceeded or min samples 
        #condition is not met, we stop
        if not len(X):
            return None
        if self.max_depth < current_depth or len(X) < self.num_of_samples:
            return self.__calculate_leaf(X[:, -1])

        best_split = self.__best_split(X)
        #if zero, we didnt achieve anything and there is no point in continuing 
        #to build the tree, we can just stop
        #we calculate the leaf value here, because we can have heterogenous
        #set with all attributes being the same
        if best_split['IG'] == 0:
            return self.__calculate_leaf(X[:, -1])
        #otherwise set the children by recursively calling it on split dataset
        left = self.__build_tree(best_split['left_X'], current_depth + 1)
        right = self.__build_tree(best_split['right_X'], current_depth + 1)
        return Node(left, right, best_split['feature_idx'], best_split['value'])

    def __calculate_leaf(self, y):
        """
        Calculate the leaf node for the decision tree by labeling it 
        with the most occuring class id. An improvement would be to also include the
        confidence as a Node attribute.

        Parameters:
        - y: Target labels.

        Returns:
        - node: Leaf node.
        """
        y = list(y)
        return Node(class_id=max(y, key=y.count))
