import numpy as np
from sklearn.base import BaseEstimator

def compute_value(y, criteria):
    if criteria == 'entropy':
        return entropy(y)
    elif criteria == 'variance':
        #return variance(np.argmax(y))
        return variance(y)
    elif criteria == 'mad_median':
        #return mad_median(np.argmax(y))
        return mad_median(y)
    else:
        return gini(y)
    
    
def entropy(y):
    EPS = 0.0005
    if y.size == 0:
        return 0    
    p = y.mean(axis=0)
    return - p @ np.log2(p + EPS)

def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    """
    if y.size == 0:
        return 0
    p = y.mean(axis=0)
    return 1 - (p**2).sum()


def variance(y):
    if not len(y):
        return 0
    return y.var()

def mad_median(y):
    if not len(y):
        return 0
    return np.abs(y - np.median(y)).mean()

def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot

def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]

class Node:

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            self.classification = True
        else:
            self.classification = False
        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        idx = X_subset[:, feature_index] < threshold
        X_left = X_subset[idx]
        X_right = X_subset[~idx]
        y_left = y_subset[idx]
        y_right = y_subset[~idx]
        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        idx = X_subset[:, feature_index] < threshold
        y_left = y_subset[idx]
        y_right = y_subset[~idx]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        self.criterion, self.classification = self.all_criterions[self.criterion_name]    
        feature_index_best, threshold_best = 0, 0
        value_best = np.Inf
        for feature_index in range(X_subset.shape[1]):
            #Thresholds = np.sort(np.unique(X_subset[:, feature_index]))[1:]
            Thresholds = np.unique(X_subset[:, feature_index])
            Thresholds = np.delete(Thresholds, np.where(Thresholds==Thresholds.min()))
            for threshold in Thresholds:
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                value_change = len(y_left) * compute_value(y_left, self.criterion_name) + len(y_right) * compute_value(y_right, self.criterion_name)
                if value_change < value_best:
                    feature_index_best, threshold_best, value_best = feature_index, threshold, value_change
        return feature_index_best, threshold_best

    def make_tree(self, X_subset, y_subset, d=0):
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        new_node.n_samples = y_subset.shape[0]
        if self.classification:
            new_node.proba = np.mean(y_subset, axis=0)
        elif self.criterion == 'variance':
            new_node.proba = np.mean(y_subset)
        else:
            new_node.proba = np.median(y_subset)
        if d < self.max_depth and len(y_subset) > self.min_samples_split:
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            new_node.left_child = self.make_tree(X_left, y_left, d+1)
            new_node.right_child = self.make_tree(X_right, y_right, d+1)
        return new_node

    def fit(self, X, y):
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
        self.root = self.make_tree(X, y)

    def predict(self, X):
        # YOUR CODE HERE
        vector = [0] * X.shape[0]
        for i in range(X.shape[0]):
            data = X[i]
            node = self.root
            while node is not None:
                vector[i] = node.proba
                if data[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
                if node is None:
                    break
        if self.classification:
            y_predicted = one_hot_decode(np.array(vector))
        else:
            y_predicted = np.array(vector).reshape(len(vector), 1)
        return y_predicted

    def predict_proba(self, X):
        assert self.classification, 'Available only for classification problem'
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))
        vector = [0] * X.shape[0]
        for i in range(X.shape[0]):
            data = X[i]
            node = self.root
            while node is not None:
                vector[i] = node.proba
                if node.feature_index is not None and data[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
                if node is None:
                    break
        y_predicted_probs = np.array(vector)
        return y_predicted_probs

if __name__ == '__main__':
    from time import time
    t1 = time()
    from sklearn.datasets import make_classification, make_regression, load_digits, load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    digits_data = load_digits().data
    digits_target = load_digits().target[:, None]
    idx_max = -1
    digits_data = digits_data[:idx_max]
    digits_target = digits_target[:idx_max]
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE)
    tree = DecisionTree(criterion_name='entropy')
    tree.fit(X_train, y_train)
    print(tree.depth)
    y_pred = tree.predict(X_test)
    b = tree.predict_proba(X_test)
    from matplotlib import pyplot as plt
    plt.close('all')
    z1 = y_test.flatten() + 0.25 * np.random.randn(len(y_pred))
    z2 = y_pred.flatten() + 0.25 * np.random.randn(len(y_pred))
    plt.scatter(z1, z2, s=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    print((y_pred==y_test).mean())
    '''
    regr_data = load_boston().data
    regr_target = load_boston().target[:, None] # to make the targets consistent with our model interfaces
    RANDOM_STATE = 42
    RX_train, RX_test, Ry_train, Ry_test = train_test_split(regr_data, regr_target, test_size=0.2, random_state=RANDOM_STATE)
    regressor = DecisionTree(max_depth=10, criterion_name='variance')
    regressor.fit(RX_train, Ry_train)
    predictions_var = regressor.predict(RX_test)
    mse_var = mean_squared_error(Ry_test, predictions_var)
    print(mse_var)
    '''
    t2 = time()
    print(t2-t1)
