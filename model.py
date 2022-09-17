from pickle import NONE
from sklearn.linear_model import LinearRegression
from importlib import reload  # Not needed in Python 2
import logging
import numpy as np
import pandas as pd


def mse(pred, target):
    return np.mean((pred - target) ** 2)


def var(x, y):
    return np.sum((x - np.mean(x)) ** 2) / (len(x) - 1)


reload(logging)
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s %(levelname)s:    %(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)
logger.setLevel(logging.DEBUG)

from sklearn.linear_model import LinearRegression
from importlib import reload  # Not needed in Python 2
import logging
import numpy as np
import pandas as pd


reload(logging)
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s %(levelname)s:    %(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)
logger.setLevel(logging.DEBUG)


class Node(object):
    def __init__(
        self,
        target,
        numerical_features,
        categorical_features,
        min_data_node=100,
        current_depth=0,
        max_depth=10,
        bin_size=30,
        root_node=None,
        linear_model_at_leaf=False,
        node_direction="root",
        parent_node=None,
        split_metric="mse",
        error_metric="mse",
    ):
        self.target = target
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.min_data_node = min_data_node
        self.bin_size = bin_size
        self.tree_depth = current_depth + 1
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.linear_model_at_leaf = linear_model_at_leaf
        self.train_index = []
        self.node_average = None
        self.node_feature = None
        self.node_split = None
        self.node_direction = node_direction
        self.parent_node = parent_node
        if root_node is None:
            self.root_node = self
        else:
            self.root_node = root_node
        self.split_metric = split_metric
        self.error_metric = error_metric
        self.split_metric_name, self.split_metric_func = self._get_metric(split_metric)
        self.error_metric_name, self.error_metric_func = self._get_metric(error_metric)
        self.left_node = None
        self.right_node = None
        self.name = f"{self.parent_node.name+'>' if self.parent_node else ''}{self.node_direction}"
        self.linear_model = None
        logging.debug(
            f"Building a {self.node_direction} node at the depth of: {self.current_depth}"
        )

    def __repr__(self):
        return self.name

    def _get_dict(self, include_train_index=False):
        return {
            "node_feature": self.node_feature,
            "node_feature_split": self.node_split,
            "current_depth": self.current_depth,
            "subtree_depth": self.tree_depth,
            "train_data_index": self.train_index if include_train_index else [],
            "count_train_data": len(self.train_index)
            if isinstance(self.train_index, list)
            else 0,
            "left_node": self.left_node._get_dict()
            if self.left_node is not None
            else None,
            "right_node": self.right_node._get_dict()
            if self.right_node is not None
            else None,
            "linear_model_intercept": self.linear_model.intercept_
            if self.linear_model is not None
            else None,
            "linear_model_coefs": list(
                zip(self.linear_model.feature_names_in_, self.linear_model.coef_)
            )
            if self.linear_model is not None
            else None,
        }

    def _get_metric(self, metric_name="mse"):
        metrics_list = ["mse", "var"]
        if metric_name == "mse":
            return "Mean Squared Error", mse
        elif metric_name == "var":
            return "Variance", var
        else:
            raise ValueError(
                f"Metric {metric_name} is not available. Metric should be one of part of: {metrics_list}"
            )

    def fit(self, data_subset, validation):
        self.train_index = list(data_subset.index)
        self.node_metric = np.inf
        self.node_feature = None
        self.node_split = None
        self.node_left_prediction = data_subset[self.target].mean()
        self.node_right_prediction = data_subset[self.target].mean()
        self.node_average = data_subset[self.target].mean()

        left_tree_depth = 0
        right_tree_depth = 0
        if self.current_depth < self.max_depth:
            (
                self.node_metric,
                self.node_feature,
                self.node_split,
                self.node_left_prediction,
                self.node_right_prediction,
            ) = self._find_best_feature(data_subset)
        if self.node_feature:
            logging.info(
                f"Split {self.name} on feature: {self.node_feature} in: {self.node_split}"
            )
        else:
            logging.info(f"No more splits possible at node {self.name}")
        logging.info(
            f"Validation {self.error_metric_name}: "
            + str(
                self.error_metric_func(
                    validation[self.target], self.root_node.predict(validation)
                )
            )
        )
        if self.tree_depth <= self.max_depth and self.node_feature is not None:
            if (
                len(data_subset[data_subset[self.node_feature] < self.node_split])
                >= self.min_data_node
            ):
                self.left_node = Node(
                    numerical_features=self.numerical_features,
                    target=self.target,
                    categorical_features=self.categorical_features,
                    min_data_node=self.min_data_node,
                    current_depth=self.current_depth + 1,
                    max_depth=self.max_depth,
                    root_node=self.root_node,
                    linear_model_at_leaf=self.linear_model_at_leaf,
                    node_direction="left",
                    parent_node=self,
                    split_metric=self.split_metric,
                    error_metric=self.error_metric,
                )
                self.left_node = self.left_node.fit(
                    data_subset[
                        data_subset[self.node_feature] < self.node_split
                    ].copy(),
                    validation=validation,
                )
                left_tree_depth = self.left_node.tree_depth
            if (
                len(data_subset[data_subset[self.node_feature] >= self.node_split])
                >= self.min_data_node
            ):
                self.right_node = Node(
                    numerical_features=self.numerical_features,
                    target=self.target,
                    categorical_features=self.categorical_features,
                    min_data_node=self.min_data_node,
                    current_depth=self.current_depth + 1,
                    max_depth=self.max_depth,
                    root_node=self.root_node,
                    linear_model_at_leaf=self.linear_model_at_leaf,
                    node_direction="right",
                    parent_node=self,
                    split_metric=self.split_metric,
                    error_metric=self.error_metric,
                )
                self.right_node = self.right_node.fit(
                    data_subset[
                        data_subset[self.node_feature] >= self.node_split
                    ].copy(),
                    validation=validation,
                )
                right_tree_depth = self.right_node.tree_depth
        else:
            if self.linear_model_at_leaf:
                logging.info(
                    f"Fitting a linear model at the node {self.name} with data size of {len(data_subset)} at depth of {self.current_depth}"
                )
                linear_model = LinearRegression(fit_intercept=True)
                linear_model.fit(
                    data_subset[self.numerical_features], data_subset[self.target]
                )
                metric_lm = self.error_metric_func(
                    linear_model.predict(data_subset[self.numerical_features]),
                    data_subset[self.target],
                )
                logging.debug(
                    f"Linear model at at the node {self.name} has a {self.error_metric_name} of: {metric_lm}"
                )
                self.linear_model = linear_model
        self.tree_depth = max(left_tree_depth, right_tree_depth) + 1

        return self

    def _find_best_split_numeric(self, data, numeric_feature):
        metric_ = np.inf
        best_split, metric_, left_prediction, right_prediction = (
            None,
            metric_,
            data[self.target].mean(),
            data[self.target].mean(),
        )
        if len(data) > self.min_data_node:
            splits = np.sort(data[numeric_feature])
            for value in splits[np.arange(0, len(splits), self.bin_size)]:
                data.loc[data[numeric_feature] < value, "prediction"] = np.mean(
                    data.loc[data[numeric_feature] < value, self.target]
                )
                data.loc[data[numeric_feature] >= value, "prediction"] = np.mean(
                    data.loc[data[numeric_feature] >= value, self.target]
                )
                metric_temp = self.split_metric_func(
                    data["prediction"], data[self.target]
                )
                if metric_temp < metric_:
                    metric_ = metric_temp
                    best_split = value
                    left_prediction = np.mean(
                        data.loc[data[numeric_feature] < value, "target"]
                    )
                    right_prediction = np.mean(
                        data.loc[data[numeric_feature] >= value, "target"]
                    )
        return best_split, metric_, left_prediction, right_prediction

    def _find_best_feature(self, data):
        metric_, best_feature, best_split, left_prediction, right_prediction = (
            np.inf,
            None,
            None,
            data[self.target].mean(),
            data[self.target].mean(),
        )
        features = self.numerical_features.copy()
        for feature in features:
            (
                best_split_,
                metric_temp,
                left_prediction_,
                right_prediction_,
            ) = self._find_best_split_numeric(data, numeric_feature=feature)
            if metric_temp < metric_:
                metric_ = metric_temp
                best_feature = feature
                best_split = best_split_
                left_prediction = left_prediction_
                right_prediction = right_prediction_
        return metric_, best_feature, best_split, left_prediction, right_prediction

    def predict(self, data):
        if len(data) < 1:
            return pd.Series([], name="prediction", dtype=float)
        if self.node_feature is None:
            if self.linear_model is None:
                data["prediction"] = self.node_average
            else:
                print("here")
                data["prediction"] = self.linear_model.predict(
                    data[self.numerical_features]
                )
        else:
            data["prediction"] = np.nan
            if self.left_node is None:
                data.loc[
                    data[self.node_feature] < self.node_split, "prediction"
                ] = self.node_left_prediction
            else:
                data.loc[
                    data[self.node_feature] < self.node_split, "prediction"
                ] = self.left_node.predict(
                    data.loc[data[self.node_feature] < self.node_split].copy()
                )
            if self.right_node is None:
                data.loc[
                    data[self.node_feature] >= self.node_split, "prediction"
                ] = self.node_right_prediction
            else:
                data.loc[
                    data[self.node_feature] >= self.node_split, "prediction"
                ] = self.right_node.predict(
                    data.loc[data[self.node_feature] >= self.node_split].copy()
                )
        return data.prediction
