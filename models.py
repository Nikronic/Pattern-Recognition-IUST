# %% import libraries
import numpy as np
import pandas as pd
import pickle

from typing import Dict
from sklearn.naive_bayes import BaseNB
from sklearn.svm import base

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sentiment_analysis.utils import F, IO


# %% constants and inits
class BAYES_MODELS:
    """
    Enum of possible models
    """
    GAUSSIAN = 1
    POLYNOMIAL = 2
    BERNOULLI = 3


class SVM_MODELS:
    """
    Enum of possible models
    """
    SVC = 1
    LINEAR_SVC = 2


# %% functions
def build_svm(model_type: int, x: np.ndarray, y: np.ndarray, svc_kernel: str = None, save: bool = True,
              path: str = None, **kwargs) -> base:
    """
    Trains a SVM model

    :param model_type: The kernel of SVM model (see `SVM_MODELS` class)
    :param svc_kernel: The possible kernels for `SVC` model (It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’)
    It will be ignored if `model_type = LINEAR_SVC`
    :param x: Features in form of numpy ndarray
    :param y: Labels in form of numpy ndarray
    :param save: Whether save trained model on disc or not
    :param path: Path to save fitted model
    :param kwargs: A dictionary of other optional arguments of models in format of {'arg_name': value} (ex: {'C': 2})
    :return:A trained VSM model
    """
    if model_type == SVM_MODELS.SVC:
        model = SVC(kernel=svc_kernel, **kwargs)
    elif model_type == SVM_MODELS.LINEAR_SVC:
        model = LinearSVC(**kwargs)
    else:
        raise Exception('Model {} is not valid'.format(model_type))

    model.fit(x, y)

    if save:
        if path is None:
            path = ''
        pickle.dump(model, open(path + model.__module__ + '.model', 'wb'))
    return model


def build_bayes(model_type: int, x: np.ndarray, y: np.ndarray, save: bool = True, path: str = None, **kwargs) -> BaseNB:
    """
    Trains a naive bayes model

    :param model_type: The kernel of Naive Bayes model (see `BAYES_MODELS` class)
    :param x: Features in form of numpy ndarray
    :param y: Labels in form of numpy ndarray
    :param save: Whether save trained model on disc or not
    :param path: Path to save fitted model
    :return: A trained Naive Bayes model
    """

    if model_type == BAYES_MODELS.GAUSSIAN:
        model = GaussianNB(**kwargs)
    elif model_type == BAYES_MODELS.BERNOULLI:
        model = BernoulliNB(binarize=None, **kwargs)
    elif model_type == BAYES_MODELS.POLYNOMIAL:
        model = MultinomialNB(**kwargs)
    else:
        raise Exception('Kernel type "{}" is not valid.'.format(model_type))

    model.fit(x, y)

    if save:
        if path is None:
            path = ''
        pickle.dump(model, open(path + model.__module__ + '.model', 'wb'))
    return model


def build_decision_tree(x: np.ndarray, y: np.ndarray, save: bool = True,
                        path: str = None, **kwargs) -> DecisionTreeClassifier:
    """
    Trains a Decision Tree model

    :param x: Features in form of numpy ndarray
    :param y: Labels in form of numpy ndarray
    :param save: Whether save trained model on disc or not
    :param path: Path to save fitted model
    :return: A trained Decision Tree model
    """
    model = DecisionTreeClassifier(**kwargs)

    model.fit(x, y)

    if save:
        if path is None:
            path = ''
        pickle.dump(model, open(path + model.__module__ + '.model', 'wb'))
    return model


def build_random_forest(x: np.ndarray, y: np.ndarray, save: bool = True,
                        path: str = None, **kwargs) -> RandomForestClassifier:
    """
    Trains a Random Forest model

    :param x: Features in form of numpy ndarray
    :param y: Labels in form of numpy ndarray
    :param save: Whether save trained model on disc or not
    :param path: Path to save fitted model
    :return: A trained Random Forest model
    """
    model = RandomForestClassifier(**kwargs)

    model.fit(x, y)

    if save:
        if path is None:
            path = ''
        pickle.dump(model, open(path + model.__module__ + '.model', 'wb'))
    return model


# %% test
if __name__ == '__main__':
    # test bayes model
    df = IO.read_all_files('sentiment_analysis/data/sample/train/')
    df['text'] = df['text'].apply(F.tokenizer)
    vocabs = F.filter_vocabulary(F.generate_vocabulary(df, False), 50, False)
    df['text'] = df['text'].apply(F.build_bow, args=(vocabs,))
    df['text'] = df['text'].apply(np.reshape, args=((1, -1),))
    x = np.concatenate(df['text'].to_numpy(), axis=0)
    y = df['label'].to_numpy()

    # bayes test
    bayes = build_bayes(BAYES_MODELS.GAUSSIAN, x, y, True, None)

    bayes.predict(x)
    bayes.score(x, y)

    # svm test
    svm = build_svm(model_type=SVM_MODELS.SVC, x=x, y=y, svc_kernel='rbf')
    svm.predict(x)
    svm.score(x, y)

    svm = build_svm(model_type=SVM_MODELS.LINEAR_SVC, x=x, y=y)
    svm.predict(x)
    svm.score(x, y)

    # decision tree test
    dt = build_decision_tree(x=x, y=y)
    dt.predict(x)
    dt.score(x, y)

    # random forest test
    rf = build_random_forest(x=x, y=y)
    rf.predict(x)
    rf.score(x, y)
