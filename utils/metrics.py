# %% import libraries
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, List


# %% functions
def accuracy(cm: np.ndarray) -> float:
    """
    Calculates accuracy using given confusion matrix

    :param cm: Confusion matrix in form of np.ndarray
    :return: A float number
    """
    true = np.trace(cm)
    false = np.sum(cm) - true
    ac = true / (true + false)
    return ac


def precision(cm: np.ndarray) -> np.ndarray:
    """
    Precision of the model using given confusion matrix

    :param cm: Confusion matrix in form of np.ndarray
    :return: A list of precisions with size of number of classes
    """
    tp = np.diag(cm)
    tpp = np.sum(cm, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.true_divide(tp, tpp)
        p[~ np.isfinite(p)] = 0
        p = np.nan_to_num(p)
    return p


def recall(cm: np.ndarray) -> np.ndarray:
    """
    Recall of the model using given confusion matrix

    :param cm: Confusion matrix in form of np.ndarray
    :return: A list of recalls with size of number of classes
    """
    tp = np.diag(cm)
    tap = np.sum(cm, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.true_divide(tp, tap)
        r[~ np.isfinite(r)] = 0
        r = np.nan_to_num(r)
    return r


def f1(cm: np.ndarray) -> float:
    """
    F1 score of the model using given confusion matrix

    :param cm: Confusion matrix in form of np.ndarray
    :return: A float number
    """
    p = precision(cm)
    r = recall(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.true_divide(p * r, p + r)
        f[~ np.isfinite(f)] = 0
        f = np.nan_to_num(f)
    return f * 2


def roc_auc(true, pred, plot=False) -> Tuple[float, Tuple[List[float], List[float], List[float]]]:
    """
    Calculates AUC and ROC on binary data

    :param true: True labels
    :param pred: Predicted labels
    :param plot: Whether or not plot the ROC curve
    :return: A tuple of (AUC value, ROC curve values)
    """
    fpr, tpr, th = roc_curve(true, pred, pos_label=1)
    auc = roc_auc_score(true, pred)
    if plot:
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.show()
    return auc, (fpr, tpr, th)


# %% tests
if __name__ == '__main__':

    from sklearn.metrics import precision_recall_fscore_support
    x = np.random.randint(0, 2, (9,))
    y = np.random.randint(0, 2, (9,))
    conf_matrix = confusion_matrix(x, y)
    assert precision_recall_fscore_support(x, y)[0].all() == precision(conf_matrix).all()
    assert precision_recall_fscore_support(x, y)[1].all() == recall(conf_matrix).all()
    assert precision_recall_fscore_support(x, y)[2].all() == f1(conf_matrix).all()

