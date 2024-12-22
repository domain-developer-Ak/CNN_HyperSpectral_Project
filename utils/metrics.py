import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def calculate_accuracy(predictions, labels):
    """
    Calculates the accuracy of predictions.
    """
    return np.mean(np.array(predictions) == np.array(labels))

def calculate_confusion_matrix(predictions, labels, num_classes=None):
    """
    Computes the confusion matrix.
    """
    return confusion_matrix(labels, predictions, labels=list(range(num_classes)) if num_classes else None)

def calculate_precision(predictions, labels, average='weighted'):
    """
    Computes the precision score.
    """
    return precision_score(labels, predictions, average=average)

def calculate_recall(predictions, labels, average='weighted'):
    """
    Computes the recall score.
    """
    return recall_score(labels, predictions, average=average)

def calculate_f1(predictions, labels, average='weighted'):
    """
    Computes the F1 score.
    """
    return f1_score(labels, predictions, average=average)
