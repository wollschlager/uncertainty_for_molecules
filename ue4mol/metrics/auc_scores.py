from sklearn import metrics


def auc_roc(pred, true):
    fpr, tpr, thresholds = metrics.roc_curve(pred.reshape(-1), true.reshape(-1))
    return metrics.auc(fpr, tpr)


def auc_apr(pred, true):
    return metrics.average_precision_score(pred.reshape(-1), true.reshape(-1))
