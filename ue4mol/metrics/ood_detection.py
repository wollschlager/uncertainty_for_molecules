import numpy as np
import torch
from src.metrics.auc_scores import auc_apr, auc_roc

# OOD detection metrics
# check pytorch metrics for speedup
def anomaly_detection(sigmas, ood_sigmas, score_type='AUROC'):
    
    corrects = np.concatenate([np.ones(sigmas.size(0)), np.zeros(ood_sigmas.size(0))], axis=0)
    scores = np.concatenate([sigmas, ood_sigmas], axis=0)
        
    if score_type == 'AUROC':
        return auc_roc(corrects, scores)
    elif score_type == 'APR':
        return auc_apr(corrects, scores)
    else:
        raise NotImplementedError
    
    
    
    
