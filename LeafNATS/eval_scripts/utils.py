'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def eval_dmsc(preds, golds):
    '''
    evaluate accuracy
    Here, the labels cannot be 0. 
    They need to be positive integer numbers.
    input:
    -- predicted labels, golden labels
    output:
    -- accuracy, mean squared error
    '''
    nm = len(preds)
    
    preds = np.array(preds)
    golds = np.array(golds)
    
    diff = preds - golds
    diff = diff * diff
    
    accu = preds - golds
    accu[accu != 0] = 1.0
    accu = 1.0 - accu
    
    return np.sum(accu)/nm, np.sum(diff)/nm