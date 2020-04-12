### Author: Puneeth Saladi 
### Date: March 24, 2020

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import *
from sklearn.svm import SVC

def smote(X, Y, ratio = 1):
    sm = SMOTE(sampling_strategy = ratio)
    return sm.fit_resample(X, Y)

# Assumption: All the minority class samples appear after majority class samples.
# This is true for all the imbalanced KEEL datasets
def esmote(X, Y, clf, ratio = 1, num_candidates = 10, iter = 15):
    candidates = []
    num_pairs = int(num_candidates*0.2)
    original_size = len(X)
    
    # Generate the candidates
    for i in range(num_candidates):
        X_res, Y_res = smote(X, Y, ratio)
        candidates.append([X_res,Y_res])
        
    res_size = len(candidates[0][0])
    f1_scores = []
    
    # Evolution steps
    for i in range(iter):
        f1_scores = []
        
        # Calculate fitness level (f1) for each candidate
        for c in range(num_candidates):
            skf = StratifiedKFold(n_splits=5)
            f1 = cross_val_score(clf, X = candidates[c][0], y = candidates[c][1], scoring = 'f1', cv = skf)
            f1_scores.append(np.mean(f1))
        
        # Eliminate the least fit candidates
        f1_scores = np.array(f1_scores)
        for i in range(num_pairs*2):
            del candidates[np.argmin(f1_scores)]
            f1_scores = np.delete(f1_scores,np.argmin(f1_scores))
            
        # Normalize the remaining fitness levels to sum to 1
        f1_scores /= sum(f1_scores)
        
        # Crossover the remaining candiates with probability 
        # proportional to their fitness level
        for i in range(num_pairs):
            parent_indices = np.random.choice(num_candidates-num_pairs*2, size = 2, p = f1_scores, replace = False)
            crossover_point = randint(original_size,res_size-1)
            c1_Y = candidates[parent_indices[0]][1]
            c2_Y = candidates[parent_indices[1]][1]
            
            # Actual crossover
            c1_X = candidates[parent_indices[0]][0][:crossover_point]
            c1_X.extend(candidates[parent_indices[1]][0][crossover_point:])
            c2_X = candidates[parent_indices[1]][0][:crossover_point]
            c2_X.extend(candidates[parent_indices[0]][0][crossover_point:])
            
            # Add the children to the candidates set
            candidates.append([c1_X,c1_Y])
            candidates.append([c2_X,c2_Y])
            
    # Determine the best of the available candidates
    f1_scores = []
    for c in range(num_candidates):
        skf = StratifiedKFold(n_splits=5)
        f1 = cross_val_score(clf, X = candidates[c][0], y = candidates[c][1], scoring = 'f1', cv = skf)
        f1_scores.append(np.mean(f1))
    return candidates[np.argmax(f1_scores)]