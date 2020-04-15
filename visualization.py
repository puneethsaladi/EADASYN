from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import *
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler
from random import randint
from imblearn.over_sampling import *
from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import glob, os

def create_dataset(n_samples=1000, weights=(0.02, 0.98), n_classes=2,
                   class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)

def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)

def adasyn(X, Y, ratio = 1):
    ada = ADASYN(sampling_strategy = ratio)
    return ada.fit_resample(X, Y)

# Needed a few changes from the origianl function to support imbearn sampler format
def eadasyn(X, Y, clf, ratio = 1, num_candidates = 10, iter = 20, fitness_metric = "f1"):
    X = X.tolist()
    Y = Y.tolist()
    candidates = []
    num_pairs = int(num_candidates*0.2)
    original_size = len(X)
    
    # Generate the candidates
    for i in range(num_candidates):
        X_res, Y_res = adasyn(X, Y, ratio)
        candidates.append([X_res,Y_res])
    
    res_size = len(candidates[0][0])
    scores = []
    cur_mean_score = 0
    
    # Evolution steps
    for i in range(iter):
        scores = []
        
        # Calculate fitness level for each candidate
        for c in range(num_candidates):
            skf = StratifiedKFold(n_splits=5)
            score = cross_val_score(clf, X = candidates[c][0], y = candidates[c][1], scoring = fitness_metric, cv = skf)
            scores.append(np.mean(score))
        
        mean_score = np.mean(scores)
        
        # If there is no improvement, early stop the number of evolutionary steps
        if mean_score == cur_mean_score:
            break
        else:
            cur_mean_score = mean_score
        
        # Eliminate the least fit candidates
        scores = np.array(scores)
        for i in range(num_pairs*2):
            del candidates[np.argmin(scores)]
            scores = np.delete(scores,np.argmin(scores))
            
        # Normalize the remaining fitness levels to sum to 1
        scores /= sum(scores)
        
        # Crossover the remaining candiates with probability 
        # proportional to their fitness level
        for i in range(num_pairs):
            parent_indices = np.random.choice(num_candidates-num_pairs*2, size = 2, p = scores, replace = False)
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
    scores = []
    for c in range(num_candidates):
        skf = StratifiedKFold(n_splits=5)
        score = cross_val_score(clf, X = candidates[c][0], y = candidates[c][1], scoring = fitness_metric, cv = skf)
        scores.append(np.mean(score))
    return np.asarray(candidates[np.argmax(scores)][0]),np.asarray(candidates[np.argmax(scores)][1])

class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y

class EADASYN(BaseSampler):
    
    _sampling_type = 'over-sampling'

    def _fit_resample(self, X, y):
        clf = DecisionTreeClassifier()
        return eadasyn(X, y, clf)

if __name__ == '__main__':
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
    X, y = create_dataset(n_samples=1000, weights=(0.94, 0.06))
    sampler = FakeSampler()
    clf = make_pipeline(sampler, LinearSVC())
    plot_resampling(X, y, sampler, ax1)
    ax1.set_title('Original data - y={}'.format(Counter(y)))

    ax_arr = (ax2, ax3, ax4, ax5, ax6)
    for ax, sampler in zip(ax_arr, (SMOTE(),
                                    BorderlineSMOTE(),
                                    SVMSMOTE(),
                                    ADASYN(),
                                    EADASYN())):
        clf = make_pipeline(sampler, LinearSVC())
        clf.fit(X, y)
        plot_resampling(X, y, sampler, ax)
        ax.set_title('Resampling using {}'.format(sampler.__class__.__name__))
    fig.tight_layout(w_pad=10)
    plt.savefig('visualization.png', dpi=300)
