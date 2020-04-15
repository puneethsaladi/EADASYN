import numpy as np
import matplotlib.pyplot as plt
from random import randint
from imblearn.over_sampling import *
from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import glob, os

def smote(X, Y, ratio = 1, random_state = 0):
    sm = SMOTE(sampling_strategy = ratio, 
               random_state = random_state, 
               n_jobs = -1)
    return sm.fit_resample(X, Y)

def borderline_smote(X, Y, ratio = 1, random_state = 0):
    sm = BorderlineSMOTE(sampling_strategy = ratio, 
                         random_state = random_state, 
                         n_jobs = -1)
    return sm.fit_resample(X, Y)

def svm_smote(X, Y, ratio = 1, random_state = 0):
    sm = SVMSMOTE(sampling_strategy = ratio, 
                  random_state = random_state, 
                  n_jobs = -1)
    return sm.fit_resample(X, Y)

def adasyn(X, Y, ratio = 1, random_state = 0):
    ada = ADASYN(sampling_strategy = ratio, 
                 random_state = random_state, 
                 n_jobs = -1)
    return ada.fit_resample(X, Y)

# Assumption: All the minority class samples appear after majority class samples;
# implemented this way in load_data()
def eadasyn(X, Y, clf = DecisionTreeClassifier(random_state = 0), ratio = 1, 
            num_candidates = 10, iter = 20, fitness_metric = "f1", plot = False):
    candidates = []
    num_pairs = int(num_candidates*0.2)
    original_size = len(X)
    
    # Generate the candidates
    for i in range(num_candidates):
        X_res, Y_res = adasyn(X, Y, ratio, random_state = i*i)
        candidates.append([X_res,Y_res])
    
    res_size = len(candidates[0][0])
    scores = []
    cur_mean_score = 0
    
    if(plot):
        arr_min = []
        arr_max = []
        arr_mean = []
    
    # Evolution steps
    for i in range(iter):
        scores = []
        
        # Calculate fitness level for each candidate
        for c in range(num_candidates):
            skf = StratifiedKFold(n_splits=5)
            score = cross_val_score(clf, X = candidates[c][0], y = candidates[c][1], scoring = fitness_metric, cv = skf)
            scores.append(np.mean(score))
        
        mean_score = np.mean(scores)
        scores = np.array(scores)
        if(plot):
	        arr_min.append(scores[np.argmin(scores)])
	        arr_max.append(scores[np.argmax(scores)])
	        arr_mean.append(mean_score)
        
        # # If there is no improvement, early stop the number of evolutionary steps
        # if mean_score <= cur_mean_score:
        #     break
        # else:
        #     cur_mean_score = mean_score
        
        # Eliminate the least fit candidates
        for i in range(num_pairs*2):
            del candidates[np.argmin(scores)]
            scores = np.delete(scores,np.argmin(scores))
            
        # Normalize the remaining fitness levels to sum to 1
        scores /= sum(scores)
        
        # Crossover the remaining candiates with probability 
        # proportional to their fitness level
        for i in range(num_pairs):
            parent_indices = np.random.choice(num_candidates-num_pairs*2, size = 2, p = scores, replace = False)
            print(parent_indices)
            crossover_point = randint(original_size,res_size-1)
            print(crossover_point)
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
            
    # Plot for rq3        
    if(plot):
        l = len(arr_min)
        plt.plot(range(1,l+1),arr_min, marker='o', color = "blue", label = "Minimum fitness")
        plt.plot(range(1,l+1),arr_max, marker='o', color = "red", label = "Maximum fitness")
        plt.plot(range(1,l+1),arr_mean, marker='o', color = "brown", label = "Mean fitness")
        plt.xticks(np.arange(1, l+1, step=1))
        plt.xlabel("evolutionary steps")
        plt.ylabel("F1 scores")
        plt.legend()
        plt.savefig('rq3.png', dpi=400)
            
    # Determine the best of the available candidates
    scores = []
    for c in range(num_candidates):
        skf = StratifiedKFold(n_splits=5)
        score = cross_val_score(clf, X = candidates[c][0], y = candidates[c][1], scoring = fitness_metric, cv = skf)
        scores.append(np.mean(score))
    return candidates[np.argmax(scores)]

def load_data(filename):
    
    X_neg = []
    Y_neg = []
    X_pos = []
    Y_pos = []
    
    f = open(filename,'r')
    for line in f:
        tokens = line.rstrip().split(",")
        if tokens[-1]=="negative":
            X_neg.append([float(i) for i in tokens[:-1]])
            Y_neg.append(0)
        elif tokens[-1]=="positive":
            X_pos.append([float(i) for i in tokens[:-1]])
            Y_pos.append(1)
        else:
            raise Exception("Issue with dataset target values.")
    f.close()
    
    test_pos = int(len(X_pos)/5)
    test_neg = int(len(X_neg)/5)
    
    X_train = X_pos[:-test_pos]
    X_train.extend(X_neg[:-test_neg])
    X_test = X_pos[-test_pos:]
    X_test.extend(X_neg[-test_neg:])
    
    Y_train = Y_pos[:-test_pos]
    Y_train.extend(Y_neg[:-test_neg])
    Y_test = Y_pos[-test_pos:]
    Y_test.extend(Y_neg[-test_neg:])
    
    return [X_train, X_test, Y_train, Y_test]
    
# Performance evaluation of EADASYN against other oversampling techniques
def rq1():

    clf = DecisionTreeClassifier(random_state = 0)

    f = open("results/rq1.txt",'w')

    for file in glob.glob("data/*.dat"):

        print("Working on " + file)

        X_train, X_test, Y_train, Y_test = load_data(file)

        Y_pred = clf.fit(X_train, Y_train).predict(X_test)
        f1_normal = f1_score(Y_test,Y_pred)

        X_res, Y_res = smote(X_train, Y_train)
        Y_pred = clf.fit(X_res, Y_res).predict(X_test)
        f1_smote = f1_score(Y_test,Y_pred)

        X_res, Y_res = borderline_smote(X_train, Y_train)
        Y_pred = clf.fit(X_res, Y_res).predict(X_test)
        f1_borderline_smote = f1_score(Y_test,Y_pred)

        X_res, Y_res = svm_smote(X_train, Y_train)
        Y_pred = clf.fit(X_res, Y_res).predict(X_test)
        f1_svm_smote = f1_score(Y_test,Y_pred)

        X_res, Y_res = adasyn(X_train, Y_train)
        Y_pred = clf.fit(X_res, Y_res).predict(X_test)
        f1_ada = f1_score(Y_test,Y_pred)

        X_res, Y_res = eadasyn(X_train, Y_train, clf, fitness_metric = 'f1')
        Y_pred = clf.fit(X_res, Y_res).predict(X_test)
        f1_eada = f1_score(Y_test,Y_pred)

        f.write(file + "\n")
        f.write("F1 score on original dataset: " + str(round(f1_normal,3)) + "\n")
        f.write("F1 score on smote dataset: " + str(round(f1_smote,3)) + "\n")
        f.write("F1 score on borderline smote dataset: " + str(round(f1_borderline_smote,3)) + "\n")
        f.write("F1 score on svm smote dataset: " + str(round(f1_svm_smote,3)) + "\n")
        f.write("F1 score on adasyn dataset: " + str(round(f1_ada,3)) + "\n")
        f.write("F1 score on eadasyn dataset: " + str(round(f1_eada,3)) + "\n\n")

        print("Completed 1 file")

    f.close()

# Fitness measure to choose
def rq2():
    
    clf = DecisionTreeClassifier(random_state = 0)
    
    fitness_measures = ['accuracy','precision','recall','f1','roc_auc']
    
    f = open("results/rq2.txt",'w')

    for file in glob.glob("data/*.dat"):

        print("Working on " + file)
        f.write(file + "\n")

        X_train, X_test, Y_train, Y_test = load_data(file)
        
        for fitness in fitness_measures:
            X_res, Y_res = eadasyn(X_train, Y_train, clf, fitness_metric = fitness)
            Y_pred = clf.fit(X_res, Y_res).predict(X_test)
            f1 = f1_score(Y_test,Y_pred)
            
            print("completed " + fitness)
            f.write("F1 score using " + fitness + " as fitness measure: " + str(round(f1,3)) + "\n")
        
        print("Completed 1 file")
        f.write('\n')

    f.close()

# Evolution of candidate solutions w.r.t fitness
def rq3():

    clf = DecisionTreeClassifier()

    X_train, X_test, Y_train, Y_test = load_data("data/ecoli-0-1-4-7_vs_5-6.dat")
    X_res, Y_res = eadasyn(X_train, Y_train, clf, fitness_metric = 'f1', plot = True)


if __name__ == '__main__':
    # Specify the reseach question function to run
    # Or just use eadasyn() to run the proposed algorithm
