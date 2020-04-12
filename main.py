from random import *
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

genetic_knn_f1 = []
genetic_rf_f1 = []
genetic_ada_f1 = []

smote_knn_f1 = []
smote_rf_f1 = []
smote_ada_f1 = []

genetic_knn_kappa = []
genetic_rf_kappa = []
genetic_ada_kappa = []

smote_knn_kappa = []
smote_rf_kappa = []
smote_ada_kappa = []

genetic_knn_acc = []
genetic_rf_acc = []
genetic_ada_acc = []

smote_knn_acc = []
smote_rf_acc = []
smote_ada_acc = []

graph_mean = []
graph_min = []
graph_max = []


def genetic():
	for master_iter in range(3):
		count_positive = 0
		count_negative = 0

		g = open("negative.txt",'r')

		negative = []
		y = []
		for line in g:
			count_negative+=1
			y = line.split(",")
			del y[-1]
			temp = []
			for i in range(len(y)):
				temp.append(float(y[i]))
			negative.append(temp)

		g.close()

		f = open("positive.txt",'r')

		pool = []
		y = []
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		pool.append([])
		for line in f:
			count_positive+=1;
			for x in range(0,10):
				y = line.split(",")
				del y[-1]
				temp = []
				for i in range(len(y)):
					temp.append(float(y[i]))
				pool[x].append(temp)

		f.close()

		print "count positive = " + str(count_positive)
		print "count negative = " + str(count_negative)

		for x in range(0,10):
			pool_count = count_positive	
			for y in range(0,(count_negative - count_positive)/2):
				p1 = pool[x][randint(0,pool_count-1)]
				p2 = pool[x][randint(0,pool_count-1)]
				a = uniform(0,1)
				b = [item*a for item in p1]
				c = [item*(1-a) for item in p2]
				c1 = [float(num1+num2) for num1,num2 in zip(b,c)]
				b = [item*(1-a) for item in p1]
				c = [item*a for item in p2]
				c2 = [float(num1+num2) for num1,num2 in zip(b,c)]
				pool[x].append(c1)
				pool[x].append(c2)
				pool_count+=2

		for x in range(0,10):
			for i in range(0,2*x):
				pool[x][randint(0,count_negative-1)] = negative[randint(0,count_negative-1)]


		# First iteration begins

		X = []
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		X.append([])
		for i in range(10):
			for j in range(count_negative):
				X[i].append(pool[i][j])
				X[i].append(negative[j])

		Y = []
		for i in range(count_negative):
			Y.append(1)
			Y.append(-1)

		if master_iter==0:
			clf = KNeighborsClassifier()
		elif master_iter==1:
			clf = RandomForestClassifier()
		else:
			clf = AdaBoostClassifier()


		x_axis = []
		y_axis = []


		for iteration in range(15):

			accuracy = []	# contains acc score of each set of pool.

			for i in range(10):

				skf = StratifiedKFold(n_splits=5)
				scores = []
				for train_indices,test_indices in skf.split(X[i],Y):
					trainX = []
					testX = []
					trainY = []
					testY = []
					for k in train_indices:
						trainX.append(X[i][k])
						trainY.append(Y[k])
					for k in test_indices:
						testX.append(X[i][k])
						testY.append(Y[k])
					
					scores.append(clf.fit(trainX, trainY).score(testX, testY))

				accuracy.append(reduce(lambda x,y:x+y,scores)/len(scores))

			print "Accuracy of 10 pools at iteration " + str(iteration)
			print accuracy
			mean = reduce(lambda x,y:x+y,accuracy)/len(accuracy)
			print "mean accuracy at iteration " + str(iteration)
			print mean

			if (master_iter==1):  

				accuracy_np = np.array(accuracy)
				minimum = accuracy[np.argmin(accuracy_np)]
				maximum = accuracy[np.argmax(accuracy_np)]

				graph_mean.append(mean)
				graph_min.append(minimum)
				graph_max.append(maximum)
			

			for i in range(4):
				accuracy_np = np.array(accuracy)
				del X[np.argmin(accuracy_np)]
				del accuracy[np.argmin(accuracy_np)]

			print "the population left after reduction have accuracies -"
			print accuracy

			total = reduce(lambda x,y:x+y,accuracy)
			for i in range(6):
				accuracy[i] = float(accuracy[i])/total
			parent_indices = []
			for x in range(2):
				indice = np.random.choice(6,2,accuracy)
				flag = 0
				while flag==0:
					if indice[0]==indice[1]:
						indice = np.random.choice(6,2,accuracy)
					else:
						flag = 1
				parent_indices.append(indice[0])
				parent_indices.append(indice[1])

			print "Indices of chosen pools for breading - "
			print parent_indices

			for i in range(2):
				p1 = parent_indices[0+2*i]
				p2 = parent_indices[1+2*i]
				c1 = []
				c2 = []
				crossover_point = randint(1,2*count_negative-1)	#random crossover point
				for j in range(crossover_point):
					c1.append(X[p1][j])
					c2.append(X[p2][j])
				for j in range(crossover_point,2*count_negative):
					c1.append(X[p2][j])
					c2.append(X[p1][j])
				X.append(c1)
				X.append(c2)

			print "\n\n"

		#########	Testing		#############

		accuracy = []	# contains acc score of each set of pool.

		for i in range(10):

			skf = StratifiedKFold(n_splits=5)
			scores = []
			for train_indices,test_indices in skf.split(X[i],Y):
				trainX = []
				testX = []
				trainY = []
				testY = []
				for k in train_indices:
					trainX.append(X[i][k])
					trainY.append(Y[k])
				for k in test_indices:
					testX.append(X[i][k])
					testY.append(Y[k])
				scores.append(clf.fit(trainX, trainY).score(testX, testY))

			accuracy.append(reduce(lambda x,y:x+y,scores)/len(scores))

		accuracy_np = np.array(accuracy)
		X_select = X[np.argmax(accuracy_np)]

		clf.fit(X_select,Y)

		y_true = []
		X_test = []

		f = open("test_dataset.txt",'r')
		for line in f:
			y = line.split(",")
			if("positive" in y[-1]):
				y_true.append(1)
			else:
				y_true.append(-1)
			del y[-1]
			temp = []
			for i in range(len(y)):
				temp.append(float(y[i]))
			X_test.append(temp)

		y_pred = np.array(clf.predict(X_test))

		accuracy_score = clf.score(X_test,y_true)
		precision = precision_score(y_true,y_pred,average='binary')
		recall = recall_score(y_true,y_pred,average='binary')
		f1 = f1_score(y_true,y_pred,average='binary')
		kappa = cohen_kappa_score(y_true, y_pred)


		print "Accuracy score on test data - "
		print accuracy_score

		print "Precision score on test data - "
		print precision

		print "Recall score on test data - "
		print recall

		print "F1 score on test data - "
		print f1

		print "Kappa score on test data - "
		print kappa

		if master_iter==0:
			genetic_knn_f1.append(f1)
			genetic_knn_kappa.append(kappa)
			genetic_knn_acc.append(accuracy_score)
		elif master_iter==1:
			genetic_rf_f1.append(f1)
			genetic_rf_kappa.append(kappa)
			genetic_rf_acc.append(accuracy_score)
		else:
			genetic_ada_f1.append(f1)
			genetic_ada_kappa.append(kappa)
			genetic_ada_acc.append(accuracy_score)

		ff = open("results.txt",'a')
		if master_iter==0:
			ff.write("Genetic results for knn\n\n")
		elif master_iter==1:
			ff.write("Genetic results for Random Forest\n\n")
		else:
			ff.write("Genetic results for Ada Boost\n\n")
		ss = "Accuracy - " + str(accuracy_score) + "\nPrescision - " + str(precision) + "\nRecall - " + str(recall) + "\nF1 - " + str(f1) + "\nKappa - " + str(kappa) + "\n\n"
		ff.write(ss)
		ff.close()



def smote():
	for master_iter in range(3):
		X = []
		Y = []
		count_pos = 0
		count_neg = 0

		f = open("positive.txt",'r')
		for line in f:
			count_pos+=1
			l = line.split(",")
			del l[-1]
			temp = []
			for i in range(len(l)):
				temp.append(float(l[i]))
			X.append(temp)
			Y.append(1)
		f.close()

		g = open("negative.txt",'r')
		for line in g:
			count_neg+=1
			l = line.split(",")
			del l[-1]
			temp = []
			for i in range(len(l)):
				temp.append(float(l[i]))
			X.append(temp)
			Y.append(-1)
		g.close()

		print "count positive - " + str(count_pos)
		print "count negative - " + str(count_neg)
		print "nos instances before oversampling - " + str(len(X))

		sm = SMOTE(random_state = 42)
		X_final,Y_final = sm.fit_sample(X,Y)

		print "nos instances after oversampling by smote - " + str(len(X_final))

		if master_iter==0:
			clf = KNeighborsClassifier()
		elif master_iter==1:
			clf = RandomForestClassifier()
		else:
			clf = AdaBoostClassifier()


		clf.fit(X_final,Y_final)

		y_true = []
		X_test = []

		f = open("test_dataset.txt",'r')
		for line in f:
			l = line.split(",")
			if("positive" in l[-1]):
				y_true.append(1)
			else:
				y_true.append(-1)
			del l[-1]
			temp = []
			for i in range(len(l)):
				temp.append(float(l[i]))
			X_test.append(temp)

		print "The true labels of test instances - "
		print y_true

		y_pred = np.array(clf.predict(X_test))
		print "The predicted labels of test instances - "
		print y_pred

		accuracy_score = clf.score(X_test,y_true)
		precision = precision_score(y_true,y_pred,average='binary')
		recall = recall_score(y_true,y_pred,average='binary')
		f1 = f1_score(y_true,y_pred,average='binary')
		kappa = cohen_kappa_score(y_true, y_pred)


		print "Accuracy score on test data - "
		print accuracy_score

		print "Precision score on test data - "
		print precision

		print "Recall score on test data - "
		print recall

		print "F1 score on test data - "
		print f1

		print "Kappa score on test data - "
		print kappa

		if master_iter==0:
			smote_knn_f1.append(f1)
			smote_knn_kappa.append(kappa)
			smote_knn_acc.append(accuracy_score)
		elif master_iter==1:
			smote_rf_f1.append(f1)
			smote_rf_kappa.append(kappa)
			smote_rf_acc.append(accuracy_score)
		else:
			smote_ada_f1.append(f1)
			smote_ada_kappa.append(kappa)
			smote_ada_acc.append(accuracy_score)

		ff = open("results.txt",'a')
		if master_iter==0:
			ff.write("SMOTE results for knn\n\n")
		elif master_iter==1:
			ff.write("SMOTE results for Random Forest\n\n")
		else:
			ff.write("SMOTE results for Ada Boost\n\n")
		ss = "Accuracy - " + str(accuracy_score) + "\nPrescision - " + str(precision) + "\nRecall - " + str(recall) + "\nF1 - " + str(f1) + "\nKappa - " + str(kappa) + "\n\n"
		ff.write(ss)
		ff.close()

f = open("results.txt",'w')
f.write("Comparision of SMOTE and GA with iterations = 15\n\n")
f.close()

datasets_list = []

import glob, os
os.chdir("") # add the path which has the datasets
for file in glob.glob("*.dat"):
	datasets_list.append(file)
	count_pos = 0
	count_neg = 0
	N = []
	P = []
	f = open(file,'r')
	for line in f:
		l = line.split(",")
		if("negative" in l[-1]):
			count_neg+=1
			N.append(line)
		else:
			P.append(line)
			count_pos+=1
	f.close()
	cp = count_pos
	cn = count_neg
	print "pos - " + str(count_pos)
	print "neg - " + str(count_neg)
	test_pos = count_pos/5
	test_neg = count_neg/5
	f = open("test_dataset.txt",'w')
	for x in range(test_pos):
		f.write(P[-1])
		del P[-1]
	for x in range(test_neg):
		f.write(N[-1])
		del N[-1]
	count_pos-= test_pos
	count_neg-= test_neg
	if((count_neg-count_pos)%2!=0):
		f.write(N[-1])
		del N[-1]
	f.close()

	f1 = open("positive.txt",'w')
	f2 = open("negative.txt",'w')
	for x in range(len(P)):
		f1.write(P[x])
	f1.close()
	for x in range(len(N)):
		f2.write(N[x])
	f1.close()
	f2.close()

	g = open("results.txt",'a')
	g.write(file + "\n\n")
	g.write("Count of positive samples - " + str(cp) + "\n")
	g.write("Count of negative samples - " + str(cn) + "\n")
	g.write("Imbalance Ratio - " + str(float(cn)/cp) + "\n\n")
	g.close()
	smote()
	genetic()