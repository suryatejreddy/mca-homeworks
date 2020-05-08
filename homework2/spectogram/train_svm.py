import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
PICKLE_PREFIX = './pickle_files/spectogram_noise_data_'
PICKLE_VAL_PREFIX = "./pickle_files/spectogram_vanilla_data_val_"

def remove_infinities(X):
	X = np.where(np.isinf(X), np.nan, X)
	X = np.nan_to_num(X)
	return X

def read_file(filename):
	X = []
	y = []
	for i in range(1,4):
		with open(filename + str(i) + ".pkl", "rb") as f:
			L = pickle.load(f)
			features = L[0]
			classes = L[1]
			X.extend(features)
			y.extend(classes)
	for i in range(len(X)):
		X[i] = X[i].flatten()
	X = np.array(X)
	X = remove_infinities(X)
	return X, y

def load_data():
	X_train, y_train = read_file(PICKLE_PREFIX)
	X_val, y_val = read_file(PICKLE_VAL_PREFIX)
	return X_train, y_train, X_val, y_val

def print_performance(X_train, y_train, X_val = None, y_val = None):
	with open('svm_with_noise.pkl' , 'rb') as f:
		clf = pickle.load(f)
	'''
	print ("Training Set")
	y_train_pred = clf.predict(X_train)
	print (accuracy_score(y_train, y_train_pred))
	print(precision_recall_fscore_support(y_train, y_train_pred, average='micro'))
	print (confusion_matrix(y_train, y_train_pred))
	'''

	print ("Validation Set")
	y_val_pred = clf.predict(X_val)
	print (accuracy_score(y_val, y_val_pred))
	print(precision_recall_fscore_support(y_val, y_val_pred, average='micro'))
	print (confusion_matrix(y_val, y_val_pred))


def train_svm(X_train,y_train):
	clf = SVC(kernel = 'linear', C = 0.5 ,gamma='auto')
	print ("Fitting SVM")
	clf.fit(X_train, y_train)
	with open('svm_with_noise.pkl' ,'wb') as f:
		pickle.dump(clf, f)	
	print ("Model Saved Succesfully")

X_train, y_train, X_val, y_val = load_data()
train_svm(X_train,y_train)
print_performance(X_train,y_train, X_val, y_val)
