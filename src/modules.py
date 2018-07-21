import os
import pickle
import zipfile
from os.path import isfile
from string import punctuation
import random, time, json

import numpy as np

from .preprocess import get_normalized_data

from .bow import make_vocabs

from .vectorizers import binary_vectorizer, \
	count_vectorizer, \
	tfidf_vectorizer

from .smote import SMOTE as smote

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .model_utils import get_best_model, \
	do_training_testing, \
	do_validation

from .metrics import accuracy_score, \
	confusion_matrix, \
	geometric_mean_score as gmean

from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer



def get_raw_data():

	app_path = os.getcwd()

	with open(app_path+'/data/data.json', 'r') as file:
		raw_data = json.load(file)

	n_data = 1500
	raw_data = raw_data[:n_data]
	random.seed(123)
	random.shuffle(raw_data)

	return raw_data

def get_data_rasio_info(raw_data):
	labels = [data['class'] for data in raw_data]
	neg = [i for i in labels if i == 0]
	n_neg = len(neg)
	n_pos = 1500 - n_neg

	return n_pos, n_neg

def get_vectorized_data(pos_texts_normalized, neg_texts_normalized):

	app_path = os.getcwd()
	file_path = app_path+'/data/dinamics/texts_vector.pkl'

	if isfile(file_path):
		print('reading pickle data...', file_path)
		with open(file_path, 'rb') as data:
			X, y, X_val, y_val = pickle.load(data)

	else:
		print('writing pickle data...', file_path)
		n_val_pos = 158
		n_val_neg = 142

		pos_tt = pos_texts_normalized[:-n_val_pos]
		neg_tt = neg_texts_normalized[:-n_val_neg]

		pos_val = pos_texts_normalized[-n_val_pos:]
		neg_val = neg_texts_normalized[-n_val_neg:]

		all_words = make_vocabs(pos_tt + neg_tt)

		array_fitur = ['presence', 'occurrence', 'tfidf']

		vectorizers = dict(zip(array_fitur, [
			binary_vectorizer,
			count_vectorizer,
			# tfidf_vectorizer,
			TfidfVectorizer(vocabulary=all_words)
		]))

		# X = {
		# 	fitur: vectorizers[fitur](pos_tt + neg_tt, all_words)
		# 	for fitur in array_fitur
		# }
		X = {}
		for fitur in array_fitur:
		    if fitur == 'tfidf':
		        vectorizers[fitur].fit(pos_tt + neg_tt)
		        X[fitur] = vectorizers[fitur].transform(
					pos_tt + neg_tt).toarray()
		    else:
		        X[fitur] = vectorizers[fitur](pos_tt + neg_tt, all_words)
		y = np.concatenate(
			[np.ones(len(pos_tt)), np.zeros(len(neg_tt))])

		# X_val = {
		# 	fitur: vectorizers[fitur](pos_val + neg_val, all_words)
		# 	for fitur in array_fitur
		# }
		X_val = {}
		for fitur in array_fitur:
		    if fitur == 'tfidf':
		        X_val[fitur] = vectorizers[fitur].transform(
					pos_val + neg_val).toarray()
		    else:
		        X_val[fitur] = vectorizers[fitur](
					pos_val + neg_val, all_words)
		y_val = np.concatenate(
			[np.ones(len(pos_val)), np.zeros(len(neg_val))])

		vectorizers = OrderedDict(sorted(vectorizers.items()))
		X = OrderedDict(sorted(X.items()))
		X_val =  OrderedDict(sorted(X_val.items()))

		with open(file_path, 'wb') as data:
			pickle.dump([X, y, X_val, y_val], data)

	return X, y, X_val, y_val


def get_vectorized_data_smote(pos_texts_normalized, neg_texts_normalized):
	

	app_path = os.getcwd()
	file_path = app_path+'/data/dinamics/texts_vector_smote.pkl'

	if isfile(file_path):
		print('reading pickle data...', file_path)
		with open(file_path, 'rb') as data:
			X_resampled, y_resampled = pickle.load(data)

	else:
		print('writing pickle data...', file_path)
		file_path_vektor = app_path+'/data/dinamics/texts_vector.pkl'

		if isfile(file_path_vektor):

			with open(file_path_vektor, 'rb') as data:
				X, y, _, _ = pickle.load(data)

				n_val_pos = 158
				n_val_neg = 142

				pos_tt = pos_texts_normalized[:-n_val_pos]
				neg_tt = neg_texts_normalized[:-n_val_neg]

				all_words = make_vocabs(pos_tt + neg_tt)

				array_fitur = ['presence', 'occurrence', 'tfidf']

				vectorizers = dict(zip(array_fitur, [
					binary_vectorizer,
					count_vectorizer,
					tfidf_vectorizer
				]))

				data_resampled = {
					fitur: smote(X[fitur], y, 200, k=3, random_seed=10)
					for fitur in array_fitur
				}
				X_resampled = {
					fitur: data_resampled[fitur][0]
					for fitur in array_fitur
				}
				y_resampled = data_resampled[array_fitur[0]][1]

				vectorizers = OrderedDict(sorted(vectorizers.items()))
				X_resampled =  OrderedDict(sorted(X_resampled.items()))

				with open(file_path, 'wb') as data:
					pickle.dump([X_resampled, y_resampled], data)
		else:
			return 404

	return X_resampled, y_resampled


def training_testing():

	clf = {
		'MNB': MultinomialNB(),
		# 'GNB': GaussianNB(),
		'SVM': LinearSVC(random_state=123),
		'LR': LogisticRegression(random_state=123),
	}

	path_vector_data = os.getcwd() + '/data/dinamics/'
	print('reading pickle data...')
	with open(path_vector_data+'texts_vector.pkl', 'rb') as model:
		X, y, X_val, y_val = pickle.load(model)
	print('reading pickle data...')
	with open(path_vector_data+'texts_vector_smote.pkl', 'rb') as model:
		X_resampled, y_resampled = pickle.load(model)

	filename = os.getcwd() + '/data/reports/'

	print('NO SAMPLING')
	best_models_no_sampling = do_training_testing(
		clf, X, y,
		filename+'kinerja_training_no_sampling.csv', show=False)
	performance_no_sampling = do_validation(
		clf, X_val, y_val, per_clf=best_models_no_sampling,
		filename=filename+'kinerja_testing_no_sampling.csv')

	print('\nRESAMPLED')
	best_models_after_sampling = do_training_testing(
		clf, X_resampled, y_resampled,
		filename+'kinerja_training_after_sampling.csv', show=False)
	performance_after_sampling = do_validation(
		clf, X_val, y_val, per_clf=best_models_after_sampling,
		filename=filename+'kinerja_testing_after_sampling.csv')

	# zip the report files for download
	zip_reports(filename)

	print(performance_no_sampling, performance_after_sampling)

	return performance_no_sampling, performance_after_sampling


def testing_live(texts_vector, best_fitur):
	file_path = os.getcwd()+'/data/dinamics/best_model.pkl'

	print('reading pickle data...', file_path)
	with open(file_path, 'rb') as data:
		best_model, _ = pickle.load(data)

	pred = best_model.predict(texts_vector)

	return pred, best_fitur


def get_vectorized_data_live():
	file_path = os.getcwd()+'/data/dinamics/vocabs.pkl'

	print('reading pickle data...', file_path)
	with open(file_path, 'rb') as data:
		all_words = pickle.load(data)

	file_path = os.getcwd()+\
		'/data/dinamics/data_traveloka_normalized_live.pkl'

	with open(file_path, 'rb') as data:
		texts_normalized = pickle.load(data)

	file_path = os.getcwd()+'/data/dinamics/best_model.pkl'

	with open(file_path, 'rb') as data:
		_, best_fitur = pickle.load(data)

	if best_fitur == 'presence':
		X = binary_vectorizer(texts_normalized, all_words)
	elif best_fitur == 'occurrence':
		X = count_vectorizer(texts_normalized, all_words)
	else:
		X = tfidf_vectorizer(texts_normalized, all_words)

	return X, best_fitur


def read_live_data():
	app_path = os.getcwd()
	file_path = app_path+'/data/dinamics/live_data.json'

	with open(file_path, 'r') as file:
		print('reading pickle data...', file_path)
		raw_data = json.load(file)

	return raw_data


def zip_reports(path):
	if os.path.exists(path+'reports.zip'):
		os.remove(path+'reports.zip')

	zipf = zipfile.ZipFile(path+'reports.zip', 'w', zipfile.ZIP_DEFLATED)

	for root, dirs, files in os.walk(path):
		for file in files:
			file = os.path.join(root, file)
			zipf.write(file, os.path.basename(file))

	print('zipping:', path+'reports.zip')
	zipf.close()
