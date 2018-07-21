"""
module berisi fungsi kfold
diguakan untuk melakukan k-fold cross validation
mengembalikan index-index data training dan testing
"""

import math
import numpy as np
import random



def kfold(Y, n_splits, random_state=0):
	"""
	fungsi untuk melakukan k-fold cross validation

	parameter:
	Y = array label/kelas/kategori
	n_splits = nilai k fold yang akan digunakan
	random_state = seed untuk melakukan randomisasi

	return:
	indices untuk training dan testing dalam bentuk rolling
	"""
	
	n_data = len(Y)
	porsi = math.floor(n_data / n_splits)
	n_train = n_data - porsi
	indices = np.arange(0, n_data)
	_indices = indices.tolist()

	test_indices_all = np.zeros([n_splits, porsi], dtype=int)
	train_indices_all = np.zeros([n_splits, n_train], dtype=int)

	for i in range(n_splits):
		random.seed(random_state)
		test_indices_chosen = random.sample(_indices, porsi)
		test_indices_chosen.sort()
		
		_indices = [index for index in _indices
					if index not in test_indices_chosen]

		test_indices = indices[test_indices_chosen]
		train_indices = np.delete(indices, test_indices)

		train_indices_all[i] = train_indices
		test_indices_all[i] = test_indices

	return train_indices_all, test_indices_all
