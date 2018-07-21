"""
Module untuk melakukan oversampling
dengan menggunakan algoritma SMOTE
Synthetic Minority Oversampling Technique
"""

import random
import numpy as np
from sklearn.neighbors import NearestNeighbors



def SMOTE(X, y, N, k=5, random_seed=0):
	"""
	fungsi untuk melakukan oversampling
	dengan menggunakan algoritma SMOTE
	(Synthetic Minority Oversampling Technique)
	
	return X_res, y_res: data (X, y) hasil oversampling
	
	parameter:
	X = data yang akan di-oversampling
	y = label data, terbatas untuk klasifikasi biner, dengan nilai 0 dan 1
	N = berapa persen oversampling yang akan dilakukan,
		asumsi kelipatan 100 persen
	k = k neirest neighbor yang akan digunakan pada smote (default=5)
	random_seed = seed untuk generate random
	"""

	X, y = np.array(X), np.array(y)

	# tidak dilakukan oversampling
	if N < 100:
		return X, y


	data_bundled = list(zip(X, y))
	X_kelas_pertama = np.array([x for x, y in data_bundled if y == 1])
	X_kelas_kedua = np.array([x for x, y in data_bundled if y == 0])

	n_kelas_pertama = len(X_kelas_pertama)
	n_kelas_kedua = len(X_kelas_kedua)

	if n_kelas_pertama < n_kelas_kedua:
		label_minor = 1
		X_minor = X_kelas_pertama
		n_minor = n_kelas_pertama
		X_mayor = X_kelas_kedua
		n_mayor = n_kelas_kedua
	else:
		label_minor = 0
		X_minor = X_kelas_kedua
		n_minor = n_kelas_kedua
		X_mayor = X_kelas_pertama
		n_mayor = n_kelas_pertama


	"""
	mendefinisikan beberapa variabel

	variabel:
	N = berapa kali dari banyak n data minor yang akan dihasilkan
	n_attrs = banyak atribut/dimensi dari data
	n_generated = counter berapa banyak data yang sudah d-generate
	X_generated = X data hasil oversampling
	"""
	N = int(N/100)
	n_attrs = len(X[0])
	n_generated = 0
	X_generated = np.zeros(shape=(N*n_minor, n_attrs))


	# get nearest neighbors dari masing-masing data X kelas minor
	# kemudian simpan k index-indexnya
	neighbors = NearestNeighbors(n_neighbors=k).fit(X_minor) \
		.kneighbors(n_neighbors=k, return_distance=False)


	# proses generasi data sintetis
	for i in range(n_minor):
		nn_array = neighbors[i]
		_N = N

		while _N != 0:
			random.seed(random_seed+i+N)
			nn = random.randint(0, k-1)

			for attr in range(n_attrs):
				distance = X_minor[nn_array[nn]][attr] - X_minor[i][attr]
				random.seed(random_seed+i+N+attr)
				gap = random.random()
				X_generated[n_generated][attr] = \
					X_minor[i][attr] + gap*distance
			
			n_generated += 1
			_N -= 1


	X_res = np.concatenate(
		[X_mayor, X_minor, X_generated], axis=0)
	y_res = np.concatenate(
		[y, [label_minor for i in range(n_generated)]], axis=0)
	

	# mengembalikan data hasil oversampling
	return X_res, y_res