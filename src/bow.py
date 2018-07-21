"""
Module berisi fungsi untuk melakukan-
pembuatan vocabulary/bow
"""

import os
import pickle
from nltk import FreqDist



def make_vocabs(normalized_data, is_pickle=True):
	"""
	Fungsi untuk melakukan pembuatan bow/vocabulary

	Proses pembuatan vocabs

	vocabs ini digunakan untuk membentuk feature vector dari normalized data
	beberapa perlakukan untuk membentuk vocabs, di antarnya:
	(1) hapus hapax: kata yang hanya muncul sekali dari seluruh corpus
	(2) seleksi hanya kata kerja
	(3) hapus hapax dan gunakan hanya kata dengan panjang > 2 karakter

	return:
	all_words = vocabs/bow hasil

	paramater:
	normalized_data = data text yang sudah dilakukan preprocessing/normalisasi
	"""

	all_words = [word for sentence in normalized_data
				 for word in sentence.split()]

	fd = FreqDist(all_words) # sebelum di-set, bentuk object freqdist

	all_words = list(sorted(set(all_words)))
	print('n fitur awal:\t\t', len(all_words))

	# (1)
	hapaxes = fd.hapaxes()
	# all_words = [word for word in all_words if word not in hapaxes]

	# (2)
	# with open('../experiment/pos_tag_indo.pkl', 'rb') as file:
	#     jj = pickle.load(file)
	# all_words_adj = [word for word in all_words if word in jj]
	# all_words = all_words_adj

	all_words = [word for word in all_words
	             if len(word) > 2
	             and word not in hapaxes]

	file_path = os.getcwd()+'/data/dinamics/vocabs.pkl'

	if is_pickle:
		with open(file_path, 'wb') as data:
			pickle.dump(all_words, data)

	return all_words