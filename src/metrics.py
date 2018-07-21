"""
module berisi tiga fungsi-
untuk melakukan perhitungan performa / performance metric

1. confusion matrix
2. geometric mean score
3. accuracy score
"""

import numpy as np
import math



def confusion_matrix(y_true, y_pred, table_show=True):
	"""
	fungsi untuk menghitung confusion matrix hasil klasifikasi

	parameter:
	y_true = label kelas asli / ground truth
	y_pred = label kelas hasil prediksi
	table_show = boolean, opsi untuk menampilkan hasil-
		dalam tabel array atau tidak

	return:
	array confusion matrix
		tp = true positives
		tn = true negatives
		fp = false positives
		fn = false negatives
	"""
	FIRST_CLASS = 1
	SECOND_CLASS = 0

	zipped = np.array(list(zip(y_true, y_pred)))
	tp, fn, fp, tn = 0, 0, 0, 0

	for y_true, y_pred in zipped:
		if y_true == y_pred and y_true == FIRST_CLASS:
			tp += 1
		elif y_true == y_pred and y_true == SECOND_CLASS:
			tn += 1
		elif y_true != y_pred and y_true == SECOND_CLASS:
			fp += 1
		else:
			fn += 1

	if table_show:
		return np.array([tp, fn, fp, tn]).reshape([2,2])

	return tp, fn, fp, tn


def geometric_mean_score(y_true, y_pred):
	"""
	fungsi untuk menghitung geometric mean hasil klasifikasi

	parameter:
	y_true = label kelas asli / ground truth
	y_pred = label kelas hasil prediksi

	dependency:
	menggunakan fungsi confusion_matrix
	untuk mendapatkan:
	array confusion matrix
		tp = true positives
		tn = true negatives
		fp = false positives
		fn = false negatives

	return:
	skor geometric mean
	"""
	tp, fn, fp, tn = confusion_matrix(y_true, y_pred, table_show=False)

	return math.sqrt((tn / (tn+fp)) * (tp / (tp+fn)))


def accuracy_score(y_true, y_pred):
	"""
	fungsi untuk menghitung accuracy hasil klasifikasi

	parameter:
	y_true = label kelas asli / ground truth
	y_pred = label kelas hasil prediksi

	dependency:
	menggunakan fungsi confusion_matrix
	untuk mendapatkan:
	array confusion matrix
		tp = true positives
		tn = true negatives
		fp = false positives
		fn = false negatives

	return:
	skor accuracy
	"""
	tp, fn, fp, tn = confusion_matrix(y_true, y_pred, table_show=False)

	return (tp+tn) / (tp+tn+fn+fp)
