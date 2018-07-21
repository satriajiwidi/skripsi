import json, os, pickle
import src.modules as modules
import src.preprocess as preprocess
from src.komentar import crawl_data
from flask import request



def get_komentar_preprocessed():
	raw_data = modules.get_raw_data()
	pos_texts_normalized, neg_texts_normalized = \
		preprocess.get_normalized_data(raw_data)
	label_pos = ['positif' for i in range(len(pos_texts_normalized))]
	label_neg = ['negatif' for i in range(len(neg_texts_normalized))]

	zipped_pos = []
	for i in range(len(label_pos)):
		record = {
			'text': pos_texts_normalized[i],
			'kelas': label_pos[i],
		}
		zipped_pos.append(record)
	zipped_neg = []
	for i in range(len(label_neg)):
		record = {
			'text': neg_texts_normalized[i],
			'kelas': label_neg[i],
		}
		zipped_pos.append(record)

	data = {
		'draw': 0,
		'recordsTotal': len(label_pos) + len(label_neg),
		'recordsFiltered': len(label_pos) + len(label_neg),
		'data': zipped_pos + zipped_neg,
		'input': []
	}

	return json.dumps(data)


def get_data_teruji_live_dt():
	X, best_fitur = modules.get_vectorized_data_live()
	pred, _ = modules.testing_live(X, best_fitur)
	pred = ['positif' if i == 1 else 'negatif' for i in pred]
	raw_data = modules.read_live_data()

	raw_data = [text['text'] for text in raw_data]
	counts = len(pred)

	zipped = list()
	for i in range(counts):
		record = {
			'kelas': pred[i],
			'text': raw_data[i],
		}
		zipped.append(record)

	data = {
		'draw': 0,
		'recordsTotal': counts,
		'recordsFiltered': counts,
		'data': zipped,
		'input': []
	}

	return json.dumps(data)


def get_performance_metrics_uji():
	X, best_fitur = modules.get_vectorized_data_live()
	pred, best_fitur = modules.testing_live(X, best_fitur)
	pred = ['positif' if i == 1 else 'negatif' for i in pred]
	n_pos, n_neg = pred.count('positif'), pred.count('negatif')

	data = {
		'pred': pred,
		'best_fitur': best_fitur,
		'n_pos': n_pos,
		'n_neg': n_neg,
	}

	return json.dumps(data)


def get_raw_data_live():
	raw_data = modules.read_live_data()
	texts_normalized = preprocess.get_normalize_live_data(raw_data)

	data = {
		'draw': 0,
		'recordsTotal': len(raw_data),
		'recordsFiltered': len(raw_data),
		'data': raw_data,
		'input': []
	}
	return json.dumps(data)


def get_komentar_preprocessed_live():
	raw_data = modules.read_live_data()
	texts_normalized = preprocess.get_normalize_live_data(raw_data)
	
	data = []
	for text in texts_normalized:
		data.append({'text': text})

	data = {
		'draw': 0,
		'recordsTotal': len(texts_normalized),
		'recordsFiltered': len(texts_normalized),
		'data': data,
		'input': []
	}
	return json.dumps(data)


def get_performance_metrics():
	performance_no_sampling, performance_after_sampling = \
		modules.training_testing()

	data = {
		'perf_no_sampl': performance_no_sampling,
		'perf_sampl': performance_after_sampling,
	}

	return json.dumps(data)


def get_vectorized_data():
	raw_data = modules.get_raw_data()
	pos_texts_normalized, neg_texts_normalized = \
		preprocess.get_normalized_data(raw_data)
	X, Y, X_val, Y_val = modules.get_vectorized_data(
		pos_texts_normalized, neg_texts_normalized)

	vektor_total = []
	for fitur in X:
		vektor_baris = []
		for baris in range(10):
			vektor_kolom = []
			for kolom in X[fitur][baris][:10]:
				vektor_kolom.append(float(round(kolom, 3)))
			vektor_baris.append(vektor_kolom)
		vektor_total.append({fitur: vektor_baris})

	X_json_sliced = json.dumps(vektor_total)

	return X_json_sliced


def get_porsi_data_after_smote():
	raw_data = modules.get_raw_data()
	pos_texts_normalized, neg_texts_normalized = \
		preprocess.get_normalized_data(raw_data)
	_, Y = modules.get_vectorized_data_smote(
		pos_texts_normalized, neg_texts_normalized)
	n_pos = len([i for i in Y if i == 1])
	n_neg = len([i for i in Y if i == 0])

	data = {
		'n_pos': n_pos,
		'n_neg': n_neg,
	}

	return json.dumps(data)


def get_raw_data_json():
	number = request.args.get('number', 10)
	raw_data = modules.get_raw_data()[:number]
	data = {
		'draw': 0,
		'recordsTotal': len(raw_data),
		'recordsFiltered': len(raw_data),
		'data': raw_data,
		'input': []
	}
	return json.dumps(data)


def get_selected_best_model():
	file_path = os.getcwd()+'/data/dinamics/best_model.pkl'

	print('reading pickle data...', file_path)

	with open(file_path, 'rb') as data:
		best_model, best_fitur = pickle.load(data)

	print(best_model, best_fitur)

	best_model = (str(best_model)).split('(')[0]

	data = {
		'best_model': best_model,
		'best_fitur': best_fitur
	}
	
	return json.dumps(data)