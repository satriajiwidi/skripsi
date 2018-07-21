import os
from flask import Flask, render_template, request, send_file
import src.modules as modules
import src.preprocess as preprocess
from src.komentar import crawl_data



def index():
	raw_data = modules.get_raw_data()
	n_pos, n_neg = modules.get_data_rasio_info(raw_data)
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
				vektor_kolom.append(float(round(kolom, 5)))
			vektor_baris.append(vektor_kolom)
		vektor_total.append({fitur: vektor_baris})
	
	return render_template('index-1.html',
		raw_data=raw_data[:10],
		porsi_data=[n_pos, n_neg],
		vektor=vektor_total,
		flag='training')


def uji():
	raw_data = modules.get_raw_data()
	n_pos, n_neg = modules.get_data_rasio_info(raw_data)
	pos_texts_normalized, neg_texts_normalized = \
		preprocess.get_normalized_data(raw_data)
	
	return render_template('index-2.html',
		raw_data=None,
		flag='uji')


def crawl():
	url = request.args.get('url', None)

	crawl_data(url)

	return '200'


def download_report():
	app_path = os.getcwd()
	filename = app_path+'/data/reports/reports.zip'

	try:
		print('downloading:', filename)
		return send_file(filename, attachment_filename='reports.zip')
	except Exception as e:
		return str(e)
