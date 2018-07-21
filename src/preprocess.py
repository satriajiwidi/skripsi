"""
Module untuk melakukan preprocessing terhadap data text
"""

import pickle
import os
import re
from os.path import isfile
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from .stopwords import get_stopwords

from .bow import make_vocabs
from pyjarowinkler import distance

from joblib import Parallel, delayed
import multiprocessing



def parallel_jw(word, kata):
    return distance.get_jaro_distance(
        word, kata, winkler=True, scaling=0.1)


def get_kamus():
    file_path = os.getcwd()+'/data/indonesian_word_list.txt'
    with open(file_path, 'r') as file:
        kamus = file.read()
    kamus = [kata for kata in kamus.split('\n')]

    kamus = [k.lower() for k in kamus if len(k) > 2
            and ' ' not in k
            and '.' not in k
            and '-' not in k]

    return kamus


def correction(all_words, kamus):
    from time import time

    num_cores = multiprocessing.cpu_count()
    kamus_hasil = {}

    start = time()
    kamus_hasil = {word: word for word in all_words if word in kamus}
    
    new_all_words = [word for word in all_words if word not in kamus]
    if len(new_all_words) > 1:
        print('Banyak kata yang dinormalisasi: {}. {} - {}'
            .format(len(new_all_words), new_all_words[0], new_all_words[-1]))
    else:
        print('Tidak ada kata baru untuk dinormalisasi')

    for index, word in enumerate(new_all_words):
        results = Parallel(n_jobs=num_cores)(
            delayed(parallel_jw)(word, kata) for kata in kamus)
        best_index = results.index(max(results))
        new_hasil = kamus[best_index]
        kamus_hasil[word] = new_hasil
        print('{}. normalisasi: {} -> {}'
            .format(index+1, word, new_hasil))

    end = time()
    print('normalisasi selesai dalam {} detik -> {} menit'
        .format(round(end-start, 2), round((end-start)/60, 2)))

    return kamus_hasil


def get_kamus_hasil():
    kamus_hasil = {}

    if isfile('kamus_hasil.pkl'):
        with open('kamus_hasil.pkl', 'rb') as data:
            kamus_hasil = pickle.load(data)

    return kamus_hasil


def get_normalized_data(raw_data):
    """
    fungsi untuk melakukan preprocessing
    dari raw data text menjadi data text ternormalisasi

    Preprocessing data
    1. pisahkan text positif dan negatif ke dalam-
        variabel array masing-masing
    2. untuk masing-masing text positif maupun negatif, lakukan:
        3. lowercase tiap kata
        4. buang kata yg bukan huruf [a-z]
        5. hapus stopwords
        6. stem tiap kata
    7. simpan sebagai pickle file

    return:
    pos_texts_normalized, neg_texts_normalized = hasil normalisasi-
        dari masing-masing kelas

    parameter:
    raw_data = raw text data file
    """

    app_path = os.getcwd()
    file_path = app_path+'/data/dinamics/data_traveloka_normalized.pkl'

    if isfile(file_path) and isfile(app_path+'/data/dinamics/kamus_hasil.pkl'):
        print('reading pickle data...', file_path)
        with open(file_path, 'rb') as data:
            pos_texts_normalized, neg_texts_normalized = \
                pickle.load(data)
        with open(app_path+'/data/dinamics/kamus_hasil.pkl', 'rb') as data:
            kamus_hasil = pickle.load(data)
    
    else:
        print('writing pickle data...', file_path)
        pos_texts = [d['text'] for d in raw_data if d['class'] == 1]
        neg_texts = [d['text'] for d in raw_data if d['class'] == 0]

        kamus_hasil = {}

        pos_texts_normalized, neg_texts_normalized = \
            normalisasi1(pos_texts, neg_texts)

        print('normalisasi (tokenisasi) 1 selesai')

        all_words = get_all_words(
            pos_texts_normalized + neg_texts_normalized)

        kamus = get_kamus()

        kamus_hasil = correction(all_words, kamus)

        print('proses koreksi selesai')

        pos_texts = pos_texts_normalized
        neg_texts = neg_texts_normalized

        pos_texts_normalized, neg_texts_normalized = \
            normalisasi2(pos_texts, neg_texts, kamus_hasil)

        print('proses normalisasi 2 (stopwords removal and stemming) selesai')
        

        with open(file_path, 'wb') as data:
            pickle.dump([pos_texts_normalized, neg_texts_normalized], data)

        with open(app_path+'/data/dinamics/kamus_hasil.pkl', 'wb') as data:
            pickle.dump(kamus_hasil, data)


    return pos_texts_normalized, neg_texts_normalized


def normalisasi1(pos_texts, neg_texts):
    pos_texts_normalized = []

    for text in pos_texts:
        pos_text_normalized = []

        for word in text.split():
            word = word.lower()
            word = re.match('[a-z]+', word)

            if word is not None:
                word = word.group(0)

                pos_text_normalized.append(word)
        
        pos_texts_normalized.append(' '.join(pos_text_normalized))
    

    neg_texts_normalized = []

    for text in neg_texts:
        neg_text_normalized = []

        for word in text.split():
            word = word.lower()
            word = re.match('[a-z]+', word)

            if word is not None:
                word = word.group(0)

                neg_text_normalized.append(word)
        
        neg_texts_normalized.append(' '.join(neg_text_normalized))

    return pos_texts_normalized, neg_texts_normalized


def normalisasi2(pos_texts, neg_texts, kamus_hasil):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords = get_stopwords()

    pos_texts_normalized = []

    for text in pos_texts:
        pos_text_normalized = []

        for word in text.split():
            # normalisasi
            word = kamus_hasil[word]
            if word not in stopwords:
                word = stemmer.stem(word)
                if word not in stopwords:
                    pos_text_normalized.append(word)
        
        pos_texts_normalized.append(' '.join(pos_text_normalized))
    

    neg_texts_normalized = []

    for text in neg_texts:
        neg_text_normalized = []

        for word in text.split():
            # normalisasi
            word = kamus_hasil[word]
            if word not in stopwords:
                word = stemmer.stem(word)
                if word not in stopwords:
                    neg_text_normalized.append(word)
        
        neg_texts_normalized.append(' '.join(neg_text_normalized))

    return pos_texts_normalized, neg_texts_normalized


def get_all_words(texts):
    all_words = [word for sentence in texts
                for word in sentence.split()]
    all_words = list(sorted(set(all_words)))

    return all_words


def get_normalize_live_data(raw_data):
    app_path = os.getcwd()
    
    texts = [d['text'] for d in raw_data]

    texts_normalized1, texts_normalized2 = normalisasi1(
        texts[:-1], texts[-1:])
    texts_normalized = texts_normalized1 + texts_normalized2

    # normalisasi pertama
    print('normalisasi 1 selesai')

    all_words = get_all_words(texts_normalized)

    kamus = get_kamus()

    kamus_hasil = {}

    file_path = app_path+'/data/dinamics/kamus_hasil.pkl'
    if isfile(file_path):
        with open(file_path, 'rb') as data:
            kamus_hasil = pickle.load(data)

    # concat, update kamus hasil
    all_words = [word for word in all_words if word not in kamus_hasil]
    kamus_hasil = {**correction(all_words, kamus), **kamus_hasil}

    texts = texts_normalized

    texts_normalized1, texts_normalized2 = normalisasi2(
        texts[:-1], texts[-1:], kamus_hasil)
    texts_normalized = texts_normalized1 + texts_normalized2

    file_path = app_path+'/data/dinamics/data_traveloka_normalized_live.pkl'
    with open(file_path, 'wb') as data:
        print('writing pickle data...', file_path)    
        pickle.dump(texts_normalized, data)

    file_path = app_path+'/data/dinamics/kamus_hasil.pkl'
    with open(file_path, 'wb') as data:
        pickle.dump(kamus_hasil, data)

    return texts_normalized
