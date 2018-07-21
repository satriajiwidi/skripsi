"""
Module vectorizers, berisi beberapa fungsi vectorizer-
untuk melakukan vektorisasi fitur menjadi array vektor

1. binary vectorizer
2. count vectorizer
3. tfidf vectorizer
"""

import math
import numpy as np


def binary_vectorizer(training_texts, all_words):
    """
    fungsi untuk melakukan vectorisasi menjadi array fitur biner

    return array fitur biner: word vectors

    parameter:
    training_text = data training berupa teks
    all_words = bow/vocabs untuk membentuk array vector-
        dengan dimensi yang sama
    """

    word_vectors = []

    for sentence in training_texts:
        sentence_words = [word for word in sentence.split()]
        sentence_words = list(sorted(set(sentence_words)))
        sentence_features = []

        for word in all_words:
            sentence_features.append(1 if word in sentence_words else 0)
        word_vectors.append(sentence_features)

    word_vectors = np.array(word_vectors)

    return word_vectors


def count_vectorizer(training_texts, all_words):
    """
    fungsi untuk melakukan vectorisasi menjadi-
        array fitur count occurence/frekuensi

    return array fitur count occurence: word vectors

    parameter:
    training_text = data training berupa teks
    all_words = bow/vocabs untuk membentuk-
        array vector dengan dimensi yang sama
    """

    word_vectors = []

    for sentence in training_texts:

        sentence_words = [word for word in sentence.split()]
        sentence_words = list(sorted(sentence_words))
        sentence_features = []

        for word in all_words:
            sentence_features.append(sentence_words.count(word))
        word_vectors.append(sentence_features)

    word_vectors = np.array(word_vectors)

    return word_vectors


def tfidf_vectorizer(training_texts, all_words):
    """
    fungsi untuk melakukan vectorisasi menjadi-
        array fitur tfidf

    return array fitur tfidf: word vectors

    parameter:
    training_text = data training berupa teks
    all_words = bow/vocabs untuk membentuk-
        array vector dengan dimensi yang sama
    """

    # tf = n kemunculan term t di doc / n kata di doc
    # idf = log ( n doc / n doc di mana term t muncul )


    word_vectors = []
    
    n_doc = len(training_texts)
    n_doc_term_occurs_dicts = {}
    
    # hitung n kemunculan setiap kata k di seluruh corpus (df)
    for word in all_words:
        n_doc_term_occurs = 1 # smoothing

        for sentence in training_texts:
            sentence_words = [word for word in sentence.split()]
            sentence_words = list(sorted(sentence_words))

            if word in sentence_words:
                n_doc_term_occurs += 1

        n_doc_term_occurs_dicts[word] = n_doc_term_occurs
    
    # lakukan perhitungan tf-idf
    for sentence in training_texts:
        sentence_words = [word for word in sentence.split()]
        sentence_words = list(sorted(sentence_words))
        sentence_features = []
        n_kata = len(sentence_words)

        for word in all_words:
            tf = sentence_words.count(word) / n_kata
            idf = math.log10(n_doc / n_doc_term_occurs_dicts[word])
            sentence_features.append(tf * idf)

        word_vectors.append(sentence_features)

    word_vectors = np.array(word_vectors)

    return word_vectors