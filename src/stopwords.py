import os

def get_stopwords():
	app_path = os.getcwd()
	file_path = app_path+'/data/stopwords.txt'
	
	with open(file_path, 'r') as file:
		stopwords = file.read().split('\n')

	return stopwords