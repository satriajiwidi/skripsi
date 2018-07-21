from flask import Flask
import src.apis as apis
import src.controllers as controllers



"""
DEFINE APP
"""
app = Flask(__name__)

app.config["CACHE_TYPE"] = "null"

app.config.update(
    DEBUG=True,
    TEMPLATES_AUTO_RELOAD=True,
)


"""
WITH CONTROLLERS
"""
@app.route('/')
def index():
	return controllers.index()

@app.route('/uji')
def uji():
	return controllers.uji()

@app.route('/crawl')
def crawl():
	return controllers.crawl()

@app.route('/download-report')
def download_report():
	return controllers.download_report()


"""
API COLLECTIONS
"""
@app.route('/api/smote')
def get_porsi_data_after_smote():
	return apis.get_porsi_data_after_smote()

@app.route('/api/vektor')
def get_vectorized_data():
	return apis.get_vectorized_data()

@app.route('/api/training')
def get_performance_metrics():
	return apis.get_performance_metrics()

@app.route('/api/raw_data_live')
def get_raw_data_live():
	return apis.get_raw_data_live()

@app.route('/api/uji-live')
def get_performance_metrics_uji():
	return apis.get_performance_metrics_uji()

@app.route('/api/komentar-preprocessed')
def get_komentar_preprocessed():
	return apis.get_komentar_preprocessed()

@app.route('/api/komentar-preprocessed-live')
def get_komentar_preprocessed_live():
	return apis.get_komentar_preprocessed_live()

@app.route('/api/uji-live/dt')
def get_data_teruji_live_dt():
	return apis.get_data_teruji_live_dt()

@app.route('/api/raw-data')
def get_raw_data_json():
	return apis.get_raw_data_json()

@app.route('/api/best-model')
def get_selected_best_model():
	return apis.get_selected_best_model()


"""
MAIN EXCECUTION
"""
if __name__ == '__main__':
	app.run(debug=True)