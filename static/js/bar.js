function cetak_chart(data) {
	Chart.defaults.global.defaultFontSize = 14;

	console.log(data);

	var data_total_nb = {
		labels: ["presence", "occurrence", "tfidf"],
		datasets: [
			{
				label: "No sampling",
				backgroundColor: 'rgba(50, 50, 205, .2)',
				borderWidth: 1,
				data: [
					data['perf_no_sampl']['MNB']['perf']['presence']['gm_score'],
					data['perf_no_sampl']['MNB']['perf']['occurrence']['gm_score'],
					data['perf_no_sampl']['MNB']['perf']['tfidf']['gm_score'],
				],
				borderColor: 'rgba(0, 0, 255, 1)',
				// xAxisID: "first",
			},        
			{
				label: "With sampling",
				backgroundColor: 'rgba(205, 50, 50, .2)',
				borderWidth: 1,
				data: [
					data['perf_sampl']['MNB']['perf']['presence']['gm_score'],
					data['perf_sampl']['MNB']['perf']['occurrence']['gm_score'],
					data['perf_sampl']['MNB']['perf']['tfidf']['gm_score'],
				],
				borderColor: 'rgba(255, 0, 0, 1)',
				// xAxisID: "second",
			}
		]
	};

	var data_total_lr = {
		labels: ["presence", "occurrence", "tfidf"],
		datasets: [
			{
				label: "No sampling",
				backgroundColor: 'rgba(50, 50, 205, .2)',
				borderWidth: 1,
				data: [
					data['perf_no_sampl']['LR']['perf']['presence']['gm_score'],
					data['perf_no_sampl']['LR']['perf']['occurrence']['gm_score'],
					data['perf_no_sampl']['LR']['perf']['tfidf']['gm_score'],
				],
				borderColor: 'rgba(0, 0, 255, 1)',
				// xAxisID: "first",
			},        
			{
				label: "With sampling",
				backgroundColor: 'rgba(205, 50, 50, .2)',
				borderWidth: 1,
				data: [
					data['perf_sampl']['LR']['perf']['presence']['gm_score'],
					data['perf_sampl']['LR']['perf']['occurrence']['gm_score'],
					data['perf_sampl']['LR']['perf']['tfidf']['gm_score'],
				],
				borderColor: 'rgba(255, 0, 0, 1)',
				// xAxisID: "second",
			}
		]
	};

	var data_total_svm = {
		labels: ["presence", "occurrence", "tfidf"],
		datasets: [
			{
				label: "No sampling",
				backgroundColor: 'rgba(50, 50, 205, .2)',
				borderWidth: 1,
				data: [
					data['perf_no_sampl']['SVM']['perf']['presence']['gm_score'],
					data['perf_no_sampl']['SVM']['perf']['occurrence']['gm_score'],
					data['perf_no_sampl']['SVM']['perf']['tfidf']['gm_score'],
				],
				borderColor: 'rgba(0, 0, 255, 1)',
				// xAxisID: "first",
			},        
			{
				label: "With sampling",
				backgroundColor: 'rgba(205, 50, 50, .2)',
				borderWidth: 1,
				data: [
					data['perf_sampl']['SVM']['perf']['presence']['gm_score'],
					data['perf_sampl']['SVM']['perf']['occurrence']['gm_score'],
					data['perf_sampl']['SVM']['perf']['tfidf']['gm_score'],
				],
				borderColor: 'rgba(255, 0, 0, 1)',
				// xAxisID: "second",
			}
		]
	};

	var barValue = {
		onComplete: function () {
			var chartInstance = this.chart,
				ctx = chartInstance.ctx;
			ctx.font = Chart.helpers.fontString(
				Chart.defaults.global.defaultFontSize,
				Chart.defaults.global.defaultFontStyle,
				Chart.defaults.global.defaultFontFamily
			);
			ctx.textAlign = 'center';
			ctx.textBaseline = 'bottom';

			this.data.datasets.forEach(function (dataset, i) {
				var meta = chartInstance.controller.getDatasetMeta(i);
				meta.data.forEach(function (bar, index) {
					var data = dataset.data[index];                            
					ctx.fillText(data, bar._model.x, bar._model.y - 5);
				});
			});
		}
	}

	var barScale = {
		xAxes: [{
			// stacked: true,
			// id: "first",
			// barThickness : 70,
		}],
		yAxes: [{
			stacked: false,
			ticks: {
				min: 0,
				max: 100,
				stepSize: 20,
			},
		}]

	}

	var options_nb = {
		title: {
	        display: true,
	        text: 'Naive Bayes'
	    },
		scales: barScale,
		events: false,
		tooltips: {
			enabled: false
		},
		hover: {
			animationDuration: 0
		},
		animation: barValue,
	};

	var options_lr = {
		title: {
	        display: true,
	        text: 'Logistic Regression'
	    },
		scales: barScale,
		events: false,
		tooltips: {
			enabled: false
		},
		hover: {
			animationDuration: 0
		},
		animation: barValue,
	};

	var options_svm = {
		title: {
	        display: true,
	        text: 'Support Vector Machine'
	    },
		scales: barScale,
		events: false,
		tooltips: {
			enabled: false
		},
		hover: {
			animationDuration: 0
		},
		animation: barValue,
	};

	var ctx_nb = document.getElementById("nb-aw").getContext("2d");
	var ctx_lr = document.getElementById("log-reg-aw").getContext("2d");
	var ctx_svm = document.getElementById("svm-aw").getContext("2d");
	var myBarChart = new Chart(ctx_nb, {
		type: 'bar',
		data: data_total_nb,
		options: options_nb,
	});
	var myLineChart = new Chart(ctx_lr, {
		type: 'bar',
		data: data_total_lr,
		options: options_lr,
	});
	var myLineChart = new Chart(ctx_svm, {
		type: 'bar',
		data: data_total_svm,
		options: options_svm,
	});

}