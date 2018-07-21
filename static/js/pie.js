function cetak_pie(n_pos, n_neg, table) {
  var data = {
    datasets: [{
      data: [n_pos, n_neg],
      backgroundColor: [
        "rgba(70, 200, 100, .5)",
        "rgba(200, 70, 70, .5)"
      ],
      borderColor: [
        'rgba(70, 200, 100, 1)',
        'rgba(200, 70, 70, 1)'
      ]
    }],
    labels: [
      "Positif",
      "Negatif"
    ]
  };

  var canvas = document.getElementById("pie-chart");
  var ctx = canvas.getContext("2d");
  var myNewChart = new Chart(ctx, {
    type: 'pie',
    data: data,
    options: {
      hover: {
        onHover: function(e, el) {
          $("#pie-chart").css("cursor", el[0] ? "pointer" : "default");
        }
      }
    }
  });

  canvas.onclick = function (evt) {
    var activePoints = myNewChart.getElementsAtEvent(evt);

    if (activePoints[0]) {
      var chartData = activePoints[0]['_chart'].config.data;
      var idx = activePoints[0]['_index'];
      var label = chartData.labels[idx];
      var search = label == 'Positif' ? 'positif' : 'negatif';

      console.log('search: ' + search);

      table.column(1).search(search).draw();
    }
  };
}
