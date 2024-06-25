// static/js/statistics.js

document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('ratingsChart').getContext('2d');
    var categories = JSON.parse(document.getElementById('categories-data').textContent.replace(/^"|"$/g, '').replace(/'/g, '"'));
    var avgRatings = JSON.parse(document.getElementById('avg-ratings-data').textContent.replace(/^"|"$/g, '').replace(/'/g, '"'));
    let delayed;

    var ratingsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: categories,
            datasets: [{
                label: 'Average Rating',
                data: avgRatings,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                type: 'bar'
            }, 
            {
                label: 'Average Rating',
                data: avgRatings,
                backgroundColor: 'rgba(232, 178, 184, 0.2)',
                borderColor: 'rgba(232, 178, 184, 1)',
                type: 'line',
            }]
        },
        options: {
            animation: {
                onComplete: () => {
                  delayed = true;
                },
                delay: (context) => {
                  let delay = 0;
                  if (context.type === 'data' && context.mode === 'default' && !delayed) {
                    delay = context.dataIndex * 300 + context.datasetIndex * 100;
                  }
                  return delay;
                },
              },
            plugins: {
                title: {
                  display: true,
                  text: 'Average Rating',
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    stacked: true,
                    max: 5,
                }
            },
        }
    });

    var ctx2 = document.getElementById('seasonChart').getContext('2d');
    var seasons = ["Winter", "Spring", "Summer", "Autumn"];
    var seasonCounts = JSON.parse(document.getElementById('season-counts-data').textContent.replace(/^"|"$/g, '').replace(/'/g, '"'));
    var data = {
        labels: seasons,
        datasets: [{
            data: seasonCounts,
            backgroundColor: ['#36A2EB',  '#4BC0C0', '#FF6384', '#FFCE56'],
            hoverBackgroundColor: ['#36A2EB', '#4BC0C0', '#FF6384',  '#FFCE56']
        }]
    };

    var options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
              display: true,
              text: 'General Savings Statistics',
            }
        },
    };

    new Chart(ctx2, {
        type: 'pie',
        data: data,
        options: options
    });

    datas = JSON.parse(document.getElementById('season-category-counts-data').textContent.replace(/^"|"$/g, '').replace(/'/g, '"'));

    Object.keys(datas).forEach(season => {
        var ctx3 = document.getElementById(`${season}CategoryChart`).getContext('2d');
        var seasonData = datas[season];

        var data = {
            labels: categories.map(category => category),
            datasets: [{
                data: categories.map(category => seasonData[category] || 0),
            }]
        };

        var options = {
            responsive: true,
            plugins: {
                title: {
                  display: true,
                  text: `${season.charAt(0).toUpperCase() + season.slice(1)} Statistics`,
                }
            },
        };

        new Chart(ctx3, {
            type: 'doughnut',
            data: data,
            options: options
        });
    });
});
