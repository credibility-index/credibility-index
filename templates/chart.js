// Глобальная переменная для хранения экземпляра графика
let chartInstance = null;

// Функция для отображения графика с использованием Chart.js
function renderChart(data) {
    const ctx = document.getElementById('chart').getContext('2d');

    // Преобразуем данные для Chart.js
    const backgroundColors = data.scores.map(score => {
        if (score > 0.8) return 'rgba(40, 167, 69, 0.7)';
        if (score > 0.6) return 'rgba(255, 193, 7, 0.7)';
        return 'rgba(220, 53, 69, 0.7)';
    });

    // Если график уже существует, уничтожаем его перед созданием нового
    if (chartInstance) {
        chartInstance.destroy();
    }

    // Создаем новый график
    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.sources,
            datasets: [{
                label: 'Credibility Score',
                data: data.scores,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Credibility Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'News Sources'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2);
                        }
                    }
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

// Функция для обновления графика
function updateChart() {
    const container = document.querySelector('.chart-container');
    const loadingHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Updating chart data...</p>
        </div>
    `;

    // Показываем индикатор загрузки
    container.insertAdjacentHTML('afterbegin', loadingHTML);

    // Имитация задержки сети
    setTimeout(() => {
        // Удаляем индикатор загрузки
        const loadingElement = container.querySelector('.loading');
        if (loadingElement) {
            loadingElement.remove();
        }

        // Перерисовываем график с текущими данными
        renderChart(chartData);
    }, 800);
}

// Инициализация графика при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    // Используем глобальные данные из index.html
    if (typeof chartData !== 'undefined') {
        renderChart(chartData);
    } else {
        // Если данные не определены, используем моковые данные
        const mockData = {
            sources: ['BBC', 'CNN', 'Fox News', 'The Guardian', 'Reuters'],
            scores: [0.9, 0.75, 0.65, 0.85, 0.95]
        };
        renderChart(mockData);
    }
});
