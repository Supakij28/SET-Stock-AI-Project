// ตัวอย่างข้อมูล (ดึงมาจากระบบ Backtesting)
const stockData = {
    labels: ["2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05"],
    prices: [100, 105, 110, 102, 108],
    buySignals: [1, 0, 0, 0, 0], // 1: Buy, 0: No Action
    sellSignals: [0, 0, 0, 1, 0] // 1: Sell, 0: No Action
};

// ผลลัพธ์การซื้อขาย
const initialBalance = 100000;
const finalBalance = 102500; // คำนวณจากระบบ Backtesting
const profitLoss = finalBalance - initialBalance;

// อัปเดตข้อมูลผลลัพธ์ใน HTML
document.getElementById("initial-balance").innerText = `${initialBalance.toFixed(2)} THB`;
document.getElementById("final-balance").innerText = `${finalBalance.toFixed(2)} THB`;
document.getElementById("profit-loss").innerText = `${profitLoss.toFixed(2)} THB`;

// เตรียมข้อมูลสำหรับแสดงผลกราฟ
const buyPoints = stockData.prices.map((price, index) =>
    stockData.buySignals[index] === 1 ? price : null
);
const sellPoints = stockData.prices.map((price, index) =>
    stockData.sellSignals[index] === 1 ? price : null
);

// สร้างกราฟด้วย Chart.js
const ctx = document.getElementById("tradingChart").getContext("2d");
const tradingChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: stockData.labels,
        datasets: [
            {
                label: "Stock Price",
                data: stockData.prices,
                borderColor: "blue",
                fill: false
            },
            {
                label: "Buy Signal",
                data: buyPoints,
                borderColor: "green",
                backgroundColor: "green",
                pointStyle: "triangle",
                pointRadius: 10,
                fill: false,
                showLine: false
            },
            {
                label: "Sell Signal",
                data: sellPoints,
                borderColor: "red",
                backgroundColor: "red",
                pointStyle: "triangle",
                pointRadius: 10,
                fill: false,
                showLine: false
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: "top"
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: "Date"
                }
            },
            y: {
                title: {
                    display: true,
                    text: "Price (THB)"
                }
            }
        }
    }
});
