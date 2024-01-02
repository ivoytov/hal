// Function to load CSV file using PapaParse
function loadCSV(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: results => {
        // Convert "SALE DATE" property to JavaScript Date objects
        results.data = results.data.map(obj => {
          const dateKey = "date_time" in obj ? "date_time" : 'date'

          // Use destructuring to get other properties if needed
          const { [dateKey]: saleDate, ...rest } = obj;

          // Convert string to Date, set to midnight (otherwise date filter doesn't work)
          const saleDateObj = new Date(saleDate);
          if (dateKey === 'date') {
            saleDateObj.setHours(24, 0, 0, 0)
          }

          // Add other properties back if needed
          return { [dateKey]: saleDateObj, ...rest };
        });


        resolve(results.data);
      },
      error: error => {
        reject(error.message);
      }
    });
  });
}

let dfData, pxData, trades

// URLs of the CSV files you want to load
const csvUrls = ['126', '127', '128', '202', '203', '204', '205', '208', '209', '210', '211', '212', '216' ]

// Array to store promises for each CSV file
const csvPromises = csvUrls.map(url => loadCSV(`trades/bond_trades_210${url}.csv`));


Promise.all([
  loadCSV("df_data.csv"),
  loadCSV("px_data.csv"),
  ...csvPromises
])
  .then(([df, px, ...tradesArrays]) => {

    dfData = df
    pxData = px
    trades = tradesArrays.flat()

    const tickers = new Set(trades.map(({ ticker }) => ticker))

    const seriesNames = []
    const series = []
    const makeSeries = (data, name) => {
      return {
        name: name,
        type: 'line',
        symbol: 'none',
        data: data
      }
    }

    const makeEvents = (data, name) => {
      return {
        name: name,
        type: 'line',
        symbol: 'triangle',
        symbolRotate: (value, params) => params.data[2] == 1 ? 0 : 180,
        showSymbol: true,
        symbolSize: 15,
        lineStyle: {
          width: 0
        },
        itemStyle: {
          color: params => params.data[2] == 1 ? 'blue' : 'yellow',
          borderColor: 'black'
        },
        data: data
      }
    }

    const makeTrades = (data, name) => {
      return {
        name: name,
        type: 'line',
        symbol: 'emptyCircle',
        showSymbol: true,
        symbolSize: 15,
        lineStyle: {
          width: 0
        },
        itemStyle: {
          color: 'green',
          borderColor: 'black'
        },
        z: 5,
        data: data
      }
    }

    for (const ticker of tickers) {
      const data = px.filter(({ ticker: t }) => ticker == t)
        .map(({ date_time, price }) => [date_time, price])
      if (data.length === 0) {
        continue
      }
      const name = `${ticker}`
      const events = df.filter(({ ticker: t}) => ticker === t)
        .map(({ date_time, close, side }) => [date_time, close, side])
      
      const bot = trades.filter(({ticker: t}) => ticker === t)
        .map(({ date, close, }) => [date, close])
      series.push(makeSeries(data, name))
      series.push(makeEvents(events, name))
      series.push(makeTrades(bot, name))

      seriesNames.push(name)

    }

    const isSelected = seriesNames.reduce((accumulator, current) => {
      accumulator[current] = false;
      return accumulator;
    }, {})

    isSelected["AL018402"] = true

    // Specify the configuration items and data for the chart
    const option = {
      title: {
        text: 'Hal Analysis'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        selected: isSelected,
        type: 'scroll',
        orient: 'horizontal',
        data: seriesNames,
        top: 20,
        left: 20,
        right: 20,
        textStyle: {
          width: 75,
          overflow: 'break'

        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 50,
          end: 100
        },
        {
          show: true,
          type: 'slider',
          top: '90%',
          start: 50,
          end: 100
        }
      ],
      grid: {
        top: 100
      },
      xAxis: {
        type: 'time'
      },
      yAxis: {
        min: 'dataMin',
      },
      series:
        series

    };

    // Display the chart using the configuration items and data just specified.
    myChart.setOption(option);
  })

// Initialize the echarts instance based on the prepared dom
var myChart = echarts.init(document.getElementById('main'));

