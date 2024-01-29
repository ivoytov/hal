function convertDateCol(obj, dateKey) {
  // Use destructuring to get other properties if needed
  const { [dateKey]: saleDate, ...rest } = obj;

  // Convert string to Date, set to midnight (otherwise date filter doesn't work)
  const saleDateObj = new Date(saleDate);
  if (dateKey === 'date') {
    saleDateObj.setHours(24, 0, 0, 0)
  }

  // Add other properties back if needed
  return { [dateKey]: saleDateObj, ...rest };
}


// Function to load CSV file using PapaParse
function loadCSV(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: results => {
        results.data = results.data.map(obj => {

          const dateKeys = ["date_time", "date", "maturity", "t1"]

          for (const dateKey of dateKeys) {
            if (dateKey in obj) {
              obj = convertDateCol(obj, dateKey)
            }
          }

          return obj
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
const csvUrls = ['126', '127', '128', '202', '203', '204', '205', '208', '209', '210', '211', '212', '216']

// Array to store promises for each CSV file
const csvPromises = csvUrls.map(url => loadCSV(`trades/bond_trades_210${url}.csv`));


const formattedPercent = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1
})

const defaultColDef = {
  flex: 1,
  minWidth: 100,
  filter: 'agTextColumnFilter',
  menuTabs: ['filterMenuTab'],
  autoHeaderHeight: true,
  wrapHeaderText: true,
  sortable: true,
  resizable: true
}

const columnDefs = [
  {
    headerName: "Trade Date",
    field: "date",
    filter: 'agDateColumnFilter',
    sort: "asc",
    sortIndex: 0,
    sortable: true
  },
  {
    field: "name",
    headerName: "Issuer"
  },
  {
    field: "cpn",
    headerName: "Coupon",
    valueFormatter: (params) => formattedPercent.format(params.value / 100)
  },
  {
    field: "maturity",
  },
  {
    field: "t1",
    headerName: "Vertical Barrier"
  },
  {
    field: "close",
    headerName: "Signal Price",
    valueFormatter: (params) => params.value.toFixed(2)
  },
  {
    headerName: "Stop Loss Price",
    field: 'stop_loss',
    valueFormatter: (params) => params.value.toFixed(2)
  },
  {
    headerName: "Profit Take Price",
    field: "profit_take",
    valueFormatter: (params) => params.value.toFixed(2)
  },
  {
    field: "signal",
    valueFormatter: (params) => formattedPercent.format(params.value)
  },
  {
    field: "trgt",
    headerName: "Position Size",
    valueFormatter: (params) => formattedPercent.format(params.value)
  },

]

// Initialize AG Grid
const gridOptions = {
  columnDefs: columnDefs,
  defaultColDef: defaultColDef,
  //masterDetail: true,
  //detailRowHeight: 200,
  detailRowAutoHeight: true,
  rowSelection: 'single',
  onSelectionChanged: onSelectionChanged,

};

// Create AG Grid
const gridDiv = document.querySelector('#myGrid');
const gridApi = agGrid.createGrid(gridDiv, gridOptions)

// gridApi.setFilterModel(defaultFilter);

Promise.all([
  loadCSV("df_data.csv"),
  loadCSV("px_data.csv"),
  ...csvPromises
])
  .then(([df, px, ...tradesArrays]) => {

    dfData = df
    pxData = px
    trades = tradesArrays.flat()

    const tickers = trades.reduce((map, trade) => {
      if (!map.has(trade.ticker)) {
        // If the ticker is not already in the map, add it with an initial value
        map.set(trade.ticker, { name: trade.name, maturity: trade.maturity.getFullYear(), coupon: trade.cpn });
      }
      return map;
    }, new Map());

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

    for (const ticker of tickers.keys()) {
      const data = px.filter(({ ticker: t }) => ticker == t)
        .map(({ date_time, price }) => [date_time, price])
      if (data.length === 0) {
        continue
      }
      const bond = tickers.get(ticker)
      const name = `${bond.name} ${bond.coupon} ${bond.maturity}`
      const events = df.filter(({ ticker: t }) => ticker === t)
        .map(({ date_time, close, side }) => [date_time, close, side])

      const bot = trades.filter(({ ticker: t }) => ticker === t)
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

    isSelected["GPOR 6 2024"] = true

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

    gridApi.setGridOption('rowData', trades)
    gridApi.sizeColumnsToFit()
  })

// Initialize the echarts instance based on the prepared dom
const myChart = echarts.init(document.getElementById('main'));

function onSelectionChanged() {
  const selectedRow = gridApi.getSelectedRows()[0];
  const name = `${selectedRow.name} ${selectedRow.cpn} ${selectedRow.maturity.getFullYear()}`

  // Suppose legend data is ['series1', 'series2', 'series3']
  const legendData = myChart.getOption().legend[0].data;

  // Deselect all legend using legendUnSelect
  legendData.forEach((item) => {
    myChart.dispatchAction({
      type: 'legendUnSelect',
      name: item   // the legend name or the series name, They are the same usually.
    });
  });

  myChart.dispatchAction({
    type: "legendToggleSelect",
    name: name
  })
  console.log(name)
}