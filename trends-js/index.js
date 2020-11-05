const googleTrends = require("google-trends-api");
const moment = require("moment");
const csv = require("csv-parser");
const fs = require("fs");
const ObjectsToCsv = require("objects-to-csv");

const getTrends = async (startDate, endDate, keyWords) => {
  return googleTrends.interestOverTime({
    keyword: keyWords,
    startTime: startDate,
    endTime: endDate,
    geo: "US",
  });
};

const readCsv = () => {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream("../datasets/output_node.csv")
      .pipe(csv())
      .on("data", (row) => {
        rows.push(row);
      })
      .on("end", () => {
        resolve(rows);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
};

const processEntities = (rows) => {
  return rows.map((row) => {
    const entities = row.entities;

    const entities_array = entities
      .trim()
      .substr(1, entities.length - 2)
      .split(",")
      .map((e) => e.trim());
    return {
      ...row,
      entities_array,
    };
  });
};
const logsPath = "./logs.txt";
const csvPath = "./out.csv";

const removeLogsFile = () => {
  try {
    fs.unlinkSync(logsPath);
  } catch (err) {}
  try {
    fs.unlinkSync(csvPath);
  } catch (err) {}
};

const writeLog = (message) => {
  fs.appendFileSync(logsPath, `${message}\n`);
};

const addTrendsToRows = async (rows) => {
  const newRows = [];

  for (let i = 0; i < rows.length; i++) {
    try {
      console.log(`Inicia proceso para fila ${i}`);
      const row = rows[i];

      const entities = row.entities_array;
      const mStartDate = moment(row.created_at, "YYYY-MM-DD");
      const mEndDate = moment(row.updated_at, "YYYY-MM-DD");

      if (entities === null || entities.length <= 0) {
        newRows.push({ ...row, entities_trend: 0 });
      } else if (entities[0] == "") {
        newRows.push({ ...row, entities_trend: 0 });
      } else {
        const trend = await getTrends(
          mStartDate.toDate(),
          mEndDate.toDate(),
          entities
        );
        try {
          const trendObject = JSON.parse(trend);
          const averages = trendObject.default.averages;
          const averageTrend = getAverage(averages);
          newRows.push({
            ...row,
            entities_trend: averageTrend,
          });
        } catch (error) {
          writeLog(`Error fila ${i}:`);
        }
        console.log(`Finaliza proceso para fila ${i}`);
      }
    } catch (error) {
      writeLog(`Error fila ${i}:`);
      console.log(`>>>>>>>>>>>>>>>>>>>>>>>> Error fila ${i}:`, error);
    }
  }

  return newRows;
};

const getAverage = (arr) => {
  if (arr.length == 0) {
    return 0;
  }

  return arr.reduce((sume, el) => sume + el, 0) / arr.length;
};

const init = async () => {
  removeLogsFile();

  let rows = await readCsv();
  rows = processEntities(rows);

  // Jonatan
  rows = rows.slice(0, 4000);
  // Sebastian
  // rows = rows.slice(4000, 8000);
  // Yesid
  // rows = rows.slice(8000, rows.length);

  rows = await addTrendsToRows(rows);

  const csv = new ObjectsToCsv(rows);

  try {
    await csv.toDisk(csvPath);
  } catch (error) {
    console.log("Falló al crearse el csv");
  }
};

init();
