const googleTrends = require("google-trends-api");

const initial_date = new Date(2014, 10, 20);
const final_date = new Date(2016, 3, 2);
const getTrends = async () => {
  const response = await googleTrends.interestOverTime({
    keyword: ["sexism", "men", "sounds", "like"],
    startTime: initial_date,
    endTime: final_date,
    geo: "US",
  });
};

getTrends();

const csv = require("csv-parser");
const fs = require("fs");

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

const init = async () => {
  let rows = await readCsv();
  rows = processEntities(rows);
  console.log(rows[10]);
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

init();
