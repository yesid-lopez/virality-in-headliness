const googleTrends = require("google-trends-api");

const initial_date = new Date(2014, 10, 20);
const final_date = new Date(2016, 3, 2);
const getTrends = async () => {
  const response = await googleTrends.interestOverTime({
    keyword: ['sexism', 'men', 'sounds', 'like'],
    startTime: initial_date,
    endTime: final_date,
    geo: "US",
  });
  console.log(response);
};

getTrends();
