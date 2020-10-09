from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Blockchain"]
# pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
# print(pytrends.trending_searches(pn='united_states'))  # trending searches in real time for United States
# print(pytrends.top_charts(2020, hl='en-US', tz=300, geo='GLOBAL'))
print(pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False))