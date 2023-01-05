import csv
import os
from datetime import date
import requests

station_list = ["GEM"]
URL = "http://cdec4gov.water.ca.gov/dynamicapp/req/CSVDataServlet"
sensor_nums = ["3", "18", "32"] #SWC, SNOW DEPTH, AIR TEMP
duration = "D"
start = "2012-01-01"
end = "2022-12-31"
params = {"SensorNums" : sensor_nums,
        "dur_code" : duration,
        "Start" : start,
        "End" : end}
count = 0

for ID in station_list:
    params["Stations"] = ID
    print("Requesting data for", ID)
    for num in sensor_nums:
        params["SensorNums"] = num
        if num == "4":
            params["dur_code"] = "H"
        else:
            params["dur_code"] = "D"
        req = requests.get(URL, params)
        content = req.content
        if(len(content) < 300):
            print("Data for", ID, num, "too small, not writing CSV")
            print("Deleting all files associated with station:", ID)
            for item in os.listdir("./data"):
                if item.endswith(ID+".csv"):
                    os.remove(os.path.join("./data", item))
            break
        print("Writing csv for", ID, num)
        csv_file = open("./data/"+num+"_" + ID +".csv", 'wb')
        csv_file.write(content)
        csv_file.close()
        req.close()
        count += 1
print("Wrote CSV files for", count, "stations")
