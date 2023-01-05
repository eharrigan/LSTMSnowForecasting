import csv
import os
from collections import defaultdict
from statistics import mean

def convert_hourly_to_daily():
    for item in os.listdir("./data"):
        f = os.path.join("./data", item)
        if os.path.splitext(f)[1] != '.csv' or item[0] != "4":
            print("Skipping file named:", f)
            continue
        with open(f) as inp, open("./data/DAILY_" + item, 'w') as out:
            reader = csv.reader(inp)
            writer = csv.writer(out, delimiter=',')
            writer.writerow(next(reader))
            for row in reader:
                OBS_DATE = row[5]
                if OBS_DATE.split()[1] == "0000":
                    writer.writerow(row)
def interpolate():
    keys = ["STATION_ID","DURATION","SENSOR_NUMBER","SENSOR_TYPE","DATE TIME","OBS DATE","VALUE","DATA_FLAG","UNITS"]
    for item in os.listdir("./data"):
        count = 0
        flag = True
        d = defaultdict(list)
        f = os.path.join("./data", item)
        if os.path.splitext(f)[1] != '.csv' or item[0] != "D":
            print("Skipping file named:", f)
            continue
        with open(f) as inp, open("./data/OUT_" + item, 'w', newline='') as out:
            reader = csv.DictReader(inp)
            missing_streak = 0
            writer = csv.DictWriter(out,fieldnames=keys, delimiter=',')
            for row in reader:
                if row["VALUE"] != "---":
                    d[row["OBS DATE"].split()[0][4:]].append(float(row["VALUE"]))
                else:
                    count += 1

            inp.seek(0)
            if count > 100:
                flag = False
            if flag:
                print(item)
                writer.writeheader()
                for row in reader:
                    if row["VALUE"] =="---":
                        row["VALUE"] = str(int(mean(d[row["OBS DATE"].split()[0][4:]])))
                        writer.writerow(row)
                    else:
                        writer.writerow(row)
        if flag == False:
            os.remove("./data/OUT_"+item)
                       
                    
                    
def missing_days():
    for item in os.listdir("./data"):
        f = os.path.join("./data", item)
        if os.path.splitext(f)[1] != ".csv": 
            continue
        with open(f) as inp:
            reader = csv.DictReader(inp)
            count = 0
            for row in reader:
                if not is_number(row["VALUE"]):
                    count += 1

        print(item, "had", str(count), "missing days")


def count_flags():
    for item in os.listdir("./data"):
        f = os.path.join("./data", item)
        if os.path.splitext(f)[1] != ".csv": 
            continue
        with open(f) as inp:
            reader = csv.DictReader(inp)
            count = 0
            for row in reader:
                if row["DATA_FLAG"] != ' ' and  row["DATA_FLAG"] != "r" :
                    
                    count += 1

            print(item, "had", str(count), "days with flags")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


count_flags()
