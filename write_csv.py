import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
import requests
directory = "./data"
header = ["STATION", "DATE", "SWC", "TEMP", "DEPTH" ]
fail_list = []
for station_id in ["CAP", "MHP"]:
    num_missing = 0
    consec_fail = 0
    max_consec = 0
    success = 0
    previous_fail = False
    print("Parsing station", station_id)
    depth_df = pd.read_csv("./data/18_" + station_id + ".csv") 
    temp_df = pd.read_csv("./data/32_" + station_id + ".csv")
    swc_df = pd.read_csv("./data/3_" + station_id + ".csv")

    #first rename the value column of each df to the correct label
    depth_df.rename(columns = {'VALUE':'DEPTH'}, inplace = True)
    temp_df.rename(columns = {'VALUE':'TEMP'}, inplace = True)
    swc_df.rename(columns = {'VALUE':'SWC'}, inplace = True)
    
    #now merge the dataframes on the column DATE TIME

    complete_df = pd.merge(depth_df[['STATION_ID', 'DATE TIME', 'DEPTH']], temp_df[['DATE TIME', 'TEMP']], on='DATE TIME', how='inner').merge(swc_df[['DATE TIME', 'SWC']], on='DATE TIME')
    complete_df['DEPTH'] = pd.to_numeric(complete_df['DEPTH'], errors='coerce')
    complete_df['SWC'] = pd.to_numeric(complete_df['SWC'], errors='coerce')
    complete_df['TEMP'] = pd.to_numeric(complete_df['TEMP'], errors='coerce')
    complete_df['DEPTH'] = complete_df['DEPTH'].interpolate(method='linear', limit_direction='forward')
    complete_df['TEMP'] = complete_df['TEMP'].interpolate(method='linear', limit_direction='forward')
    complete_df['SWC'] = complete_df['SWC'].interpolate(method='linear', limit_direction='forward')
    complete_df['DATE TIME'] = pd.to_datetime(complete_df['DATE TIME'])
    complete_df.to_pickle("data/" + station_id + ".pkl")
    print(complete_df.head)
#        depth_reader = csv.DictReader(depth)
#        temp_reader = csv.DictReader(temp)
#        swc_reader = csv.DictReader(swc)
#        prev_data = ""
#        writer.writerow(header)
#        for row in reader:
#            #iterate through every row in the revised csv
#            OBS_DATE = row["OBS DATE"].split()[0]
#            day = OBS_DATE[4:]
#            #ignore the day if it's the 29th of february
#            if OBS_DATE[4:] == "0229":
#                continue
#            year = OBS_DATE[:4]
#            #if the data is missing, use the previous data 
#            data = row["VALUE"]
#            if data == "---":
#                
#                data = prev_data
#                if data != '0.00':
#                    num_missing += 1
#                    if previous_fail:
#                        consec_fail += 1
#                    else:
#                        previous_fail = True
#                        consec_fail = 1
#            else:
#                previous_fail = False
#                if consec_fail > max_consec:
#                    max_consec = consec_fail
#                consec_fail = 0
#            if data[0] == "-":
#                data = "0.00"
#            if consec_fail < 10:
#                record = [station, OBS_DATE, data]
#                writer.writerow(record)
#                prev_data = data
#                success += 1
#                
#
#        print(station+ ": " + "data for", str(num_missing), "days missing out of ", str(success), "total days written")
#        print("Maximum number of consecutive days with missing data:", max_consec)
#
#        
#                
#            
#
#
#


