# api
# application programming interface

# server application => client in application
# client application => server in application

# url => uniform resource locator

import requests
import json
import pandas as pd

url = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/23.0225,72.5714;21.170240,72.831062?approaches=curb;curb&access_token=pk.eyJ1IjoibWFwYm94Z2VvY29kaW5nIiwiYSI6ImNrdDhoZnpqdzEyM2YydHBlanM0dDN1eW8ifQ._wqJ3VfJi1HaEPuvX_sTew"

response =  requests.get(url,headers={"Accept":"application/json"})

df =  pd.read_json(response.text)
print(df.info())

if response.status_code == 200:
    dict =  json.loads(response.text)
    if(dict["code"] == "NoRoute"):
        print("No route found")
    else:
        print(dict["distances"])
else:
    print(response.text)


resposne = requests.get("https://www.google.com/search?client=ubuntu-sn&channel=fs&q=ronin")
print(resposne.text)
