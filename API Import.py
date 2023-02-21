import requests
import json
import pandas as pd
#pd.set_option('display.max_columns', None)

response_API = requests.get('https://api.thedogapi.com/v1/breeds')

#Check connection, 200 means all good
print(response_API.status_code)

data = pd.DataFrame(response_API.json())
data.set_index('name')

print (data[['name','bred_for']])

