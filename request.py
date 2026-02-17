import requests

url = "http://127.0.0.1:8000/run-model"

with open("item_forecast.csv","rb") as f1, \
     open("onhand_inventory.csv","rb") as f2, \
     open("item_shelf_life.csv","rb") as f3, \
     open("item_delivery_schedule.csv","rb") as f4:

    files = {
        "forecast": f1,
        "inventory": f2,
        "shelf": f3,
        "delivery": f4,
    }

    response = requests.post(url, files=files)

print(response.status_code)
print(response.text)
