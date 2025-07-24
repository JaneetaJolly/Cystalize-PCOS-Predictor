import requests

url = 'http://127.0.0.1:5000/predict'

data = {
    "Age (yrs)": 25,
    "Weight (Kg)": 65,
    "Height(Cm)": 160,
    "BMI": 25.4,
    "Cycle(R/I)": 1,
    "Fast food (Y/N)": 1,
    "Pimples(Y/N)": 1,
    "hair growth(Y/N)": 0,
    "Skin darkening (Y/N)": 1,
    "Follicle No. (L)": 10,
    "Follicle No. (R)": 12,
    "Avg. F size (L) (mm)": 5.8,
    "Avg. F size (R) (mm)": 6.1
}

response = requests.post(url, json=data)
print(response.json())
