from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

iris = load_iris()

loaded_model = load('logreg.joblib')

app = FastAPI()

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

@app.post('/predict')

def predict(data : request_body):
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    class_idx = loaded_model.predict(new_data)[0]

    return {'class' : iris.target_names[class_idx]}

@app.post('/hello')
def hello():
    text = 'Hello world !'
    return text

if __name__ == "__main__":
    app.run()
