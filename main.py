#import library
import uvicorn
from fastapi import FastAPI
from BankDataModel import BankNote
import numpy as np
import pickle
import pandas as pd

#create the app object
app=FastAPI()
pickle_model=open("classifier.pkl","rb")
classifier=pickle.load(pickle_model)

#default route
@app.get('/')
def index():
    return{"message":"Hello IDM Students"}

#default route
@app.get('/api-demo')
def index():
    return{"message":"This is demo API"}

#Prediction Function, return the predicted result in JSON
@app.post('/predict')
def predict(data:BankNote):
    #convert data obj to dictionary
    data=dict(data)
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    #prediction
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    #return probability
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

#Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

#Command to run API server   
#python -m uvicorn main:app --reload

