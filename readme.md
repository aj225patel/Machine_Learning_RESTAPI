This project demonstrates the development and deployment of a machine-learning model for banknote authentication using FastAPI and the Uvicorn ASGI web server.

To run this FastAPI application, follow these steps:

## 1. Install the required libraries using pip:
```bash
pip install fastapi uvicorn numpy pickle pandas scikit-learn
```

## 2. Command to run API server:
```bash
python -m uvicorn main:app --reload
```

## 3. The API will be accessible at http://127.0.0.1:8000
- You can make predictions by sending POST requests to http://127.0.0.1:8000/docs with JSON data in the format specified by the 'BankNote' model in 'BankDataModel.py'.
- Here's an example request:
```json
{
  "variance": 3.6216,
  "skewness": 8.6661,
  "curtosis": -2.8073,
  "entropy": -0.44699
}
```
- The API will return a prediction indicating whether the banknote is fake or genuine.
