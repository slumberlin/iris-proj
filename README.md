# iris-proj

## Intro
Goal of the project is to build a machine learning model using the xgboost algorithm to predict the flowers, using the  *iris* dataset.
In addition to the model, a user would be able to submit requests with new data and get a prediction as a response.

## Materials

## Methods

## Installing XGBoost

Follow "Build from the source code - advanced method" https://xgboost.readthedocs.io/en/latest/build.html. I needed to install cmake for that to work and at the end install using `python setup.py install`

## First model
If you run the notebook under `notebooks/iris-xgboost.ipynb`, it trains an xgboost model and saves it in `app/model.pkl`

## Adding flask API as scoring platform

Run `python app.py` to start the server

Sample request: `curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data '{"sepal length (cm)": "6.7","sepal width (cm)": "3.1","petal length (cm)":"4.4", "petal width (cm)":"1.4"}'` 

This is expected to return `{"PREDICTION": "versicolor"}`
