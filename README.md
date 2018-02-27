# RNN-TimeSerie-Predicton-App

This project is based in a Recurrent NeuralNet written in Python using [Tensorflow](https://github.com/tensorflow/tensorflow) for predicting near-real-time data from a seasonal TimeSerie. The project
is embedded in a web application framework built with [Flask](https://github.com/pallets/flask).

## Data

The data consist on a TimeSerie with 15min step between measures. The aim of the application is to estimate the very next data to the last one.
The structure of the data expect by the NeuralNet is a five columns .csv in which the columns means the following:

  * Date: The current date.
  * Data: The current value of the data.
  * Month: The value o the data at the exact same hour 30 days ago.
  * Week: The value o the data at the exact same hour 7 days ago.
  * Year: The value o the data at the exact same hour 365 days ago.

## NeuralNet Architecture

The neuralnet is explained step by step in the notebook [RNN_Tensorflow_explained.ipynb](https://github.com/DelgadoPanadero/RNN-TimeSerie-Predicton-App/blob/master/RNN_Tensorflow_explained.ipynb). The input is a four column matrix and the neural architecture is as follows:
 
  * One recurrent hidden layer four neurons with a Sigmoid activation function. The recurrence is four steps back.
  * One output layer with a Sigmoid activation function.
  
The optimizer is the Adagrad optimezer implented in Tensorflow.


## Schema of the application

The web application (it has not been deployed to run out of local yet) is divided in three modules. The first module is [main.py](https://github.com/DelgadoPanadero/RNN-TimeSerie-Predicton-App/blob/master/main.py) which is the one that runs the app.
The modules [train.py](https://github.com/DelgadoPanadero/RNN-TimeSerie-Predicton-App/blob/master/train.py) and [predict.py](https://github.com/DelgadoPanadero/RNN-TimeSerie-Predicton-App/blob/master/predict.py) contain the train model and the predictive model of the NeuralNet.
The module [get_data.py](https://github.com/DelgadoPanadero/RNN-TimeSerie-Predicton-App/blob/master/get_data.py) is not required for running the app, this is only needed for simulating the TimeSerie data coming from a database.

## How to run the App

First you have to run the Flask app to raise the localserver. This can be made executing main.py in the terminal as follows

```
python main.py
```

Once the localserver has been created, the train model can be executed throughout a GET request. The model reads to the data in the relative path `data/train.csv` (This is very important).

```
curl http://localhost:5000/test
```

The response of the request is a list with the losses values through the training process. The predictive model can be executed with a POST request, posting the that in which you want to predict.

```
#curl -F "data=@./data/test.csv" http://127.0.0.1:5000/predict
```
