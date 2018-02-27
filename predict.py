import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
#from StringIO import StringIO #para leer los datos de train
from collections import deque #para leer los datos de train
from flask import Flask, make_response, send_file, request, jsonify
from main import app

app = Flask(__name__)


@app.route("/predict", methods = ['POST']) #curl -F "data=@./data/test.csv" http://127.0.0.1:5000/predict
def predict():



        file_csv = request.files['data']
        test_data = pd.read_csv(file_csv, index_col = 0)
        scaler = MinMaxScaler(feature_range=(-1,1))
        test_data = scaler.fit_transform(test_data)
        test_data = np.transpose(test_data)






        # New Hyperparams

        '''
        num_epochs = number of iteracions over the entire dataset
        total_series_length = number of entries
        truncated_backprop_length = 15
        state_size = dimension of the matrix for saving the recurrence values
        batch_size = numero de variables que le pasamos a la red 
        num_batches = number of group of elements between trainings
        '''

        batch_size = 4
        state_size = 4
        total_series_length = test_data.shape[1]
        truncated_backprop_length = 200
        num_batches = total_series_length//truncated_backprop_length










        # Read train data
        
        with open('data/train.csv','r') as line:
                q = deque(line,truncated_backprop_length)

        train_data = pd.read_csv(StringIO(''.join(q)), index_col = 0,header = None)        










        # Restore the model

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph('./model/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./model'))

        graph = tf.get_default_graph()
        batchX_placeholder = graph.get_tensor_by_name("batchX_placeholder:0")
        batchY_placeholder = graph.get_tensor_by_name("batchY_placeholder:0")
        init_state = graph.get_tensor_by_name("init_state:0")

        sess.run(tf.global_variables_initializer())












        # Prediction

        prediction = []
        predictions_series = graph.get_tensor_by_name("predictions_series:0")
        delete = sess.run('first_current_state:0')
        _current_state = sess.run('first_current_state:0')  #FALLO


        for batch_idx in range(num_batches):
    
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
    
            x=test_data[:,start_idx:end_idx]
    
    
            _predictions_series = sess.run([predictions_series],
                                           feed_dict={
                                               batchX_placeholder:x,
                                               init_state:_current_state
                                           })


            prediction = prediction + _predictions_series
        prediction = np.concatenate(prediction,axis = 0)
        prediction = np.stack(prediction,axis = 0)[:,0,0]
 
        # Reescalado Opcion 1
        # prediction = scaler.inverse_transform(prediction)


        # Reescalado Opcion 2
        prediction = prediction*train_data.iloc[:,0].std()/prediction.std()
        prediction = prediction+train_data.iloc[:,0].mean()-prediction.mean()


        #response = make_response(np.savetxt('init.txt',delete))
        #response.headers['Content-Disposition'] = "attachment; filename = init.csv"
        #response.headers['Content-Type'] = 'text/csv'


        return jsonify(np.array(prediction).tolist())
