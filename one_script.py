import os
import io
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque #para leer los datos de train
from io import StringIO #para leer los datos de train
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, make_response, send_file, request, jsonify







app = Flask(__name__)









@app.route("/get_data")
def get_data():
        #Time
        date = pd.date_range(start = "1/1/2014", end = "1/1/2018", freq = '15Min')[:-1]

        #Ruido
        season_noise = np.random.rand(len(date))
        hour_noise = np.random.rand(len(date))
        trend_noise = np.random.rand(len(date))


        #Components of the time serie
        trend = date.dayofyear/366+(date.year-2014) #Crecimiento lineal a diario
        season = np.cos(4*np.pi*(date.dayofyear/365+season_noise))  #Estacionalidad ondulatoria con dos periodos al año
        day = np.floor(date.dayofweek/5)            #Con esto los días Viernes y Sábado tienen valor 1 y los demás 0
        hour = np.cos(2*np.pi*((date.hour-19)/24)+3*hour_noise)
        anomalies = []


        #TimeSerie
        data = trend+season+day/2+hour+trend_noise+6
        ts = pd.Series(data = data, index = date)


        #Dataframe
        df = pd.DataFrame(index = ts[ts.index>="2015-01-01 00:00:00"].index,
        		data = {'data' : ts[ts.index>="2015-01-01 00:00:00"].values,
        			'year' : ts[ts.index<"2017-01-01 00:00:00"].values,
        			'week' : ts[np.logical_and(ts.index>="2014-12-25 00:00:00",ts.index<="2017-12-24 23:45:00")].values,
        			'month' : ts[np.logical_and(ts.index>="2014-12-01 00:00:00",ts.index<="2017-11-30 23:45:00")].values})

        response = make_response(df.iloc[:(len(df)-600),:].to_csv())
        response.headers['Content-Disposition'] = "attachment; filename = train.csv"
        response.headers['Content-Type'] = 'text/csv'

        #response = make_response(df.iloc[(len(df)-600-200):,:].to_csv())
        #response.headers['Content-Disposition'] = "attachment; filename = test.csv"
        #response.headers['Content-Type'] = 'text/csv'
        
        #response = make_response(df.to_json())
        #response.headers['Content-Disposition'] = "attachment; filename = data.json"
        #return df.to_json()

        return response
	


	






@app.route("/predict", methods = ['POST']) #curl -F "data=@./data/test.csv" http://127.0.0.1:5000/predict
def predict():



        file_csv = request.files['data']
        #stream = io.StringIO(file_csv.stream.read().decode("UTF8"), newline=None)
        #test_data = csv.reader(file_csv)
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
        num_batches = total_series_length//truncated_backprop_length#//batch_size





        # Read train data

        with open('data/train.csv','r') as line:
                q = deque(line,truncated_backprop_length)

        train_data = pd.read_csv(StringIO(''.join(q)), index_col = 0,header = None)
        # train_data = pd.read_csv('data/train.csv', index_col = 0, nrow = truncated_backprop_length) #first lines







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
    
            prediction = prediction + _predictions_series#[0]      

        prediction = np.stack(prediction,axis = 0)[:,:,0]

        # Reescalado Opcion 1
        # prediction = scaler.inverse_transform(prediction)


        # Reescalado Opcion 2
        prediction = prediction[:,0]
        prediction = prediction*train_data.iloc[:,0].std()/prediction.std()
        prediction = prediction+train_data.iloc[:,0].mean()-prediction.mean()


        #response = make_response(np.savetxt('init.txt',delete))
        #response.headers['Content-Disposition'] = "attachment; filename = init.csv"
        #response.headers['Content-Type'] = 'text/csv'


        return jsonify(np.array(prediction).tolist())















@app.route("/train")
def train():


        train_data = pd.read_csv('data/train.csv', index_col = 0)
        scaler = MinMaxScaler(feature_range=(-1,1))
        train_data = scaler.fit_transform(train_data)
        train_data = np.transpose(train_data)

        # Hyperparams

        '''
        num_epochs = number of iteracions over the entire dataset
        total_series_length = number of entries
        truncated_backprop_length = 15
        state_size = dimension of the matrix for saving the recurrence values
        batch_size = numero de variables que le pasamos a la red 
        num_batches = number of group of elements between trainings
        '''

        num_epochs = 1
        total_series_length = train_data.shape[1]
        truncated_backprop_length = 200
        state_size = 4
        batch_size = 4
        num_batches = total_series_length//truncated_backprop_length#//batch_size
        skip = total_series_length-num_batches*truncated_backprop_length-1 #saltamos estos elementos del dataset para tener un número entero de batches





        # Input data

        batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length], name='batchX_placeholder')
        batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length], name='batchY_placeholder')
        init_state = tf.placeholder(tf.float32, [batch_size, state_size], name = 'init_state') #Aquí se pasa la información de los datos anteriores (recurrencia)






        # Recurrent Neural Net

        '''
        Red neuronal con una capa oculta recurrente (W,b) y una capa de salida (W2,b2). Data un set de datos de batch_size entradas, al introducirlo en la red, se ejecutan batch_size iteraciones sobre la primera capa

        Capa recurrente:
            - Recibe un vector de datos de entrada concatenado con un arra (state_size,state_size), por eso W tiene state_size+1 columnas.
    	    - Para un batch de datos de batch_size entradas, se ejecutan batch_size iteraciones sobre la primera capa y se obtiene de salida una lista de batch_size vectores antes de pasar los resultados a la siguiente capa


        Capa de salida:
            - Recibe, para cada iteración un array(state_size,1) y devuelve un número
            - Se itera sobre los batch_size elementos de la lista que se obtiene de la capa anterior
        '''

        	# Hidden layer
        W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
        b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

        	# Output layer
        W2 = tf.Variable(np.random.rand(state_size, 1),dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1,1)), dtype=tf.float32)



        # Unpack the input data

        '''
        Esto se hace para conviertir un array con dimensiones (m,n) en una lista de n arrays de dimensión (m,1).
        Cada elemento de la lista es un set de datos del que se saca una predicción. 

            batchX_placeholder: np.array(m,n)
            batchX_placeholder: np.array(m,n)
            inputs_series: list(np.array(m,1), self.len = n)
            labels_series: list(np.array(m,1), self.len = n)
    
        Esto se hace para poder iterar de manera más sencilla sobre las columnas de las matrices de los datos de entrada.
        '''

        inputs_series = tf.unstack(batchX_placeholder, axis=1)
        labels_series = tf.unstack(batchY_placeholder, axis=1)
        labels = batchY_placeholder[1,:]



        # Forward Propagation

        current_state = init_state  #Aquí guardamos la salida de la hidden_layer en memoria, para introducirla en la siguiente iteración (RNN)
        states_series = [] #Aquí guardamos todas las salidas para calcular el error

        '''
        Pasamos cada batch de datos a la red, hacemos la propagación directa y guardamos las salidas de la hidden layer para introducirlas en la siguiente iteración. De esta forma obtenemos, del conjunto de iteraciones una lista de arrays(state_size,1).

        Recorremos la lista de la hiden layer aplicándole la output_layer. Esto nos da un vector de batch_size predicciones.
        '''

        # Hidden layer

        for current_input in inputs_series:

    
            current_input = tf.reshape(current_input, [batch_size, 1]) # Esto es porque tiene que tener formato (n,1) y no (n,) para que sume bien con el vector b
            input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Concatenamos a la entrada, las salidas de la salida de la iteración anterior
            next_state = tf.sigmoid(tf.matmul(input_and_state_concatenated, W) + b)  # Suma y activación de los datos de entrada y los datos de salida de la iteración anterior

    
            states_series.append(next_state) #Guardamos la salida pasársela a la capa siguiente.
            current_state = next_state #Guardamos la salida para la iteración siguiente.
    
    
    
    



        # Output layer

        logits_series = [tf.matmul(state, W2) + b2 for state in states_series]  #Comprehension list para iterar sobre todos lo elementos de salida de la capa anterior
        predictions_series = [tf.sigmoid(logits, name = 'predictions_series') for logits in logits_series]








        # Loss Calculation

        '''
        Calculamos la función de perdida para cada uno de las predicciones del batch y luego calculamos el valor medio
        '''

        losses = [tf.losses.mean_squared_error(tf.reshape(labels, [batch_size, 1]),logits) for logits, labels in zip(predictions_series,labels_series)]
        total_loss = tf.reduce_mean(losses)








        # Backpropagation

        train_step = tf.train.AdagradOptimizer(0.003).minimize(total_loss) # gracias google :_)









        # Save the model
        saver = tf.train.Saver()

        #model_inputs = tf.saved_model.utils.build_tensor_info(x)
        #model_outputs_y = tf.saved_model.utils.build_tensor_info(y)
        #model_outputs_z = tf.saved_model.utils.build_tensor_info(z)


        #if tf.gfile.Exists("./model"):
        #         tf.gfile.DeleteRecursively("./model")

        #builder = tf.saved_model.builder.SavedModelBuilder("./model")







        # Execute the model



        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        loss_list = [] #to show the loss decrease





        x,y = train_data[:,0:total_series_length],train_data[:,1:(total_series_length+1)]
        first_current_state = np.zeros((batch_size, state_size)) # Estado inicial cero


        # Iteramos sobre los datos num_epoch veces
        for epoch_idx in range(num_epochs):
            _current_state = first_current_state
    
    
    
        # Separamos los datos en batches    
            for batch_idx in range(num_batches):
                '''
                    starting and ending point per batch
                    since weights reoccuer at every layer through time
                    These layers will not be unrolled to the beginning of time, 
                    that would be too computationally expensive, and are therefore truncated 
                    at a limited number of time-steps
                '''
        
                start_idx = batch_idx * truncated_backprop_length+skip
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:,start_idx:end_idx]
                batchY = y[:,start_idx:end_idx]
            
                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series],
                    feed_dict={
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY,
                            init_state:_current_state
                        })

                loss_list.append(_total_loss)

        #        if batch_idx==(num_batches-1):
        #            print('Epoch',epoch_idx,"Step",batch_idx,"Loss", _total_loss)



        first_current_state = tf.Variable(_current_state, name = 'first_current_state')
        saver.save(sess, "./model/model")



        #builder.add_meta_graph_and_variables(sess,[tag_constants.TRAINING],signature_def_map=foo_signatures,assets_collection=foo_assets)


        
        #return jsonify(np.array(_predictions_series).shape) 
        return jsonify(np.asarray(loss_list).tolist()) #chapuza




if __name__ == '__main__':
	app.run(debug=True)
