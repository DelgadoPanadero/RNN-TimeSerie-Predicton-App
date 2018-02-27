import io
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, make_response, send_file, request, jsonify
from main import app


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
        predictions_series = tf.sigmoid([logits for logits in logits_series], name = 'predictions_series')














        # Loss Calculation

        '''
        Calculamos la función de perdida para cada uno de las predicciones del batch y luego calculamos el valor medio
        '''

        losses = tf.losses.mean_squared_error(tf.stack(labels_series, axis = 0),predictions_series[:,:,0])
        total_loss = tf.reduce_mean(losses)










        # Backpropagation

        train_step = tf.train.AdagradOptimizer(0.003).minimize(total_loss) # gracias google :_)











        # Save the model

        saver = tf.train.Saver()












        # Execute the model

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        loss_list = [] 


        x,y = train_data[:,0:total_series_length],train_data[:,1:(total_series_length+1)]
        first_current_state = np.zeros((batch_size, state_size)) # Estado inicial cero


        for epoch_idx in range(num_epochs):        # Iteramos sobre los datos num_epoch veces
            _current_state = first_current_state
   
            for batch_idx in range(num_batches):         # Separamos los datos en batches 
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


        first_current_state = tf.Variable(_current_state, name = 'first_current_state')
        saver.save(sess, "./model/model")


 
        return jsonify(np.asarray(loss_list).tolist())
