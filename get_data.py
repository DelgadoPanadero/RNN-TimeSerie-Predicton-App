import numpy as np
import pandas as pd
from flask import make_response
from main import app


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

        return response
