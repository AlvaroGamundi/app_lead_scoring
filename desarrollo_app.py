import numpy as np
import pandas as pd
import cloudpickle

from janitor import clean_names

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline







def ejecutar_modelos(x):
    #Función calidad de datos
    def calidad_datos(x):
        # primero las categoricas imputacion de nulos
        for variable, valor in zip(var_imputar_valor, valores):
            x[variable] = x[variable].fillna(valor)

        for variable in resto_cat:
                if pd.api.types.is_integer_dtype(x[variable]):

                    x[variable] = x[variable].fillna(int(x[variable].mode()[0]))
                else:
                    x[variable] = x[variable].fillna(x[variable].mode()[0])

        #Luego las numéricas imputacion de nulos y atípicos
        for variable in var_imputar_mediana:
            if pd.api.types.is_integer_dtype(x[variable]):
                x[variable] = x[variable].fillna(int(x[variable].median()))  
            else:
                x[variable] = x[variable].fillna(x[variable].median()) 

        #windsorizacion 1        
        for variable in var_imputar_mediana:

            lower_quantile = x[variable].quantile(p_min)
            upper_quantile = x[variable].quantile(p_max)

            lower_quantile = lower_quantile.iloc[0] if isinstance(lower_quantile, pd.Series) else lower_quantile
            upper_quantile = upper_quantile.iloc[0] if isinstance(upper_quantile, pd.Series) else upper_quantile

            lower_quantile = int(round(lower_quantile))
            upper_quantile = int(round(upper_quantile))

            x[variable] = x[variable].clip(lower=lower_quantile, upper=upper_quantile)

        #windsorizacion 2    
        for variable in windsorizar_saldo:

            lower_quantile = x[variable].quantile(p_min_1)
            upper_quantile = x[variable].quantile(p_max_1)

            lower_quantile = lower_quantile.iloc[0] if isinstance(lower_quantile, pd.Series) else lower_quantile
            upper_quantile = upper_quantile.iloc[0] if isinstance(upper_quantile, pd.Series) else upper_quantile

            lower_quantile = int(round(lower_quantile))
            upper_quantile = int(round(upper_quantile))

            x[variable] = x[variable].clip(lower=lower_quantile, upper=upper_quantile)

        #convertimos variables cat binarias a variables num 0/1
        for variable in variables_binarias:

            x[variable] = np.where(x[variable] == 'si', 1, 0)


        return(x)
    
    
#      with open('pipe_ejecucion_pd.pickle', mode='rb') as file:
#        pipe_ejecucion_pd = pickle.load(file)
