import numpy as np
import pandas as pd
import cloudpickle
import pickle

from janitor import clean_names

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import streamlit as st

#CONFIGURACION DE LA PÁGINA
st.set_page_config(
     page_title = 'Lead Score Analyzer',
     page_icon = 'risk_score.jpg',
     layout = 'wide')

st.title('LEAD SCORE ANALYZER')

st.sidebar.image('risk_score.jpg', use_container_width=True)
archivo = st.sidebar.file_uploader('Selecciona un archivo csv')



if archivo is not None:
    x = pd.read_csv(archivo, sep=None, engine='python', encoding='latin1',index_col=0)
    
    
    variables_finales = ['campaign',
                     'contactos_anteriores',
                     'duracion',
                     'edad',
                     'educacion',
                     'estado_civil',
                     'resultado_campanas_anteriores',
                     'saldo',
                     'tiempo_transcurrido',
                     'tipo_contacto',
                     'trabajo',
                     'vivienda',
                    'fecha_contacto']
    x = clean_names(x)
    x.drop_duplicates(inplace = True)
    
    y = x['target'].copy()
    y=y.map({'si': 1, 'no': 0})
    x = x[variables_finales].copy()
    
    #Creacion de nuevas variables
    def marcar_nulos(x):
        con_nulos = x.isna().sum() > 0
        x_con_nulos = x.loc[:,con_nulos]
        x_con_nulos = x_con_nulos.apply(lambda variable: np.where(variable.isna(), 1, 0))
        x_con_nulos = x_con_nulos.add_suffix('_nulos')
        return(pd.concat([x,x_con_nulos], axis = 'columns'))

    nulos_flag=marcar_nulos(x[['resultado_campanas_anteriores','tipo_contacto']])
    nulos_flag=nulos_flag.loc[:, nulos_flag.columns[-2:]]
    x = pd.merge(x, nulos_flag, on='ID', how='left')

    x['fecha_contacto'] = pd.to_datetime(x['fecha_contacto'], format='%d-%b-%Y')
    x['mes'] = x['fecha_contacto'].dt.month
    x['dia'] = x['fecha_contacto'].dt.month
    x = x.drop(columns=['fecha_contacto'])
    x['cliente_campaña_anterior'] = np.where(x['contactos_anteriores'] == 0, 0, 1)

    x['cliente_campaña_anterior'] = np.where(x['contactos_anteriores'] == 0, 0, 1)
    
    
    
    #Definir listas
   #categoricas por valor
    var_imputar_valor = ['resultado_campanas_anteriores','tipo_contacto','educacion']
    valores = ['no_campaña_anterior','movil','secundaria/superiores']

       #resto de categoricas
    resto_cat= x.select_dtypes(exclude = 'number').columns.tolist()
    for var in var_imputar_valor:
        while var in resto_cat:  
            resto_cat.remove(var)

       #variables numéricas
    var_imputar_mediana = x.select_dtypes(include = 'number').columns.tolist()
    windsorizar_saldo=['saldo']
    p_min = 0
    p_max = 0.99
    p_min_1 = 0.005
    p_max_1 = 1

        #variables binarias
    variables_binarias=['vivienda']

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
    
    
    
    with open('pipe_ejecucion.pickle', mode='rb') as file:
        pipe_ejecucion = pickle.load(file)
    
    scoring = pipe_ejecucion.predict_proba(x)[:, 1]
    scoring = pd.Series(scoring, name='scoring')
    
    x_id = x.reset_index()
    x_id = pd.concat([x_id['ID'], scoring], axis=1)
    x_id['scoring'] = x_id['scoring'].round(2)
    x_id['ID'] = x_id['ID'].apply(lambda x: f"{x:,}".replace(",", "."))
    x_id = x_id.set_index('ID').sort_values(by='scoring', ascending=False)

    
    st.write("Probabilidades de la clase positiva (1):")
    st.write(x_id)
    
    
else:
    st.stop()
    
    
