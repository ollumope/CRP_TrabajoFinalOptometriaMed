# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

#Graficos
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.tree import export_graphviz

#ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# * El dataset inscripción corresponde al listado de personas que se inscriben en el programa "Salud Visual" de la Alcaldía de Medellín el cual busca asignar recursos del presupuesto participativo que se asignaron a la atención visual sobre las personas mayores de 18 años
# * El dataset atención corresponde al listado de pacientes atendidos en el programa 
# * El dataset sintomas corresponde a la información de descripción del dataset inscripcion, las cuales fueron procesadas en Atlas TI para determinar molestias claves

# %%
atencion = pd.read_excel('ATENCIONES_D.xlsx',sheet_name='ATENCIONES', skiprows = 1)
inscripcion = pd.read_excel('ATENCIONES_D.xlsx',sheet_name='INSCRITOS')
sintomas = pd.read_excel('Sintomas.xlsx',sheet_name='sintomas')

# %% [markdown]
# Integración de bases de datos de atención, inscripción y sintomas

# %%
inscritos_sintomas = inscripcion.merge(sintomas, on = ['ID','Ficho_Inscripcion'], how = 'left' )


# %%
inscritos_sintomas.shape


# %%
optometria_raw = atencion.merge(inscritos_sintomas, on = 'ID')


# %%
optometria_raw.shape


# %%
optometria_raw.describe()

# %% [markdown]
# Del análisis preliminar de la información los pacientes atendidos, se depuran las variables no estan categorizadas en función del diagnostico
# 
# Caracteristicas usadas para generar cobro
# 
# * Tipo de lente
# * Total lentes unitarios
# * Valor_usuario
# * Referencia_montura
# * Lentes monofocales en CR-39 terminados
# * Lentes monofocales en CR-39 tallados
# * Lentes monofocales en policarbonato terminados
# * Lentes monofocales en policarbonato tallado
# * Lentes bifocales flat top CR-39 terminados
# * Lentes bifocales flat top CR-39 tallados
# * Lentes bifocales flat top policarbonato
# * Lentes bifocales invisible CR-39 terminados
# * Lentes bifocales invisible CR-39 tallados
# * Lentes bifocales invisible policarbonato
# * Lentes progresivos en CR-39
# * Lentes progresivos en policarbonato
# 
# Carateristicas de identificación
# * Consecutivo. Corresponde al id entregado en la atención y no aporta información al modelo
# * Ficho de inscripcion
# * Direccion Residencia
# * Fecha de nacimiento
# * Tipo de documento
# * Barrio
# * Direccion Correspondencia
# * Sisben 
# * Descripcion
# 
# Caracteristicas de uso administrativo de la Alcaldía de Medellín
# * Poblacion
# * Etnia. 
# * Ocupación. 
# 
# 
# El diagnostico Z010 corresponde a una consulta ....pendiente optometra
# %% [markdown]
# Selección de características a partir del juicio del experto

# %%
columnas = ['EDAD','CONSULTA','ESTUCHE Y PAÑO','MONTURA', 'DX\nPPAL\nCIE 10','ESFERA-OD','CILINDRO-OD','EJE-OD','ADD-OD','ESFERA-OI','CILINDRO-OI','EJE-OI','ADD-OI','Genero','Exam_ojos','Usa_Gafas','Tiem_gafas','Glaucoma','Enfermedad','Picazón_irritación_Ardor', 'Ojos_cansados_doloridos', 'Ojos_llorosos', 'Ojos_secos', 'Visión_borrosa', 'Visión_doble', 'Mayor_sensibilidad_a_la_luz',
'Dificultad_para_ver', 'Dificultad_para_leer_letras_pequeñas', 'Leer_a_una_distancia_mayor_', 'Secreciones_(Pus_o_mucosidad)',      'Leer_a_una_distancia_menor_', 'Ojos_Enrojecidos', 'Mareos_migraña_y_vomito', 'Inflamación', 'Temblor_en_los_ojos',       'Cestelleos_manchas_flotantes_y_desprendimiento_de_la_retina','ID']


# %%
optometria_dep = optometria_raw[columnas]

# %% [markdown]
# Ajuste de forma de encabezado

# %%
optometria_dep.rename(columns={'DX\nPPAL\nCIE 10':'Diagnostico'}, inplace = True)
optometria_dep.rename(str.lower, axis='columns', inplace = True)

# %% [markdown]
# Análisis de los datos

# %%
#Se borran los registros duplicados por id en caso de que existan
optometria_dep.drop_duplicates('id', keep = 'last', inplace = True)


# %%
optometria_dep.shape


# %%
#Se eliminan toda aquellas filas que contengan en todos su campos valores nulos
optometria_dep.dropna(how = 'all', inplace = True)


# %%
optometria_dep.shape

# %% [markdown]
# EDAD
# La edades oscilan entre los 19 a 81 anios, teniendo unos valores atipicos atipicos por encima de 81 anios y de cero anios.
# En la fuente de datos, la edad más frecuente es 59

# %%
optometria_dep['edad'].describe()


# %%
optometria_dep['edad'].hist() 


# %%
optometria_dep.boxplot('edad')


# %%
optometria_dep[pd.isnull(optometria_dep['edad'])]

# %% [markdown]
# Se revisan los outliers de edad. El programa "Salud Visual" esta diseniado para mayores de edad, por lo tanto se restringuen los valores del data frame bajo este criterio

# %%
optometria_dep.drop(optometria_dep[optometria_dep['edad'] < 18].index, axis = 0, inplace = True)


# %%
optometria_dep.shape

# %% [markdown]
# CONSULTA. La consulta toma el valor de 1, lo cual quiere decir que fue atendido, dado que hay poca varibilidad se considera candidata a eliminar

# %%
optometria_dep['consulta'].describe()


# %%
optometria_dep['consulta'].hist() 


# %%
optometria_dep['estuche y paño'].hist()


# %%
optometria_dep['estuche y paño'].unique()

# %% [markdown]
# Imputacón de nulos por 0

# %%
values_em = {'estuche y paño': 0, 'montura' : 0, 'esfera-od':0, 'cilindro-od':0, 'eje-od':0, 'add-od':0, 'esfera-oi':0,
       'cilindro-oi':0, 'eje-oi':0, 'add-oi':0, 'picazón_irritación_ardor':0, 'ojos_cansados_doloridos':0, 'ojos_llorosos':0,
       'ojos_secos':0, 'visión_borrosa':0, 'visión_doble':0, 'mayor_sensibilidad_a_la_luz':0, 'dificultad_para_ver':0,
       'dificultad_para_leer_letras_pequeñas':0, 'leer_a_una_distancia_mayor_':0, 'secreciones_(pus_o_mucosidad)':0, 
       'leer_a_una_distancia_menor_':0, 'ojos_enrojecidos':0, 'mareos_migraña_y_vomito':0, 'inflamación':0,
       'temblor_en_los_ojos':0,'cestelleos_manchas_flotantes_y_desprendimiento_de_la_retina':0} 


# %%
optometria_dep.fillna(value = values_em, inplace = True)


# %%
optometria_dep['montura'].hist()


# %%
optometria_dep.groupby(['montura'])['montura'].count()


# %%
optometria_dep.groupby(['estuche y paño'])['estuche y paño'].count()


# %% [markdown]
#Las variables estuche y paño y montura indican que al paciente le recetaron gafas, por lo tanto se puede agrupar en usa sola dimensión.


# %%
optometria_dep['requiere_gafas'] = np.where((optometria_dep['estuche y paño'] == 1) & (optometria_dep['montura'] == 1),1,0) #1 Tiene gafas, 0 no tiene
optometria_dep['requiere_gafas'].hist()

# %% [markdown]
# Los diagnosticos con mayor frecuencia en el estudio realizado en el 2019 son H524 - Presbicia, H522 - Astigmatismo, H521 - Miopía. El diagnostico Z010 significa que el paciente fue diagnosticado sin enfermedad visual.

# %%
optometria_dep['diagnostico'] = optometria_dep['diagnostico'].str.upper()
optometria_dep['diagnostico'].hist(figsize = (17,5))


# %%
optometria_dep['diagnostico'].unique()

# %% [markdown]
# Se depuran los registros que no tienen diagnostico, en caso de que existan

# %%
optometria_dep.drop(optometria_dep[optometria_dep['diagnostico'].isnull()].index, axis = 0, inplace = True)


# %%
optometria_dep.shape

# %% [markdown]
# Análisis de medidas visuales
# %% [markdown]
# * El valor de balance se refiere a cuando un ojo no recupera visión, el optometra le asigna una fórmula similar a la del otro ojo 
# * N significa neutro y toma valor a cero
# * Los valores impracticable y no aplica significan que el examen por alguna razón no se puedo prácticar, se eliminan los registros que tomen ese valor del todo el dataset

# %%
optometria_dep['esfera-od'].unique()


# %%
optometria_dep['esfera-od'] = np.where((optometria_dep['esfera-od']  == 'N') | (optometria_dep['esfera-od']  == 'n') | (optometria_dep['esfera-od']  == ' N') | (optometria_dep['esfera-od']  == 'N ') | (optometria_dep['esfera-od']  == ' n') | (optometria_dep['esfera-od']  == 'n '), 0.0, optometria_dep['esfera-od'])
optometria_dep['esfera-od'] = np.where((optometria_dep['esfera-od']  == 'BALANCE') | (optometria_dep['esfera-od']  == 'BALANCE '), optometria_dep['esfera-oi'], optometria_dep['esfera-od']) 


# %%
optometria_dep.drop((optometria_dep[optometria_dep['esfera-od'] == 'IMPRACTICABLE'].index) | (optometria_dep[optometria_dep['esfera-od'] == 'IMPRACTICABLE '].index) | (optometria_dep[optometria_dep['esfera-od'] == 'NO APLICA '].index) | (optometria_dep[optometria_dep['esfera-od'] == 'NO APLICA '].index), axis = 0, inplace = True)


# %%
optometria_dep['cilindro-od'].unique()


# %%
optometria_dep['cilindro-od'] = np.where((optometria_dep['cilindro-od']  == 'BALANCE') | (optometria_dep['cilindro-od']  == 'BALANCE '), optometria_dep['cilindro-oi'], optometria_dep['cilindro-od']) 
optometria_dep['cilindro-od'] = np.where((optometria_dep['cilindro-od'] == ' '), np.nan, optometria_dep['cilindro-od'])


# %%
optometria_dep['add-od'].unique()


# %%
#optometria_dep['add-od'] = optometria_dep['add-od'].str.strip()
optometria_dep['add-od'] = np.where((optometria_dep['add-od'] == '   ') | (optometria_dep['add-od'] == ' '),0.0,optometria_dep['add-od'])


# %%
optometria_dep['esfera-oi'].unique()


# %%
optometria_dep['esfera-oi'] = np.where((optometria_dep['esfera-oi']  == 'N') | (optometria_dep['esfera-oi']  == ' N') | (optometria_dep['esfera-oi']  == 'N ') | (optometria_dep['esfera-oi']  == 'n') | (optometria_dep['esfera-oi']  == ' n') | (optometria_dep['esfera-oi']  == 'n '), 0.0, optometria_dep['esfera-oi'])

optometria_dep.drop((optometria_dep[optometria_dep['esfera-oi'] == 'NO APLICA '].index) | (optometria_dep[optometria_dep['esfera-oi'] == 'NO APLICA'].index), axis = 0, inplace = True)


# %%
optometria_dep['cilindro-oi'].unique() 


# %%
optometria_dep['cilindro-oi'] = np.where((optometria_dep['cilindro-oi']  == 'N'), 0.0, optometria_dep['cilindro-oi'])

optometria_dep['cilindro-oi'] = np.where((optometria_dep['cilindro-oi']  == 'BALANCE') | (optometria_dep['cilindro-oi']  == 'BALANCE '), optometria_dep['cilindro-od'], optometria_dep['cilindro-oi']) 


# %%
optometria_dep['eje-od'].unique()


# %%
optometria_dep['eje-oi'].unique()


# %%
optometria_dep['eje-oi'] = np.where((optometria_dep['eje-oi'] == 'O'), 0.0, optometria_dep['add-oi'])
optometria_dep['eje-oi'] = np.where((optometria_dep['eje-oi'] == ' '), 0.0, optometria_dep['add-oi'])


# %%
optometria_dep['add-oi'].unique() 


# %%
optometria_dep['add-oi'] = np.where((optometria_dep['add-oi'] == ' '), 0.0, optometria_dep['add-oi'])


# %%
optometria_dep['genero'].unique() 


# %%
optometria_dep['genero'] = optometria_dep['genero'].str.strip() #Revisar nulos
dfDummiesgenero = pd.get_dummies(optometria_dep['genero'], prefix = 'genero')
optometria_dep = pd.concat([optometria_dep, dfDummiesgenero], axis=1)


# %%
optometria_dep['usa_gafas'].unique()


# %%
optometria_dep['usa_gafas'] = optometria_dep['usa_gafas'].str.upper()
optometria_dep['usa_gafas'] = np.where(optometria_dep['usa_gafas'].isnull(), 'NO', optometria_dep['usa_gafas'])


# %%
#Codificar valores de si una gafas o no
dfDummiesUsaGafas = pd.get_dummies(optometria_dep['usa_gafas'], prefix = 'usa_gafas')
optometria_dep = pd.concat([optometria_dep, dfDummiesUsaGafas], axis=1)


# %%
#Se ajuste el nombre del glaucoma de la bd de inscritos, ya que corresponde a la pregunta: Alguien en su familia sufre de glaucoma?
optometria_dep.rename(columns = {'glaucoma':'glaucoma_flia'}, inplace = True)


# %%
optometria_dep['glaucoma_flia'].unique()  #Todo valor vacio significa que no tiene glaucoma


# %%
optometria_dep['glaucoma_flia'] = optometria_dep['glaucoma_flia'].str.upper()
optometria_dep['glaucoma_flia'] = np.where(optometria_dep['glaucoma_flia'].isnull(), 0, optometria_dep['glaucoma_flia'])
optometria_dep['glaucoma_flia'] = np.where(optometria_dep['glaucoma_flia'] == 'SI', 1, 0)


# %%
optometria_dep['enfermedad'].unique() #Esta columna reune las enfermedades de Hipertension, Glaucoma, Diabetes

# %% [markdown]
# Derivación a partir de la columna enfermedad

# %%
optometria_dep['hipertension'] = optometria_dep['enfermedad'].map(lambda x: str(x).upper().find('H') != -1)
optometria_dep['diabetes'] = optometria_dep['enfermedad'].map(lambda x: str(x).upper().find('D') != -1)
optometria_dep['glaucoma'] = optometria_dep['enfermedad'].map(lambda x: str(x).upper().find('G') != -1)


# %%
optometria_dep['exam_ojos'].unique() #Si esta vacio es porque nunca se ha realizado un examen de ojos


# %%
optometria_dep['exam_ojos'] = optometria_dep['exam_ojos'].str.upper()
optometria_dep['exam_ojos'] = np.where(optometria_dep['exam_ojos'].isnull(), 'NUNCA', optometria_dep['exam_ojos'])


# %%
#Codificar valores de examen de los ojos
dfDummiesExam = pd.get_dummies(optometria_dep['exam_ojos'], prefix = 'exam_ojos')
optometria_dep = pd.concat([optometria_dep, dfDummiesExam], axis=1)


# %%
optometria_dep['tiem_gafas'].unique()


# %%
optometria_dep['tiem_gafas'] = optometria_dep['tiem_gafas'].str.upper()
optometria_dep['tiem_gafas'] = np.where(optometria_dep['tiem_gafas'].isnull(), 'NO', optometria_dep['tiem_gafas'])


# %%
#Codificar valores de examen de los ojos
dfDummiesGafas = pd.get_dummies(optometria_dep['tiem_gafas'], prefix = 'tiem_gafas')
optometria_dep = pd.concat([optometria_dep, dfDummiesGafas], axis=1)

# %% [markdown]
# Luego de realizar el análisis de cada una de las caracteristicas y realizar ajustes de acuerdo a las definiones del negocios, se revisa el estado de los valores faltantes en el dataset

# %%
optometria_dep.isna().sum()

  # %% [markdown]
# Análisis de la distribución de los pacientes de acuerdo a la información rescatada del verbatim de descripción. Siendo:
# 
# * 0. Nunca ha tenido el síntoma
# * 1. Ha tenido el síntoma

# %%
optometria_dep[['picazón_irritación_ardor', 'ojos_cansados_doloridos', 'ojos_llorosos', 'ojos_secos', 'visión_borrosa', 'visión_doble',  'mayor_sensibilidad_a_la_luz', 'dificultad_para_ver', 'dificultad_para_leer_letras_pequeñas', 'leer_a_una_distancia_mayor_', 'secreciones_(pus_o_mucosidad)', 'leer_a_una_distancia_menor_', 'ojos_enrojecidos', 'mareos_migraña_y_vomito', 'inflamación', 'temblor_en_los_ojos',        'cestelleos_manchas_flotantes_y_desprendimiento_de_la_retina']].hist(alpha = 0.5, figsize = (14,12))

# %% [markdown]
# REDUCCION DE CARACTERISTICAS 
# 
# Una vez se ha revisado la información contenida en cada una de las columnas del dataframe y aplicado limpieza sobre ellas, se procede a realizar análisis de las dimensiones
# 
# De la limpieza de datos se concluye:
# 
# - La variable Enfermedad se abre en tres variables:
# 
#   * Hipertensión
#   * Glaucoma
#   * Diabetes
# 
# - Todos los valores 'na' de las mediciones no pueden ser interpretados como valores cero
# 
# - Dado que el campo consulta indica si recibio consulta o no y para los datos del dataframe no hay varibilidad mostrada con una desviación estandar igual a cero, esta columna puede ser eliminada
# 
# - Las columnas de montura y estuche y paño juntas significa que una persona luego de ser atendido por el optómetra le fueron recetados lentes, esta información se consolida en requiere_gafas
# 
# - Las caracteristicas genero, usa_gafas, exam_ojos, tiem_gafas, pueden ser eliminadas ya que fueron numerizadas.

# %%
optometria_dep.drop(['enfermedad','consulta','montura', 'estuche y paño', 'genero', 'usa_gafas', 'exam_ojos', 'tiem_gafas', 'id'], axis = 1, inplace = True)

# %% [markdown]
# Construcción de la variable objetivo

# %%
optometria_dep['astigmatismo'] = np.where(optometria_dep['diagnostico'] == 'H522',1,0) #1 Tiene astigmatismo, 0 No
#H522 astigmatismo
#H521 miopia
#H524 presbicia 


# %%
optometria_dep['astigmatismo'].hist(alpha = 0.5)

# %% [markdown]
# Las clases estan desbalanceadas, es importante al aplicar un modelo de machine learning balancear las clases

# %%
#Se elimina la caracteriticas diagnostico ya que redunda con la variable objetivo
optometria_dep.drop(['diagnostico'], axis = 1, inplace = True)

# %% [markdown]
# Dado que varias de las variables de medición óptica esan de tipo object y para efectos del balanceo deben ser númericas, se hace la conversión

# %%
optometria_dep['esfera-od'] = optometria_dep['esfera-od'].astype(float)
optometria_dep['cilindro-od'] = optometria_dep['cilindro-od'].astype(float)
optometria_dep['eje-od'] = optometria_dep['cilindro-od'].astype(float)
optometria_dep['add-od'] = optometria_dep['add-od'].astype(float)
optometria_dep['esfera-oi'] = optometria_dep['esfera-oi'].astype(float)
optometria_dep['cilindro-oi'] = optometria_dep['cilindro-oi'].astype(float)
optometria_dep['eje-oi'] = optometria_dep['eje-oi'].astype(float)
optometria_dep['add-oi'] = optometria_dep['add-oi'].astype(float)

# %% [markdown]
# Existen tipos de lentes de acuerdo a las medidas tomadas en la revisión:
# - Esfericos Monofocal. Tienen valores en las esferas (EM)
# - Esferocilindricos Monofocal. Tiene esfera, cilindro y eje (ECM)
# - Esfericos Bifocal. Tiene esfera y adición (EB)
# - Esferocilindricos Bifocal. Tienen esfera, cilindro, eje y adición (ECB)

# %%
optometria_dep['lente_EM_oi'] = np.where((optometria_dep['esfera-oi'].notna()) & (np.isnan(optometria_dep['cilindro-oi'])) & (np.isnan(optometria_dep['eje-oi'])) & (np.isnan(optometria_dep['add-oi'])),1,0)

optometria_dep['lente_ECM_oi'] = np.where((optometria_dep['esfera-oi'].notna()) & (optometria_dep['cilindro-oi'].notna()) & (optometria_dep['eje-oi'].notna()) & (np.isnan(optometria_dep['add-oi'])),1,0)

optometria_dep['lente_EB_oi'] = np.where((optometria_dep['esfera-oi'].notna()) & (np.isnan(optometria_dep['cilindro-oi'])) & (np.isnan(optometria_dep['eje-oi'])) & (optometria_dep['add-oi'].notna()),1,0)

optometria_dep['lente_ECB_oi'] = np.where((optometria_dep['esfera-oi'].notna()) & (optometria_dep['cilindro-oi'].notna()) & (optometria_dep['eje-oi'].notna()) & (optometria_dep['add-oi'].notna()),1,0)


# %%
optometria_dep['lente_EM_od'] = np.where((optometria_dep['esfera-od'].notna()) & (np.isnan(optometria_dep['cilindro-od'])) & (np.isnan(optometria_dep['eje-od'])) & (np.isnan(optometria_dep['add-od'])),1,0)

optometria_dep['lente_ECM_od'] = np.where((optometria_dep['esfera-od'].notna()) & (optometria_dep['cilindro-od'].notna()) & (optometria_dep['eje-od'].notna()) & (np.isnan(optometria_dep['add-od'])),1,0)

optometria_dep['lente_EB_od'] = np.where((optometria_dep['esfera-od'].notna()) & (np.isnan(optometria_dep['cilindro-od'])) & (np.isnan(optometria_dep['eje-od'])) & (optometria_dep['add-od'].notna()),1,0)

optometria_dep['lente_ECB_od'] = np.where((optometria_dep['esfera-od'].notna()) & (optometria_dep['cilindro-od'].notna()) & (optometria_dep['eje-od'].notna()) & (optometria_dep['add-od'].notna()),1,0)

# %% [markdown]
# Análisis descriptivo de las variables una vez realizado la seleccón de caracteristicas

# %%
matriz_corr = optometria_dep.corr()
mask = np.triu(np.ones_like(matriz_corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(14, 11))
cmap = sn.diverging_palette(20, 220, n = 200)
sn.heatmap(matriz_corr, mask = mask, cmap=cmap, vmin = -1, vmax = 1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# %% [markdown]
# **Conclusiones matriz de correlación**
# 
# Correlaciones positivas
# - edad y add-oi
# - edad y add-od
# - edad y eje-oi
# - esfera-oi y esfera-od
# - eje-od y cilindro-od
# - add-od y eje-oi
# - add-od y add-oi
# - add-oi y eje-oi
# - lente_EBC_od y lente_EBC_oi
# 
# 
# Correlaciones negativas
# - tiem_gafas_MAS DE 2 AÑOS y tiem_gafas_NO
# - genero_H y genero_M
# - usa_gafas_NO y usa_gafas_SI


# %%
sn.pairplot(optometria_dep[['edad', 'esfera-od', 'cilindro-od', 'eje-od', 'add-od', 'esfera-oi',
       'cilindro-oi', 'eje-oi', 'add-oi', 'glaucoma_flia', 'astigmatismo']], hue = 'astigmatismo')


#%% [markdown]
#El glaucoma familiar no describe el astigmatismo, ni las esferas , por lo tanto se pueden eliminar del dataset

# %%
sn.pairplot(optometria_dep[['picazón_irritación_ardor', 'ojos_cansados_doloridos', 'ojos_llorosos',
       'ojos_secos', 'visión_borrosa', 'visión_doble', 'mayor_sensibilidad_a_la_luz', 'dificultad_para_ver',
       'dificultad_para_leer_letras_pequeñas', 'leer_a_una_distancia_mayor_', 'astigmatismo']], hue = 
       'astigmatismo')
#%% [markdown]
# La sintomatología anterior no describe el astigmatimo

# %% 
sn.pairplot(optometria_dep[['secreciones_(pus_o_mucosidad)', 
       'leer_a_una_distancia_menor_', 'ojos_enrojecidos', 'mareos_migraña_y_vomito', 'inflamación',
       'temblor_en_los_ojos', 'cestelleos_manchas_flotantes_y_desprendimiento_de_la_retina', 'astigmatismo']], hue = 
       'astigmatismo')

#%%
sn.pairplot(optometria_dep[['requiere_gafas', 'genero_H', 'genero_M', 'usa_gafas_NO',
       'usa_gafas_SI', 'hipertension', 'diabetes', 'glaucoma',
       'exam_ojos_DE 1 A 2 AÑOS', 'exam_ojos_MAS DE 2 AÑOS','astigmatismo']], hue = 
       'astigmatismo')

#%%
sn.pairplot(optometria_dep[['exam_ojos_MENOS DE 1 AÑO', 'exam_ojos_MENOS DE 1 AÑO ',
       'exam_ojos_NUNCA', 'tiem_gafas_DE 1 A 2 AÑOS',
       'tiem_gafas_MAS DE  2 AÑOS', 'tiem_gafas_MAS DE 2 AÑOS',
       'tiem_gafas_MENOS DE 1 AÑO', 'tiem_gafas_NO', 'astigmatismo']], hue = 
       'astigmatismo')


# %%
columnas = ['edad', 'cilindro-od', 'eje-od', 'add-od', 'esfera-oi', 'cilindro-oi', 'eje-oi', 'add-oi', 'tiem_gafas_MAS DE 2 AÑOS', 'exam_ojos_MENOS DE 1 AÑO ', 'astigmatismo']
#'ojos_enrojecidos', 'leer_a_una_distancia_mayor_'

# %%
optometria = optometria_dep[columnas]


# %%
matriz_corr_new = optometria.corr()
plt.figure(figsize=(12, 9))
ax = sn.heatmap(matriz_corr_new, vmin = -1, vmax = 1, center = 0, cmap = sn.diverging_palette(20, 220, n = 200), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

# %% [markdown]
# Luego de seleccionar solo las caracteristicas ......


# %% [markdown]
# Extraccón de caracteristicas usando:
# 
# - Árbol de decisión
# - Bosque aleatorio
# - PCA
# %% [markdown]
# División de datos de entrenamiento y prueba

# %%
modelos_vbles = optometria.drop('astigmatismo', axis = 1)
X = modelos_vbles
y = optometria.astigmatismo
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# %% [markdown]
# Balanceo de Clases
# %% [markdown]
# Evidentemente las clases estan desbalanceadas

# %%
y_train.hist()


# %%
sm = SMOTE(random_state = 0)
x_est, y_est = sm.fit_sample(x_train, y_train)


# %% [markdown]
#Revisemos como estan las clases luego de aplicar el balance a través de SMOTE


# %%
y_est.hist()

# %% [markdown]
# Árbol de decisiones

# %%
arbol = DecisionTreeClassifier(random_state=0)
arbol.fit(x_train, y_train)
y_pred = arbol.predict(x_test)


# %%
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# %%
mask = arbol.feature_importances_ > 0.09
reduced_X = X.loc[:,mask]
print(reduced_X.columns)


# %%
print(dict(zip(X.columns, arbol.feature_importances_.round(2))))

# %% [markdown]
# Bosque Aleatorio

# %%
rf = RandomForestClassifier(random_state=0)
rf.fit(x_est, y_est)
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# %% [markdown]
# PCA

# %%
#Escalar los datos
x = StandardScaler().fit_transform(X)


# %% [markdown]
#Se revisa que la normalización de los datos tengan media cero y desviación estandar de uno


# %%
print('Media: ', np.mean(x))
print('Desviación estándar: ', np.std(x))


# %%
pca_opt = PCA(n_components = 2)
principalComponents = pca_opt.fit_transform(x)


# %%
principalComponents.shape


# %%
optometria_PCA = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2'])


# %%
optometria_PCA.head(2)


# %%
explained_variance = np.var(principalComponents, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
#print('Varianza: ', explained_variance)
print('Ratio de varianza ',explained_variance_ratio)

# %% [markdown]
# El componente 1 explica el 60% de la información mientras que el componente 2 contiene 39% de la información. Proyectar la data a dos dimensiones se pierde casi el 1% de información.

# %%
Xax = principalComponents[:,0]
Yax = principalComponents[:,1]
labels = y
cdict = {0:'red',1:'green'}
labl = {0:'No Astigmatismo',1:'Astigmatismo'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}
fig,ax = plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40, label=labl[l],marker=marker[l],alpha=alpha[l])
# for loop ends
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()

# %%
principalComponents.components_

# %% [markdown]
# Por árboles de decisión y random forest se eliminan las variables exam_ojos_MENOS DE 1 AÑO ya que su factor de importancia es cero 