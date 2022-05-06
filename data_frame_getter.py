import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def get_dataframe() -> pd.DataFrame:
    dataframe = pd.read_csv(
        'healthcare-dataset-stroke-data.csv')  # wczytywanie danych z pliku za pomocą pandas dataframe
    dataframe = dataframe.fillna(dataframe.mean())
    dataframe.drop('id', inplace=True, axis=1)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(dataframe['ever_married'])
    dataframe['ever_married'] = label_encoder.transform(dataframe['ever_married'])
    label_encoder.fit(dataframe['gender'])
    dataframe['gender'] = label_encoder.transform(dataframe['gender'])
    dataframe = pd.get_dummies(dataframe, columns=['work_type', 'Residence_type', 'smoking_status'])
    columns_to_transofrm = ['age', 'avg_glucose_level', 'bmi']
    dataframe[columns_to_transofrm] = StandardScaler().fit_transform(dataframe[columns_to_transofrm])
    dataframe.drop(columns=['work_type_children', 'Residence_type_Rural', 'ever_married'])
    return dataframe
