import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def delete_file(file_name, file_format):  # usuwanie plików przed rysowaniem histogramu i boxplota
    if os.path.exists(file_name + file_format):
        os.remove(file_name + file_format)
    else:
        print('The file ' + file_name + file_format + ' does not exist')


def histograms(file_names, dataframe):  # file_name - nazwy plików, dataframe - tam gdzie mamy dane pobrane przez pandas
    for head in file_names:
        delete_file(head, '_histogram.png')
        plt.hist(dataframe[head])  # plot histogramu z dataframe
        plt.title(head)
        plt.savefig(head + '_histogram.png')
        plt.close()


def boxplots(file_names, dataframe):  # j/w
    for head in file_names:
        delete_file(head, '_boxplots.png')
        plt.boxplot(dataframe[head])
        plt.title(head)
        plt.savefig(head + '_boxplots.png')
        plt.close()


df = pd.read_csv('healthcare-dataset-stroke-data.csv')  # wczytywanie danych z pliku za pomocą pandas dataframe
df = df.fillna(df.mean())

df = pd.read_csv('healthcare-dataset-stroke-data.csv')  # wczytywanie danych z pliku za pomocą pandas dataframe
df = df.fillna(df.mean())  # podkładanie średniej za wszystkie wartości NaN w zbiorze
# df = df.fillna(df.median()) # podkładanie mediany za wszystkie wartości NaN w zbiorze (możemy usunąć więcej tych danych dla lepszego efektu)

headers_boxplot = ['age', 'avg_glucose_level',
                   'bmi']  # tylko te mają wartości liczbowe, reszta to bardziej klasyfikacja dlatego nie da się wyrysować boxplotów
headers_histogram = list(df.columns)
headers_histogram.remove('id')

# histograms(headers_histogram, df)
# boxplots(headers_boxplot, df)

# korelacje
corr1 = df['age'].corr(df[
                           'avg_glucose_level'])  # korelacja może przyjmować wartości -1 do 1, im bliżej 0 tym korelacja między dwoma parametrami jest mniejsza
corr2 = df['age'].corr(df['bmi'])
corr3 = df['bmi'].corr(df['avg_glucose_level'])

print('Correlation age:avg_glucose_level, value: ', corr1)
print('Correlation age:bmi, value: ', corr2)
print('Correlation avg_glucose_level:bmi, value: ', corr3)

# te stringowe wartości można chyba zrzutować na wartości  0,1,2,3... i opisać co jest czym w FAQ, dzięki czemu będzie można te korelacje policzyć dal nich także

# transofrmacja danych, usunięcie id, zmiana wartości tak/nie na binarne itp
df.drop('id', inplace=True, axis=1)

le = preprocessing.LabelEncoder()
le.fit(df['ever_married'])

df['ever_married'] = le.transform(df['ever_married'])

le.fit(df['gender'])
df['gender'] = le.transform(df['gender'])

df = pd.get_dummies(df, columns=['work_type', 'Residence_type', 'smoking_status'])

columns_to_transofrm = ['age', 'avg_glucose_level', 'bmi']

df[columns_to_transofrm] = StandardScaler().fit_transform(df[columns_to_transofrm])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

df.drop(columns=['work_type_children', 'Residence_type_Rural', 'ever_married'], inplace=True)

# work_type_children, Residence_type_Rural, ever_married
