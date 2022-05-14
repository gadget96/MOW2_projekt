from data_frame_getter import get_dataframe
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = get_dataframe()

stroke_df = df[['stroke']].copy()
df.drop(columns=['stroke'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, stroke_df, test_size=0.3, random_state=65)

model = MLPClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

plt.show()
