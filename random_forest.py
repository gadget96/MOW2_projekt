import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from data_frame_getter import get_dataframe


df2 = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = get_dataframe()

# Oversampling https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
# Za pomocą algorytmu smote (który wykorzystuje KNN) generowanie syntetycznych danych do mniejszej klasy

x = df.loc[:, df.columns != "stroke"]

y = df["stroke"]

smote = SMOTE()

x_smote, y_smote = smote.fit_resample(x, y)

# Podczas trenowania modelu wykorzystany jest nie zbiór oryginalny ale "uzupełniony" o wartości z klasą 'stroke'.
# Dzięki temu model może być trenowany na zbalansowanym zbiorze.


X_train, X_test, y_train, y_test = train_test_split(
    x_smote, y_smote, test_size=0.33, random_state=42
)

model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


lr_probs = model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
ns_probs = [0 for _ in range(len(lr_probs))]
ns_auc = metrics.roc_auc_score(y_test, ns_probs)
lr_auc = metrics.roc_auc_score(y_test, lr_probs)
# summarize scores
print("Logistic: ROC AUC=%.3f" % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
plt.plot(lr_fpr, lr_tpr, marker=".", label="Logistic")
# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# show the legend
plt.legend()
plt.savefig("LogisticRegression_Smote.png")
plt.close()

# Zwykle algorytmmy średnio sobie radzą z daymi z niezbalansowanuych zbiorów.
# Wtedy wykorzystanie drzew dezycyjnych może być lepszym pomysłem.


rfc = RandomForestClassifier()

stroke_df = df[["stroke"]].copy()
df.drop(columns=["stroke"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df, stroke_df, test_size=0.33, random_state=42
)

# fit the predictor and target
rfc.fit(X_train, y_train)

y_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


lr_probs = model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
ns_probs = [0 for _ in range(len(lr_probs))]
ns_auc = metrics.roc_auc_score(y_test, ns_probs)
lr_auc = metrics.roc_auc_score(y_test, lr_probs)
# summarize scores
print("Random forest: ROC AUC=%.3f" % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
plt.plot(lr_fpr, lr_tpr, marker=".", label="Random forest")
# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# show the legend
plt.legend()
plt.savefig("RandomForest.png")
plt.close()
