from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_frame_getter import get_dataframe
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

df = get_dataframe()

X = df.loc[:, df.columns != "stroke"]
y = df["stroke"]

print(f"Logistic regression score: {cross_val_score(LogisticRegression(), X, y)}")
print(f"SVC score: {cross_val_score(SVC(), X, y)}")
print(f"Random forest score: {cross_val_score(RandomForestClassifier(), X, y)}")
