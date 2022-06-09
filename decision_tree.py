from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns

from data_frame_getter import get_dataframe

data = get_dataframe()
stroke_df = data[['stroke']].copy()
data.drop(columns=['stroke'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data, stroke_df, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_leaf=5, max_depth=6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
fn = data.columns.values.tolist()
tree.plot_tree(clf,
               feature_names=fn,
               filled=True)
plt.savefig('decision_tree.pdf')
plt.close()

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
plt.savefig('decision_tree_confusion_matrix.png')
plt.close()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

lr_probs = clf.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
ns_probs = [0 for _ in range(len(lr_probs))]
ns_auc = metrics.roc_auc_score(y_test, ns_probs)
lr_auc = metrics.roc_auc_score(y_test, lr_probs)
# summarize scores
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig('decision_tree_ROC.png')
plt.close()