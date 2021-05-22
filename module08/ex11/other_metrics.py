def confusion_matrix(y_true, y_hat, label=None):
def accuracy_score_(y, y_hat):
    tp = 

# def precision_score_(y, y_hat, pos_label=1):
# def recall_score_(y, y_hat, pos_label=1):
# def f1_score_(y, y_hat, pos_label=1):
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score   

# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y = np.array([1, 0, 0, 1, 0, 1, 0, 0])

# Accuracy
## your implementation
accuracy_score_(y, y_hat)
## Output:
0.5
## sklearn implementation
accuracy_score(y, y_hat)
## Output:
0.5

# Precision
## your implementation
precision_score_(y, y_hat)
## Output:
0.4
## sklearn implementation
precision_score(y, y_hat)
## Output:
0.4

# Recall
## your implementation
recall_score_(y, y_hat)
## Output:
0.6666666666666666
## sklearn implementation
recall_score(y, y_hat)
## Output:
0.6666666666666666

# F1-score
## your implementation
f1_score_(y, y_hat)
## Output:
0.5
## sklearn implementation
f1_score(y, y_hat)
## Output:
0.5

