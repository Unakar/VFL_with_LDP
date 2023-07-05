import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    breast = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, random_state=1)
    
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    return X_train, y_train, X_test, y_test

def vertically_partition_data(X, X_test, A_idx, B_idx):
    XA = X[:, A_idx]  
    XB = X[:, B_idx]  
    XB = np.c_[np.ones(X.shape[0]), XB]
    XA_test = X_test[:, A_idx]
    XB_test = X_test[:, B_idx]
    XB_test = np.c_[np.ones(XB_test.shape[0]), XB_test]
    return XA, XB, XA_test, XB_test

# 计算accuracy...
def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall and F1 score
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy, precision, recall, F1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# sigmoid
def predict_sigmoid(weights, X_test, b):
    z = np.dot(weights, X_test.T) + b
    y_pred = 1/(1+np.exp(-z))
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5]= 0
    return y_pred

def loss_acc_fig(loss_A, accuracy_A, loss_B, accuracy_B):
    fig = plt.figure(figsize=(12, 6))
    x=[i+1 for i in range(len(loss_A))]
    plt.subplot(1, 2, 1)
    l1=plt.plot(x,loss_A,'r--',label='loss_A')
    l2=plt.plot(x,accuracy_A,'g--',label='accuracy_A')
    plt.plot(x,loss_A,'ro-',x,accuracy_A,'g+-')
    plt.title('Training and validation accuracy')
    plt.xlabel('n_iter')
    plt.ylabel('loss_A/accuracy_A')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    l3=plt.plot(x,loss_B,'r--',label='loss_B')
    l4=plt.plot(x,accuracy_B,'g--',label='accuracy_B')
    plt.plot(x,loss_B,'ro-',x,accuracy_B,'g+-')
    plt.title('Training and validation accuracy')
    plt.xlabel('n_iter')
    plt.ylabel('loss_B/accuracy_B')
    plt.legend()
    plt.show()