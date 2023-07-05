import json
import numpy as np
from model import *
from utils import *
pred1,pred2=[],[]



def model_training(X, y, X_test, y_test, config):
    
    XA, XB, XA_test, XB_test = vertically_partition_data(X, X_test, config['A_idx'], config['B_idx'])
    print('XA:',XA.shape, '   XB:',XB.shape)
    
    ## 各参与方的初始化
    Guest_A = GuestA(XA, config)
    print("Guest_A successfully initialized.")
    Host_B = HostB(XB, y, config)
    print("Host_B successfully initialized.")
    Sever_C =  SeverC(XA.shape, XB.shape, config)
    print("Sever_C successfully initialized.")
    
    ## 各参与方之间连接的建立
    Guest_A.connect("B", Host_B)
    Guest_A.connect("C", Sever_C)
    Host_B.connect("A", Guest_A)
    Host_B.connect("C", Sever_C)
    Sever_C.connect("A", Guest_A)
    Sever_C.connect("B", Host_B)
    
    ## 训练
    accuracy_A, accuracy_B = [],[]
    Host_B.task_0()
    for i in range(1,config['n_iter']+1):
        print(f"**********epoch{i}**********")
        Sever_C.task_1("A", "B")
        Guest_A.task_1("B")
        if i%5 == 0:
            Host_B.task_flipback("A")
        else:
            Host_B.task_1("A") 
        Guest_A.task_2("C")
        Host_B.task_2("C")
        Sever_C.task_2("A", "B")
        Guest_A.task_3()
        Host_B.task_3()
        
        ### A做预测
        yA_pred = predict_sigmoid(Guest_A.weights, XA_test, np.zeros_like(XA_test.shape[0]))
        yA_accuracy, yA_precision, yA_recall, yA_f1 = calculate_metrics(y_test, yA_pred)
        print(f"yA_accuracy:{yA_accuracy}, yA_precision:{yA_precision}, yA_recall:{yA_recall}, yA_f1:{yA_f1}")
        accuracy_A.append(yA_accuracy)
        
        ### B做预测
        yB_pred = predict_sigmoid(Host_B.weights, XB_test, np.zeros_like(XB_test.shape[0]))
        yB_accuracy, yB_precision, yB_recall, yB_f1 = calculate_metrics(y_test, yB_pred)
        print(f"yB_accuracy:{yB_accuracy}, yB_precision:{yB_precision}, yB_recall:{yB_recall}, yB_f1:{yB_f1}")
        accuracy_B.append(yB_accuracy)
        
    print("All process done.")
    
    loss_acc_fig(Sever_C.loss, accuracy_A, Sever_C.loss, accuracy_B)
    
    return True

X, y, X_test, y_test = load_data()
config = json.load(open('.\Code\config.json'))
model_training(X, y, X_test, y_test, config)