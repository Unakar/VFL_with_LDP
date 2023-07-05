from phe import paillier
from utils import *
import math
import numpy as np
import json

config = json.load(open('.\Code\config.json'))


class Client:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.other_client = {}
    
    def connect(self, other, target_client):
        self.other_client[other] = target_client
    
    def communicate(self, data, target_client):
        target_client.data.update(data)
    
class HostB(Client):
    def __init__(self, X, y, config):
        super().__init__(config)
        self.X = X
        self.y = y
        self.weights = np.random.normal(loc=0,scale=1.0/15,size=X.shape[1])
        self.data = {}

    #flip labels
    def flip_labels(self):
        flip_prob = 1/(1+math.exp(config['epsilon']))
        num_flips = int(flip_prob * len(self.y))
        flip_indices = np.random.choice(len(self.y), num_flips, replace=False)
        self.y[flip_indices] = 1 - self.y[flip_indices]
    

    def compute_u_b(self):
        WX_b = np.dot(self.X, self.weights)
        u_b = 0.25 * WX_b - self.y + 0.5
        return WX_b, u_b

    def compute_encrypted_dL_b(self, encrypted_u):
        encrypted_dL_b = self.X.T.dot(encrypted_u) + self.config['lambda'] * self.weights
        return encrypted_dL_b

    def update_weight(self, dL_b):
        self.weights = self.weights - self.config["lr"] * dL_b / len(self.X)
    
    def task_0(self):
        try:
            self.flip_labels()
            print("Random process of flipping labels is done.")
        except Exception as e:
            print("Wrong 0 in B: %s" % e)
        
    ## B: step2
    def task_1(self, Guest_A_name):
        try:
            dt = self.data
            assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
            public_key = dt['public_key']
        except Exception as e:
            print("B step 1 exception: %s" % e)
        try:
            WX_b, u_b = self.compute_u_b()
            encrypted_u_b = np.asarray([public_key.encrypt(x) for x in u_b])
            dt.update({"encrypted_u_b": encrypted_u_b})
            dt.update({"WX_b": WX_b})
        except Exception as e:
            print("Wrong 1 in B: %s" % e)

        data_to_A= {"encrypted_u_b": encrypted_u_b}
        self.communicate(data_to_A, self.other_client[Guest_A_name])
	
    ## B: step3、4
    def task_2(self,Sever_C_name):
        try:
            dt = self.data
            assert "encrypted_u_a" in dt.keys(), "Error: 'encrypt_u_a' from A in step 1 not successfully received."
            encrypted_u_a = dt['encrypted_u_a']
            encrypted_u = encrypted_u_a + dt['encrypted_u_b']
            encrypted_dL_b = self.compute_encrypted_dL_b(encrypted_u)
            mask = np.random.rand(len(encrypted_dL_b))
            encrypted_masked_dL_b = encrypted_dL_b + mask
            dt.update({"mask": mask})
        except Exception as e:
            print("B step 2 exception: %s" % e)
        try:
            assert "encrypted_WX_a_square" in dt.keys(), "Error: 'encrypted_WX_a_square' from A in step 1 not successfully received."
            encrypted_z = 4*encrypted_u_a + dt['WX_b']
            encrypted_loss = np.sum((0.5-self.y)*encrypted_z + 0.125*dt["encrypted_WX_a_square"] + 0.125*dt["WX_b"] * (encrypted_z+4*encrypted_u_a))
        except Exception as e:
            print("B step 2 exception: %s" % e)
        data_to_C = {"encrypted_masked_dL_b": encrypted_masked_dL_b, "encrypted_loss": encrypted_loss}
        self.communicate(data_to_C, self.other_client[Sever_C_name])
	
    ## B: step6
    def task_3(self):
        try:
            dt = self.data
            assert "masked_dL_b" in dt.keys(), "Error: 'masked_dL_b' from C in step 2 not successfully received."
            masked_dL_b = dt['masked_dL_b']
            dL_b = masked_dL_b - dt['mask']
            self.update_weight(dL_b)
        except Exception as e:
            print("A step 3 exception: %s" % e)
        print(f"B weight: {self.weights}")
        return



    def task_flipback(self,Guest_A_name):
        try:
            dt = self.data
            assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
            public_key = dt['public_key']
        except Exception as e:
            print("B step 1 exception: %s" % e)
        try:
            yB_pred = predict_sigmoid(self.weights, self.X, np.zeros_like(self.X.shape[0]))
            #y_dist记为yB_pred和真实的y之间的差距，选出其中差距最大的前2%的样本，将其标签翻转
            y_dist = np.abs(yB_pred - self.y)
            flip_indices = np.argsort(y_dist)[-int(0.02*len(y_dist)):]
            self.y[flip_indices] = 1 - self.y[flip_indices]

            WX_b, u_b = self.compute_u_b()
            encrypted_u_b = np.asarray([public_key.encrypt(x) for x in u_b])
            dt.update({"encrypted_u_b": encrypted_u_b})
            dt.update({"WX_b": WX_b})
        except Exception as e:
            print("Wrong 1 in B: %s" % e)

        data_to_A= {"encrypted_u_b": encrypted_u_b}
        self.communicate(data_to_A, self.other_client[Guest_A_name])
    



class GuestA(Client):
    def __init__(self, X, config):
        super().__init__(config)
        self.X = X
        self.weights = np.random.normal(loc=0,scale=1.0/15,size=X.shape[1])#
        
    def compute_WX_a(self):
        WX_a = np.dot(self.X, self.weights)
        return WX_a
    
	## 加密梯度的计算，对应step4
    def compute_encrypted_dL_a(self, encrypted_u):
        encrypted_dL_a = self.X.T.dot(encrypted_u) + self.config['lambda'] * self.weights
        return encrypted_dL_a
    
	##参数的更新
    def update_weight(self, dL_a):
        self.weights = self.weights - self.config["lr"] * dL_a / len(self.X)
        return

    ## A: step2
    def task_1(self, Host_B_name):
        dt = self.data
        assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
        public_key = dt['public_key']
        WX_a = self.compute_WX_a()
        u_a = 0.25 * WX_a
        WX_a_square = WX_a ** 2
        encrypted_u_a = np.asarray([public_key.encrypt(x) for x in u_a])
        encrypted_WX_a_square = np.asarray([public_key.encrypt(x) for x in WX_a_square])
        dt.update({"encrypted_u_a": encrypted_u_a})
        data_to_B = {"encrypted_u_a": encrypted_u_a, "encrypted_WX_a_square": encrypted_WX_a_square}
        self.communicate(data_to_B, self.other_client[Host_B_name])
    
    ## A: step3、4
    def task_2(self, Sever_C_name):
        dt = self.data
        assert "encrypted_u_b" in dt.keys(), "Error: 'encrypted_u_b' from B in step 1 not successfully received."
        encrypted_u_b = dt['encrypted_u_b']
        encrypted_u = encrypted_u_b + dt['encrypted_u_a']
        encrypted_dL_a = self.compute_encrypted_dL_a(encrypted_u)
        mask = np.random.rand(len(encrypted_dL_a))
        encrypted_masked_dL_a = encrypted_dL_a + mask
        dt.update({"mask": mask})
        data_to_C = {'encrypted_masked_dL_a': encrypted_masked_dL_a}
        self.communicate(data_to_C, self.other_client[Sever_C_name])
       
    ## A: step6
    def task_3(self):
        dt = self.data
        assert "masked_dL_a" in dt.keys(), "Error: 'masked_dL_a' from C in step 2 not successfully received."
        masked_dL_a = dt['masked_dL_a']
        dL_a = masked_dL_a - dt['mask']
        self.update_weight(dL_a)
        print(f"A weight: {self.weights}")
        return




class SeverC(Client):


    def __init__(self, A_d_shape, B_d_shape, config):
        super().__init__(config)
        self.A_data_shape = A_d_shape
        self.B_data_shape = B_d_shape
        self.public_key = None
        self.private_key = None
        self.loss = []
	
    ## C: step1
    def task_1(self, Guest_A_name, Host_B_name):
        try:
            public_key, private_key = paillier.generate_paillier_keypair()
            self.public_key = public_key
            self.private_key = private_key
        except Exception as e:
            print("C step 1 error 1: %s" % e)

        data_to_AB = {"public_key": public_key}
        self.communicate(data_to_AB, self.other_client[Guest_A_name])
        self.communicate(data_to_AB, self.other_client[Host_B_name])
        return
	
    ## C: step5
    def task_2(self, Guest_A_name, Host_B_name):
        try:
            dt = self.data
            assert "encrypted_masked_dL_a" in dt.keys() and "encrypted_masked_dL_b" in dt.keys(), "Error: 'masked_dL_a' from A or 'masked_dL_b' from B in step 2 not successfully received."
            encrypted_masked_dL_a = dt['encrypted_masked_dL_a']
            encrypted_masked_dL_b = dt['encrypted_masked_dL_b']
            masked_dL_a = np.asarray([self.private_key.decrypt(x) for x in encrypted_masked_dL_a])
            masked_dL_b = np.asarray([self.private_key.decrypt(x) for x in encrypted_masked_dL_b])
        except Exception as e:
            print("C step 2 exception: %s" % e)

        try:
            assert "encrypted_loss" in dt.keys(), "Error: 'encrypted_loss' from B in step 2 not successfully received."
            encrypted_loss = dt['encrypted_loss']
            loss = self.private_key.decrypt(encrypted_loss) / self.A_data_shape[0] + math.log(2)
            print("loss: ", loss)
            self.loss.append(loss)
        except Exception as e:
            print("C step 2 exception: %s" % e)

        data_to_A = {"masked_dL_a": masked_dL_a}
        data_to_B = {"masked_dL_b": masked_dL_b}
        self.communicate(data_to_A, self.other_client[Guest_A_name])
        self.communicate(data_to_B, self.other_client[Host_B_name])
        return