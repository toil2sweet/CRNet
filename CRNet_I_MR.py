import numpy as np
from numpy import random
import scipy.io as scio
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime


# define metric
def classification_accuracy(predict_label, Label):
    count = 0
    label = Label.argmax(axis=1)
    prediction = predict_label.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label[j] == prediction[j]:
            count += 1
    return round(count / len(Label), 8)


# Choice of csommonly uesed activation functions
def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def relu(data):
    return np.maximum(data, 0)


def pseudo_inv(A, reg):  # reg: regularization coefficient for ill condition
    A_p = np.linalg.pinv(A.T.dot(A)).dot(A.T)
    return np.array(A_p)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_weights(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


def preprocess(train_x):

    N1 = 10
    N2 = 10
    N3 = 670
    s = 300
    train_x = preprocessing.scale(train_x, axis=1)
    train_x_bias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    Z = np.zeros([train_x.shape[0], N2 * N1])
    
    We_set = []
    max_min = []
    min_value = []
    # time_start = time.time()


    for i in range(N2):
        random.seed(i)
        We = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        X_We_B = np.dot(train_x_bias, We)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_We_B)
        feature1 = scaler1.transform(X_We_B)
        We_star = sparse_weights(feature1, train_x_bias).T
        We_set.append(We_star)
        Zi = np.dot(train_x_bias, We_star)

        max_min.append(np.max(Zi, axis=0) - np.min(Zi, axis=0))
        min_value.append(np.min(Zi, axis=0))
        Zi = (Zi - min_value[i]) / max_min[i]
        Z[:, N1 * i:N1 * (i + 1)] = Zi
        del Zi
        del X_We_B
        del We
    

    Z_bias = np.hstack([Z, 0.1 * np.ones((Z.shape[0], 1))])

    if N1 * N2 >= N3:
        random.seed(67797325)
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    Z_Wh_B = np.dot(Z_bias, Wh)
    param_shrink = s / np.max(Z_Wh_B)
    H = tansig(Z_Wh_B * param_shrink)
    A = np.hstack([Z, H])

    return Z, H, A


# computing empirical fisher information matrix of beta
def log_liklihoods(OutputWeight, A, train_y):
    OutputWeight = torch.from_numpy(OutputWeight)
    OutputWeight.requires_grad = True
    A = torch.from_numpy(A)
    output = torch.mm(A, OutputWeight)
    label = torch.from_numpy(train_y)
    log_liklihoods = F.cross_entropy(output, label.max(1)[1])
    log_liklihoods.backward()
    FIM = OutputWeight.grad ** 2    
    return FIM.numpy() * train_y.shape[0]  # to mitigate value

# # computing empirical fisher infomation matrix of beta --CUDA is used
# def log_liklihoods(OutputWeight, A, train_y):
#     OutputWeight = torch.from_numpy(OutputWeight).cuda()
#     OutputWeight.requires_grad = True
#     A = torch.from_numpy(A).cuda()
#     output = torch.mm(A, OutputWeight)
#     label = torch.from_numpy(train_y).cuda()
#     log_liklihoods = F.cross_entropy(output, label.max(1)[1])
#     log_liklihoods.backward()
#     FIM = OutputWeight.grad ** 2    
#     return FIM.detach().cpu().numpy() * train_y.shape[0]  # to mitigate value


def confusion_matrix(baseline, result):

    max_len = max((len(l) for l in result))
    result = list(map(lambda l:l + [0]*(max_len - len(l)), result))
    
    result=torch.Tensor(result)
    baseline=torch.Tensor(baseline)

    nt = result.size(0)
    acc = result.diag()
    fin = result[nt - 1]
    bwt = fin - acc
    fwt = acc - baseline

    return fin.mean(), bwt[0:(nt-1)].mean(), fwt[1:nt].mean() 


def train(train_x, train_y, dl, C, Alpha, Lmax, task_index, FIM, lamb, OutputWeight_ed):
    
    time_start = time.time()

    Z, H, A= preprocess(train_x)
    train_x = A
    
    np.random.seed(2505)
    InputWeight = Alpha * (2 * np.random.rand(Lmax, train_x.shape[1]) - 1)
    InputBias = Alpha * (2 * np.random.rand(Lmax, 1) - 1)
    tempH = np.dot(InputWeight, train_x.T) + InputBias
    H = relu(tempH.T)

    if dl == 1:
        A = np.hstack([train_x, H])
    else:
        A = H

    if task_index == 1:
        A_p = pseudo_inv(A, C)
        OutputWeight = np.dot(A_p, train_y)
    else:

        L = A.shape[1]
        OutputWeight = np.empty((L, 0))
        for q in np.arange(train_y.shape[1]):
            sum_lambF = np.zeros((L, L))
            for t in range(1, task_index):
                FIM_q = np.diag(FIM[t-1][:, q])
                sum_lambF += lamb[t-1] * FIM_q
            beta_q = pseudo_inv(A.T.dot(A) + sum_lambF, C).dot(A.T.dot(train_y[:,q:q+1]) + sum_lambF.dot(OutputWeight_ed[:,q:q+1]))
            OutputWeight = np.concatenate((OutputWeight, beta_q), axis=1)

    train_output = np.dot(A, OutputWeight)
    train_acc = classification_accuracy(train_output, train_y)
    time_end = time.time()
    train_time = time_end - time_start
    
    print('Training accuracy on task {} is {:.4f} %'.format(task_index, train_acc * 100))
    # print('Training time on task {} is {:.4f} s'.format(task_index, train_time))
    return InputWeight, InputBias, OutputWeight, A



def test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, task_index):
    
    time_start = time.time()

    Z, H, A= preprocess(test_x)
    test_x = A

    tempH_test = np.dot(InputWeight, test_x.T) + InputBias
    H_test = relu(tempH_test.T)

    if dl == 1:
        A_test = np.hstack([test_x, H_test])
    else:
        A_test = H_test

    test_output = np.dot(A_test, OutputWeight)
    test_acc = classification_accuracy(test_output, test_y)
    time_end = time.time()
    test_time = time_end - time_start
    print('Test accuracy on task_{} is {:.4f}%'.format(task_index, test_acc * 100))
    # print('Test time on {} is {}s'.format(task_index, test_time)) 
    return test_acc 
  

# Implementation details for CRNet-I
def main():

    # Dataloader in one epoch setting. Optionally, one can use datasets.FashionMNIST 
    dataFile = 'Split5_FashionMNIST_Class_IL/FashionMNIST_split5_CIL.mat'
    data = scio.loadmat(dataFile)
    
    # parameter setting
    dl = 0  # with or without direct links
    Alpha = 1
    Lmax = 900 
    C = 2 ** -30
    lamb = [3000, 5000, 5000, 6000]
    FIM = []
    OutputWeight = []
    TN = 5
    R = []
    Ri = []
    task_index=0

    time_start = time.time()
    print('-------------------CRNet—I---------------------------')
    random.seed()
    # task_orders = [1,2,3,4,5]
    task_orders = np.random.permutation(TN)+1
    print('Command: train on current task and then test on seen ones')
    print('#'*8,'random task order of current run:', task_orders)
    # first learn task i
    for i in task_orders:
        task_index += 1 
        train_x = np.double(data['train_x_'+str((i-1)*2)+str(i*2-1)])
        train_y = np.double(data['train_y_'+str((i-1)*2)+str(i*2-1)])
        InputWeight, InputBias, OutputWeight, G = train(train_x, train_y, dl, C, Alpha, Lmax, task_index, FIM, lamb, OutputWeight)  # train on task i
        
        # then test the seen tasks 
        for j in task_orders[0:task_index]:
            test_x = np.double(data['test_x_'+str((j-1)*2)+str(j*2-1)])
            test_y = np.double(data['test_y_'+str((j-1)*2)+str(j*2-1)])
            test_accij = test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, j)  # test on task j, R_ij
            Ri.append(test_accij) 
        R.append(Ri)
        Ri = []
        if task_index != TN: 
            FIM_i = log_liklihoods(OutputWeight, G, train_y)
            FIM.append(FIM_i) 
    
    time_end = time.time()
    all_performed_time = time_end - time_start
    print('Accumulative Training and test time on {} tasks  is {:.4f} s'.format(task_index, all_performed_time))
    print('Average accuracy on {} tasks  is {:.4f}%'.format(task_index, np.mean(R[4]) * 100))
    

    # obtain the classification accuracy of an independent model trained only on each task.
    print('Independent model trained only on each task...')   
    baseline=[]
    # train and test task i by the above task order
    for i in task_orders:
        task_index = 1 # be seperatedly trained without consolidation
        train_x = np.double(data['train_x_'+str((i-1)*2)+str(i*2-1)])
        train_y = np.double(data['train_y_'+str((i-1)*2)+str(i*2-1)])
        test_x = np.double(data['test_x_'+str((i-1)*2)+str(i*2-1)])
        test_y = np.double(data['test_y_'+str((i-1)*2)+str(i*2-1)])
        InputWeight, InputBias, OutputWeight, G = train(train_x, train_y, dl, C, Alpha, Lmax, task_index, FIM, lamb, OutputWeight) 
        test_acci = test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, i)  
        baseline.append(test_acci)
        
    acc, bwt, fwt = confusion_matrix(baseline, R)
    return acc, bwt, fwt, all_performed_time


if __name__ == '__main__':
    
    Multiple = 5 # multiple runs for the mean and std of metrics
    
    ACC = []
    BWT = []
    FWT = []
    Time = []
    for multi_runs in range(Multiple):
        acc, bwt, fwt, all_performed_time=main()
        ACC.append(acc)
        BWT.append(bwt)
        FWT.append(fwt)
        Time.append(all_performed_time)
    print('Results of {} repeated runs'.format(Multiple))
    print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)))
    print('Backward: mean {:.4f}, std {:.4f}'.format(np.mean(BWT), np.std(BWT, ddof=1)))
    print('Forward:  mean {:.4f}, std {:.4f}'.format(np.mean(FWT), np.std(FWT, ddof=1)))
    print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)))

    # # also saved in the Results files
    # method = 'CRNet-I'
    # fname = method
    # fname += datetime.datetime.now().strftime("_%m_%d_%H_%M")  # "%Y_%m_%d_%H_%M_%S"
    # fname = os.path.join('Split_FashionMNIST_Class_IL/Results', fname + '.txt')
    # if fname is not None:
    #     f = open(fname, 'w')
    #     print('Results of {} with {} repeated runs'.format(method ,Multiple), file=f)
    #     print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)), file=f)
    #     print('Backward: mean {:.4f}, std {:.4f}'.format(np.mean(BWT), np.std(BWT, ddof=1)), file=f)
    #     print('Forward:  mean {:.4f}, std {:.4f}'.format(np.mean(FWT), np.std(FWT, ddof=1)), file=f)
    #     print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)), file=f)
    #     f.close()

