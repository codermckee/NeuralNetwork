# -*- coding: utf-8 -*-
import numpy as np


def tanh(x):
    return np.tanh(x)     #定义一个双曲函数

def tanh_derivative(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1.0/(1+np.exp(-x)) #定义一个逻辑函数

def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))

class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):
        '''
        :param layers:列表，包含每一层的神经结点个数(但神经网络的层数=len(layers)-1)
        :param activation: 使用的activation function
        '''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative
        self.weights = []
        self.bias = []
        for i in range(1,len(layers)):
            self.weights.append(  (np.random.random((layers[i-1],layers[i]))-0.5)/2.0 )
            self.bias.append(    (np.random.random((1,layers[i]))-0.5)/2.0  )
           # self.bias.append((np.random.random((layers[i-1],layers[i]))-0.5)/2.0)
            #每一层的weight，由上一层的结点和下一层的结点相连接，因此是一个矩阵形式，行数为上一层结点数，列数为下一层结点数
            #初始化值为-0.25到0.25之间
    def fit(self,X,y,lr = 0.1, epochs = 500):
        '''
        :param X: 训练样本
        :param y: class_label
        :param lr: learning rate
        :param epochs:训练次数
        :return:
        '''
        print '初始化权重'
        print self.weights
        print '初始化bias'
        print self.bias

        X = np.atleast_2d(X)
        print '训练集'
        print X
        #temp = np.ones([X.shape[0],X.shape[1]+1])
        #temp[:,0:-1] = X
        #X = temp
        y = np.array(y)
        print 'label:',y


        for k in range(epochs):
            print '第%d次迭代' %(k+1)
            print '旧的权重'
            print self.weights
            i = np.random.randint(X.shape[0])
            #print i
            a = [X[i]] #输入层神经结点的值
            print a
            print '权重list长度',len(self.weights)
            for j in range(len(self.weights)):
                print 'a[%d]'%j,a[j]
                print 'weights',self.weights[j]
                a.append((self.activation(np.dot(a[j],self.weights[j]))+self.bias[j]))
                print '实时的a',a
                #矩阵相乘，再求和，然后进行非线性转化
                #若a[j]为1xn ,weights[j]为nx5,则结果为1x5； 结果保存至a中
                #最终每个结点的值都存放于a中
            print '加权后的a为',a
            #print '---------------'
            deviation = y[i] - a[-1]
            print 'y[i]',y[i]
            print '真实值与预测值之差',deviation
            Err = [deviation * self.activation_deriv(a[-1])]#第一个位置上存放Output layer上的Err，后面即将存放hidden layer
            #print 'Err',Err
            #delats用来存储每一层的Err
            #BP
            for j in range(len(a)-2,0,-1):
                Err.append(Err[-1].dot(self.weights[j].T)*self.activation_deriv(a[j]))#隐藏层的误差
            Err.reverse()
            print '完成后的delta',Err
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                print 'layer',layer
                delta = np.atleast_2d(Err[i])
                print 'delta',delta
                self.weights[i] += lr * layer.T.dot(delta)#权重的更新
                self.bias[i] += lr * delta
                print 'lr*delta',lr*delta
            print '新的权重'
            print self.weights
            print '新的bias'
            print self.bias
    def predict(self,x):
        a = np.array(x)
        #temp = np.ones(x.shape[0]+1)
        #temp[0:-1] = x
        #a = x
        #print '--------------',a
        for j in range(0,len(self.weights)):
            a = self.activation(     (np.dot(a,self.weights[j]))  + self.bias[j]  )
            #print np.dot(a[j],self.weights[j])
            #print '####'
            #print '------',a
        return a







if __name__ == '__main__':
    nn = NeuralNetwork([2,2,1],'tanh')
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    nn.fit(X,y)
    print '----------预测结果----------------'
    for i in [[0,0],[0,1],[1,0],[1,1]]:
        print i,nn.predict(i)
        #print '预测为:',nn.predict(i)


