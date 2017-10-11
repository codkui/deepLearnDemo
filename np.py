#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

class FullConnectedLayer(object):
    '''
    全连接层实现类
    '''
    def __init__(self,input_size,output_size,activator):
        '''
        构造函数
        input_size 输入向量维度
        output_size 输出向量维度
        activator  激活函数
        '''
        self.input_size=input_size
        self.output_size=output_size
        self.activator=activator

        #权重数组w
        self.W=np.random.uniform(-0.1,0.1,(output_size,input_size))

        #偏置项b
        self.b=np.zeros((output_size,1))

        #输出向量
        self.output=np.zeros((output_size,1))

    def forward(self,input_array):
        '''
        前向计算
        input_array 输入向量，维度必须等于input_size
        '''
        #式子2
        self.input=input_array

        self.output=self.activator.forward(
            np.dot(self.W,input_array)+self.b
        )


    def backward(self,delta_array):
        '''
        反响计算w和b的梯度
        delta_array  从上一层传来的误差项
        '''
        #式子8
        self.delta=self.activator.backward(self.input)*np.dot( self.W.T,delta_array)
        #怀疑是权重和偏置项的当前值
        self.W_grad=np.dot(delta_array,self.input.T)
        self.b_grad=delta_array

    def update(self,learning_rate):
        '''
        使用梯度下降算法，更新权重
        '''
        self.W+=learning_rate*self.W_grad
        self.b+=learning_rate*self.b_grad

class SigmoidActivator(object):
    def forward(self,weighted_input):
        return 1.0/(1.0+np.exp(-weighted_input))

    def backward(self,output):
        return output*(1-output)


class Network(object):
    def __init__(self,layers):
        self.layers=[]
        for i in range(len(layers)-1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i],layers[i+1],SigmoidActivator()
                )
            )
    def predict(self,sample):
        '''
        进行预测
        将每一层的输出作为下一层的输入进行计算，将最后一层的输出作为最终结果，将原始值作为输入层的前置层输出
        '''
        output=sample
        for layer in self.layers:
            layer.forward(output)
            output=layer.output
        return output

    def train(self,labels,data_set,rate,epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)

    def train_one_sample(self,label,sample,rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)
    
    def calc_gradient(self,label):
        #print self.layers[-1].output
        delta=self.layers[-1].activator.backward(
            self.layers[-1].output
        )*(
            label-self.layers[-1].output
        )
        #反向计算，从输出层向前计算梯度与偏差
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta=layer.delta
        return delta

    def update_weight(self,rate):
        for layer in self.layers:
            layer.update(rate)

def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg)
        , args
    )