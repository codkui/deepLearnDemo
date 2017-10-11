#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
该代码是学习MNIST的手写数字数据进行训练
'''
import struct
from  np import *
from datetime import datetime


def now():
    return datetime.now().strftime('%c')


#数据加载基类
class Loader(object):
    def __init__(self,path,count):
        '''
        初始化加载器
        path 路径
        count 文件中样本个数
        '''
        self.path=path
        self.count=count

    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path,'rb')
        content=f.read()
        f.close()
        return content
    
    def to_int(self,byte):
        '''
        将byte转为整数
        '''
        return struct.unpack('B',byte)[0]



#图像加载器
class ImageLoader(Loader):
    def get_picture(self,content,index):
        '''
        内部函数，从内存中获取图像数据
        从16位开始，读取28*28矩阵
        将二进制标识转为整数
        '''
        start=index*28*28+16
        picture=[]
        for i in range(28):
            picture.append([])
            for n in range(28):
                picture[i].append(
                    self.to_int(content[start+i*28+n])
                )
        
        return picture
    def get_one_sample(self,picture):
        '''
        内部函数，将图像转为样本的输入向量
        '''
        sample=[]
        for i in range(28):
            for n in range(28):
                sample.append(picture[i][n])

        return sample

    def load(self):
        '''
        加载数据，获取所有样本的输入向量
        '''
        content=self.get_file_content()
        data_set=[]
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content,index)
                )
            )

        return data_set

#标签加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获取全部样本的标签向量
        '''
        content=self.get_file_content()
        labels=[]
        for index in range(self.count):
            labels.append(self.norm(
                content[index+8]
            ))
        return labels

    def norm(self,label):
        '''
        内部函数，将一个值转为10维标签向量
        '''
        label_vec=[]
        label_value=self.to_int(label)
        for i in range(10):
            if i==label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_train_data_set():
    '''
    获取训练数据
    '''
    image_loader=ImageLoader('train-images-idx3-ubyte/data',60000)
    label_loader=LabelLoader('train-labels-idx1-ubyte/data',60000)
    return image_loader.load(),label_loader.load()

def get_test_data_set():
    '''
    获取测试数据
    '''
    image_loader=ImageLoader('t10k-images-idx3-ubyte/data',10000)
    label_loader=LabelLoader('t10k-labels-idx1-ubyte/data',10000)
    return image_loader.load(),label_loader.load()

def get_result(vec):
    '''
    对输出向量进行转换，当前是将最大值的区位转为识别数字
    '''
    max_value_index=0
    max_value=0
    for i in range(len(vec)):
        if vec[i]>max_value:
            max_value=vec[i]
            max_value_index=i
    return max_value_index

def evaluate(network,test_data_set,test_labels):
    '''
    错误率计算
    '''
    error=0
    total=len(test_data_set)

    for i in range(total):
        label=get_result(test_labels[i])
        predict=get_result(network.predict(test_data_set[i]))
        if label != predict:
            error+=1
    return float(error)/float(total)


def train_and_evaluate():
    last_error_ratio=1.0
    epoch=0
    print '%s 开始读取训练数据' % (now())
    train_data_set,train_labels=transpose(get_train_data_set())
    print '%s 开始读取测试数据' % (now())
    test_data_set,test_labels=transpose(get_test_data_set())
    print '%s 开始构建神经网络' % (now())
    network=Network([784,300,10])
    print '%s 完成构建神经网络' % (now())

    while True:
        epoch+=1
        print '%s 训练轮次 %d 开始' % (now(),epoch)
        network.train(train_labels,train_data_set,0.01,1)
        
        print '%s 训练轮次 %d 完毕' % (now(),epoch)
        if epoch % 1==0:
            error_ratio=evaluate(network,test_data_set,test_labels)
            print '%s 训练轮次 %d, 错误率为 %f' % (now(), epoch, error_ratio)
            if error_ratio>last_error_ratio:
                #出现当前轮次错误率高于前一次错误率，即越过了谷底，停止训练
                break
            else:
                last_error_ratio=error_ratio

if __name__=="__main__":
    train_and_evaluate()