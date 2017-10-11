# -*- coding:utf-8 -*-

import numpy as np
'''
卷积神经网络实现
'''



'''
公共函数区域
'''

def element_wise_op(array,op):
    '''
    numpy的数组array 按元素进行操作，结果写回数组
    '''
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...]=op(i)


def conv(input_array,kernel_array,output_array,stride,bias):
    '''
    卷积计算 自动适配2d 3d
    输入数据
    卷积层权重
    输出数据
    卷积步长
    偏置项
    '''
    #shape 对象的维度如 [1,3,5] 标识 1* 3*5的三维数据
    channel_number=input_array.ndim
    output_width=output_array.shape[1]
    output_height=output_array.shape[0]
    kernel_width=kernel_array.shape[-1]
    kernel_height=kernel_array.shape[-2]

    for i in range(output_height):
        for j in range(output_width):
            #卷积操作 奇怪的是这样操作应该尺寸不会变化，卷积后没有进行压缩
            output_array[i][j]=(
                get_patch(input_array,i,j,kernel_width,kernel_height,stride)*kernel_array
            ).sum()+bias
    

def padding(input_array,zp):
    '''
    为数组增加zero padding ，自动适配2d 3d
    '''
    if zp==0:
        return input_array
    else:
        if input_array.ndim==3:
            input_width=input_array.shape[2]
            input_height=input_array.shape[1]
            input_depth=input_array.shape[0]

            #全0 数组
            padded_array=np.zeros((
                input_depth,input_height+2*zp,input_width+2*zp
            ))

            #填充原始数据到0数组内
            padded_array[:,zp:zp+input_height,zp:zp+input_width]=input_array

            return padded_array
        elif input_array.ndim==2:
            input_width=input_array.shape[1]
            input_height=input_array.shape[0]
            #全0 数组
            padded_array=np.zeros((
                input_height+2*zp,input_width+2*zp
            ))

            #填充原始数据到0数组内
            padded_array[zp:zp+input_height,zp:zp+input_width]=input_array

            return padded_array




class ConvLayer(object):
    '''
    卷积神经层
    '''
    def __init__(self,input_width,input_height,channel_number,filter_width,filter_height,filter_number,zero_padding,stride,activator,learning_rate):
        '''
        input_* 输入数据参数
        channel_number 怀疑是输入数据维度或者特征检测器维度 通道数？
        filter_* 特征检测器参数
        stride 卷积步长
        zero_padding 外围补零参数
        activator 激活函数 当前使用relu
        '''
        self.input_width=input_width
        self.input_height=input_height
        self.channel_number=channel_number
        self.filter_width=filter_width
        self.filter_height=filter_height
        self.filter_number=filter_number
        self.zero_padding=zero_padding
        self.stride=stride
        #计算卷积输出尺寸  初始化卷积输出
        self.output_width=ConvLayer.calculate_output_size(
            self.input_width,self.filter_width,zero_padding,stride
        )
        self.output_height=ConvLayer.calculate_output_size(
            self.input_height,self.filter_height,zero_padding,stride
        )

        self.output_array=np.zeros((self.filter_number,self.output_height,self.output_width))

        #初始化卷积特征器
        self.filters=[]
        for i in range(filter_number):
            self.filters.append(
                Filter(filter_width,filter_height,self.channel_number)
            )
        self.activator=activator
        self.learning_rate=learning_rate


    @staticmethod
    def calculate_output_size(input_size,filter_size,zero_padding,stride):
        '''
        输出尺寸计算 staticmethod 私有方法
        '''
        return (input_size-filter_size+2*zero_padding)/stride+1

    def forward(self,input_array):
        '''
        前向计算，即根据前向输入计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array=input_array
        #进行padding补0
        self.padded_input_array=padding(input_array,self.zero_padding)

        for f in range(self.filter_number):
            filter=self.filters[f]
            #卷积计算
            conv(
                self.padded_input_array,filter.get_weights(),self.output_array[f],self.stride,filter.get_bias()
            )
        #对输出进行前向计算
        element_wise_op(self.output_array,self.activator.forward)
    
    def backward(self, input_array, sensitivity_array, 
                 activator):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array,
                                activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self,sensitivity_array,activator):
        '''
        该函数用作传递误差项，即将本层的误差计算出上一层的误差，提交给上层
        '''
        #处理卷积步长，对原始的误差进行拓展
        expanded_array=self.expand_sensitivity_map(sensitivity_array)

        


class Filter(object):
    def __init__(self,width,height,depth):
        '''
        初始化特征器
        '''
        self.weights=np.random.uniform(-1e-4,1e-4,(depth,height,width))
        #偏置项 初始
        self.bias=0
        #当前计算的权重结果，作为中间缓存，不改变真实权重
        self.weights_grad=np.zeros(
            self.weights.shape
        )
        self.bias_grad=0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self,learning_rate):
        '''
        更新权重与偏置项
        '''
        self.weights-=learning_rate*self.weights_grad
        self.bias-=learning_rate*self.bias_grad

