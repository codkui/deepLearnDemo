# -*- coding:utf-8 -*-  
'''
简单线形单元，来自于教程
https://www.zybuluo.com/hanbingtao/note/448086
'''

from perceptron import Perceptron

f= lambda x:x

class LinearUnit(Perceptron):
    def __init__(self,input_num):
        Perceptron.__init__(self,input_num,f)

def get_training_dataset():
    '''
    训练数据，教程为根据工作年限计算公子
    参数2 学历 1高中及以下 2专科 3本科 4本科以上
    参数3 工作评级 1 2 3 4 5
    '''
    input_vecs=[[5,2,2],[3,2,1],[8,2,4],[1.4,2,1],[10.1,3,5]]

    lables=[5500,2300,7600,1800,11400]

    return input_vecs,lables

def train_linear_unit():
    '''
    使用数据训练线形单元
    '''
    #创建线形单元，特征数量为1
    lu=LinearUnit(3)

    input_vecs,lables=get_training_dataset()
    #训练，迭代次数10，学习速率0.01
    lu.train(input_vecs,lables,3400,0.0001)

    return lu

if __name__=='__main__':
    '''训练线形单元'''
    linear_unit=train_linear_unit()
    print linear_unit

    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([5,2,2])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([3,2,1])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([8,2,4])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([1.4,2,1])

    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4,1,1])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15,2,2])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5,3,5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3,2,1])
