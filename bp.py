# -*- coding:utf-8 -*-
#节点类
import random
from numpy import *

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))
class Node(object):
    def __init__(self,layer_index,node_index):
        '''
        构造节点
        layer_index 神经层编号
        node_index  层内节点编号
        downstream  下游连接
        upstream    上游连接
        output      输出
        delta       误差
        '''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0

    def set_output(self,output):
        '''
        设置节点输出值，该方法用于节点是输入层时
        '''
        self.output=output

    def append_downstream_connection(self,conn):
        '''
        添加一个下游连接
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self,conn):
        '''
        添加一个上游连接
        '''
        self.upstream.append(conn)
    
    def calc_output(self):
        '''
        根据式子1计算节点输出
        '''
        output=reduce(lambda ret,conn:ret+conn.upstream_node.output*conn.weight,
        self.upstream,0)
        self.output=sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        当节点时隐藏层时，根据式4计算delta
        '''
        downstream_delta=reduce(
            lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,
            self.downstream,0.0
        )
        self.delta=self.output*(1-self.output)*downstream_delta

    def calc_output_layer_delta(self,label):
        '''
        当节点是输出层时，根据式3计算delta
        '''
        self.delta=self.output * (1-self.output)*(label-self.output)

    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    def __init__(self,layer_index,node_index):
        '''
        构造一个输出恒定为1的节点，作为w0x0,另一种说法是作为偏置项
        '''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=1

    def append_downstream_connection(self,conn):
        self.downstream.append(conn)
    
    def calc_hidden_layer_delta(self):
        downstream_delta=reduce(
            lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,
            self.downstream,0.0
        )
        self.delta=self.output*(1-self.output)*downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

class Layer(object):
    def __init__(self,layer_index,node_count):
        '''
        初始化一层
        layer_index 层编号
        node_count  层内节点数量
        '''
        self.layer_index=layer_index
        self.nodes=[]
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))

    def set_output(self,data):
        '''
        设置层输出，层为输入层时用到
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])
        
    def calc_output(self):
        '''
        计算层输出向量
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        打印层信息
        '''
        for node in self.nodes:
            print node


class Connection(object):
    '''
    连接对象，记录连接上下游和权重
    '''
    def __init__(self,upstream_node,downstream_node):
        self.upstream_node=upstream_node
        self.downstream_node=downstream_node
        self.weight=random.uniform(-.1,0.1)
        self.gradient=0.0

    def calc_gradient(self):
        '''
        计算梯度
        '''
        self.gradient=self.downstream_node.delta*self.upstream_node.output

    def get_gradient(self):
        '''
        获取当前梯度
        '''
        return self.gradient

    def update_weight(self,rateSet):
        '''
        根据梯度下降算法更新权重
        '''
        self.calc_gradient()
        self.weight+=rateSet*self.gradient

    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)

class Connections(object):
    '''
    连接对象集合
    '''
    def __init__(self):
        self.connections=[]

    def add_connection(self,connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print conn
    
class Network(object):
    '''
    神经网络对象，提供对外api
    '''
    def __init__(self,layers):
        '''
        初始化一个全连接神经网络
        layers：二维数组，描述神经网每层节点数
        '''
        self.connections=Connections()
        self.layers=[]
        layer_count=len(layers)
        node_count=0
        #构建神经层
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        
        for layer in range(layer_count-1):
            '''
            初始化每层的连接集合，提取该层所有节点的连接和下一层所有（除偏置项）节点
            '''
            connections=[Connection(upstream_node,downstream_node)
            for upstream_node in self.layers[layer].nodes
            for downstream_node in self.layers[layer+1].nodes[:-1]
            ]
            #遍历节点
            for conn in connections:
                #添加到节点总集合
                self.connections.add_connection(conn)
                #向上游连接添加自己到下游节点，向下游连接添加自己到上游节点
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rateSet,iteration):
        '''
        lables 训练标签
        data_set 二维数组，训练样本特征，每个元素时一个样本的特征，如[1,3,1] 表示 本科 3年工作经验 1年公司工作时间
        rateSet 训练步长
        iteration 迭代次数
        '''

        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rateSet)

    def train_one_sample(self,label,sample,rateSet):
        '''
        用一个样本训练网络
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rateSet)

    def calc_delta(self,label):
        '''
        内部函数，计算节点的delta
        '''
        #取输出层的节点，计算输出层的误差
        output_nodes=self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        #理论是取所有隐藏层，计算误差，但是切片不是很明白
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()
    
    def update_weight(self,rateSet):
        '''
        内部函数，更新每一个连接的权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rateSet)
    
    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()
    
    def get_gradient(self,label,sample):
        '''
        获取网络在一个样本下，每个连接上的梯度
        label 样本标签
        sample 样本
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self,sample):
        '''
        根据输入预测输出值
        '''
        #初始化输入层
        self.layers[0].set_output(sample)
        #遍历除输出层外所有层，计算输出
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        #返回输出层输出
        return map(lambda node:node.output,self.layers[-1].nodes[:-1])

    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()

def gradient_check(network,sample_feature,sample_label):
    '''
    梯度检查，用于测试神经网络是否正常梯度下降
    network     神经网络
    sample_feature  样本特征
    sanple_label    样本标签
    '''

    #计算网络误差
    network_error=lambda vec1,vec2:0.5*reduce(
        lambda a,b:a+b,
        map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2))
    )

    #获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature,sample_label)

    #对每个权重做梯度检查
    for conn in network.connections.connections:
        #获取指定连接的梯度
        actual_gradient=conn.get_gradient()

        #增加一个很小的值，计算网络误差
        epsilon=0.0001
        conn.weight+=epsilon
        error1=network_error(network.predict(sample_feature),sample_label)

        #减去一个很小的值，计算网络误差
        conn.weight-=2*epsilon
        error2=network_error(network.predict(sample_feature),sample_label)

        #根据式6计算期望的梯度
        expected_gradient=(error2-error1)/(2*epsilon)

        #打印
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient)

if __name__=="__main__":
    #a=Network([2,2,2])
    gradient_check(Network([2,2,2]),[0,1],[0,1])