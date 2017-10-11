#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.datasets import mnist  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D  
from keras.utils import np_utils  
from keras import backend as K
from keras.models import model_from_json

import SimpleHTTPServer
from BaseHTTPServer import HTTPServer,BaseHTTPRequestHandler
import SocketServer
import json
  
# 全局变量 

newModel=True
#批梯度下降样本数
batch_size = 128 
#分类数量
nb_classes = 10 
#学习轮次
epochs = 18  
#输入图片数据的宽高
img_rows, img_cols = 28, 28  
#特征器数量 
nb_filters = 32  
# 最大池化的矩阵尺寸 
pool_size = (2, 2)  
#特征器尺寸  
kernel_size = (3, 3)  
  
# the data, shuffled and split between train and test sets  
(X_train, y_train), (X_test, y_test) = mnist.load_data()  
  
# 根据不同的backend定下不同的格式 

if K.image_dim_ordering() == 'th':  
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)  
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)  
    input_shape = (1, img_rows, img_cols)  
else:  
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)  
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)  
    input_shape = (img_rows, img_cols, 1) 


#数据归一化
X_train = X_train.astype('float32')  
X_test = X_test.astype('float32')  
X_train /= 255  
X_test /= 255  
print('X_train shape:', X_train.shape)  
print(X_train.shape[0], 'train samples')  
print(X_test.shape[0], 'test samples')  
  
# 转换为one_hot类型  
Y_train = np_utils.to_categorical(y_train, nb_classes)  
Y_test = np_utils.to_categorical(y_test, nb_classes)  

if newModel==False:
    #加载模型数据和weights  
    model = model_from_json(open('my_model_architecture.json').read())    
    model.load_weights('my_model_weights.h5')
else:
    #构建模型  
    model = Sequential()  
    """ 
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], 
                            border_mode='same', 
                            input_shape=input_shape)) 
    """ 
    
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),  
                            padding='same',  
                            input_shape=input_shape)) # 卷积层1  
    model.add(Activation('relu')) #激活层  
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2  
    model.add(Activation('relu')) #激活层  
    model.add(MaxPooling2D(pool_size=pool_size)) #池化层  
    model.add(Dropout(0.25)) #神经元随机失活  
    model.add(Flatten()) #拉成一维数据
    
    
    # model.add(Convolution2D(1, 1, 1, 
    #                         border_mode='same', 
    #                         input_shape=input_shape)) 
    # model.add(Flatten()) 

    model.add(Dense(300)) #全连接层1  
    model.add(Activation('relu')) #激活层  
    model.add(Dropout(0.5)) #随机失活 
    model.add(Dense(300)) #全连接层1  
    model.add(Activation('relu')) #激活层  
    model.add(Dropout(0.5)) #随机失活  
    model.add(Dense(nb_classes)) #全连接层2  
    model.add(Activation('softmax')) #Softmax评分  
    
    #编译模型  
    # model.compile(loss='categorical_crossentropy',  
    #             optimizer='adadelta',  
    #             metrics=['accuracy'])  
#训练模型
model.compile(loss='categorical_crossentropy',  
                optimizer='adadelta',  
                metrics=['accuracy']) 
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  
#           verbose=1, validation_data=(X_test, Y_test)) 
#评估模型  
score = model.evaluate(X_test, Y_test, verbose=0)
valueCheck=model.predict(X_test[:1])
#保存模型
json_string = model.to_json()  #等价于 json_string = model.get_config()  
open('my_model_architecture.json','w').write(json_string)    
model.save_weights('my_model_weights.h5')  

print('Test score:', score[0])  
print('Test accuracy:', score[1])

PORT = 8000


class TestHTTPHandle(BaseHTTPRequestHandler): 
    def do_GET(self):
        buf = "It works"
        self.protocal_version = "HTTP/1.1"
  
        self.send_response(200)
  
        self.send_header("Welcome", "Contect")     
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers()

        if(self.path=="/favicon.ico"):
            self.wfile.write(buf)
            return
        print("收到请求")
        
        meg=self.path[1:]
        req=json.loads(meg)
        req=np.array([req])
       
        res=model.predict(req)
        maxre=0
        ret=-1
        
        for n in range(len(res[0])):
            if res[0][n]>maxre:
                maxre=res[0][n]
                ret=n
                
        print(ret)
        buf=ret
        self.wfile.write(buf)
 
httpd = SocketServer.TCPServer(("", PORT), TestHTTPHandle)
 
print "serving at port", PORT
httpd.serve_forever()

