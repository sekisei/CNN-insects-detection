# coding: UTF-8
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Activation, Input
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.utils.vis_utils import plot_model
from multiprocessing import Process, Queue
from keras.models import model_from_json
from keras.optimizers import SGD,Adam
from keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import cv2
import gc

image_width=64
image_height=64
image_num_train=20000
image_num_test=50    
all_image_num_train=2*image_num_train
all_image_num_test=2*image_num_test
epochs=200
base_dir_for_train='/home/kai/デスクトップ/Programs/pythonDNN/samples/samples_remake_20000/'
base_dir_for_test='/home/kai/デスクトップ/Programs/pythonDNN/samples/data_test_samples_50/'

#------------------------------------------------------------------------------------------------------------------------------------------------
#学習データ
#------------------------------------------------------------------------------------------------------------------------------------------------
def set_training_data_all():
    all_img_array_train=np.empty((all_image_num_train,image_width,image_height,3))
    for pic_num in range(0,image_num_train):
        all_img_array_train[pic_num]=load_img(base_dir_for_train+'sample'+str(pic_num)+'.jpg',target_size=(image_width,image_height))
        all_img_array_train[image_num_train+pic_num]=load_img(base_dir_for_train+'samplele/sample'+str(pic_num)+'.jpg',target_size=(image_width,image_height))
    ydata1_train=np.array([1,0])
    ydata2_train=np.array([0,1])
    ydata1_train=np.tile(ydata1_train,(1,image_num_train))
    ydata2_train=np.tile(ydata2_train,(1,image_num_train))
    Y_train_data=np.append(ydata1_train,ydata2_train)
    Y_train_data=Y_train_data.reshape(-1,2)
    return all_img_array_train,Y_train_data
#------------------------------------------------------------------------------------------------------------------------------------------------
#試験データ
#------------------------------------------------------------------------------------------------------------------------------------------------
def set_test_data_all():
    all_img_array_test=np.empty((all_image_num_test,image_width,image_height,3))
    for pic_num in range(0,image_num_test):
        all_img_array_test[pic_num]=load_img(base_dir_for_test+'sample'+str(pic_num)+'.jpg',target_size=(image_width,image_height))
        all_img_array_test[image_num_test+pic_num]=load_img(base_dir_for_test+'samplele/sample'+str(pic_num)+'.jpg',target_size=(image_width,image_height))
    ydata1_test=np.array([1,0])
    ydata2_test=np.array([0,1])
    ydata1_test=np.tile(ydata1_test,(1,image_num_test))
    ydata2_test=np.tile(ydata2_test,(1,image_num_test))
    Y_test_data=np.append(ydata1_test,ydata2_test)
    Y_test_data=Y_test_data.reshape(-1,2)
    return all_img_array_test,Y_test_data
#------------------------------------------------------------------------------------------------------------------------------------------------
#テスト結果表示
#------------------------------------------------------------------------------------------------------------------------------------------------
def show_result_of_test(out,Y):
    for num in range(0,all_image_num_test):
        print('----------------------------------------------------')
        print('データ番号['+str(num)+']')
        print('害虫である可能性(理想値['+str(Y[num][0])+']): '+str(out[num][0]))
        print('植物である可能性(理想値['+str(Y[num][1])+']): '+str(out[num][1]))
        print('----------------------------------------------------')     
#------------------------------------------------------------------------------------------------------------------------------------------------
#グラフ表示
#------------------------------------------------------------------------------------------------------------------------------------------------
def show_figure_of_history(h):
    loss=h.history['loss']
    acc=h.history['acc']
    val_loss=h.history['val_loss']
    val_acc=h.history['val_acc']
    plt.rc('font',family='serif')
    plt.plot(range(epochs),loss,label='loss',color='black')
    plt.plot(range(epochs),acc,label='acc',color='blue')
    plt.plot(range(epochs),val_loss,label='val_loss',color='green')
    plt.plot(range(epochs),val_acc,label='val_acc',color='red')
    plt.xlabel('epochs')
    #plt.show()
    plt.savefig('figure.pdf')
#------------------------------------------------------------------------------------------------------------------------------------------------
#'''
model = Sequential()
model.add(Convolution2D(5,18,strides=(2,2),padding='valid',input_shape=(image_width,image_height,3)))#24
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))#12
model.add(Convolution2D(10,3,strides=(1,1),padding='valid'))#10
model.add(Activation('relu'))
model.add(Convolution2D(10,3 ,strides=(1,1),padding='valid'))#8
model.add(Activation('relu'))
model.add(Convolution2D(20,3,strides=(1,1),padding='valid'))#6
model.add(Activation('relu'))
model.add(Convolution2D(20,3,strides=(1,1),padding='valid'))#4
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(2,2)))#10
#'''
model.add(Flatten())#ここから全結合
model.add(Dense(140))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights('my_model_weights.h5')#重みのロード
#plot_model(model, to_file="model.png", show_shapes=True)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy'])
X_train,Y_train=set_training_data_all()
X_test,Y_test=set_test_data_all()
'''
hist=model.fit(X_train, Y_train, epochs=epochs, batch_size=2000, validation_split=0.1)
show_figure_of_history(hist)
loss_and_metrics=model.evaluate(X_test,Y_test)
print(loss_and_metrics)
'''
'''
output=model.predict(X_test,all_image_num_test,0)
np.set_printoptions(precision=4,suppress=True)
show_result_of_test(output,Y_test)
'''

#---------------------------------------------------------------------------画像認識
cap=cv2.VideoCapture("/home/kai/デスクトップ/Programs/pythonDNN/samples/sample_video/not_mine.mp4")
#cap=cv2.VideoCapture(0) 
ret, frame=cap.read()
size=frame.shape
(width,height)=(size[1],size[0])
(strideW,strideH)=(int((width-256)/20),int((height-256)/20))
crop_image_arr=np.empty((strideW*strideH,64,64,3))
index_and_pos=np.empty((strideW*strideH,2),dtype=int)
(cntX,cntY)=(range(0,strideW),range(0,strideH))
font = cv2.FONT_HERSHEY_SIMPLEX
for x in cntX:
    for y in cntY:
        index_and_pos[x*strideH+y][0]=int(20*x)
        index_and_pos[x*strideH+y][1]=int(20*y)
while(cap.isOpened()):
     ret, frame=cap.read()
     BGRframe=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
     crop_img_arr=np.array([cv2.resize(BGRframe[20*y:20*y+256,20*x:20*x+256],None,fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA) for x in cntX for y in cntY])
     output=model.predict(crop_img_arr,strideW*strideH,0)
     maxResult=output.argmax(axis=0)
     maxResultIndex0=maxResult[0]
     mRI=maxResultIndex0
     if (output[maxResultIndex0][0]-0.9)>output[maxResultIndex0][1]:
        cv2.rectangle(frame,(index_and_pos[mRI][0],index_and_pos[mRI][1]),(index_and_pos[mRI][0]+256,index_and_pos[mRI][1]+256),(0,0,255),2)
        cv2.putText(frame,str(output[maxResultIndex0][0]),(index_and_pos[mRI][0]+10,index_and_pos[mRI][1]+30),font,1,(0,0,255),2,cv2.LINE_AA)
     cv2.imshow('frame',frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):break
cap.release()              
cv2.destroyAllWindows()          
#---------------------------------------------------------------------------
model.save_weights('my_model_weights.h5')#重みのセーブ
gc.collect()

