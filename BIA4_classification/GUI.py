# -*-coding:utf-8-*-
import tkinter as tk
from os.path import dirname

import tensorflow as tf
from keras import models
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import TensorBoard
from tkinter import ttk
#input the image
def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (512, 512))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)
    pre_x = np.array(pre_x) / 255.0
    return pre_x
# prediction of y, match the label
def put_prey(pre_y,label):
    output=[]
    for y in pre_y:
        if y[0]<0.5:
            output.append([label[0],1-y[0]])
        else:
            output.append([label[1], y[0]])
    return output

save_pathway=dirname(__file__)+"h5/"

originalmodel=models.load_model(save_pathway+'acc_0.9624999761581421_batch32_op_sgd_model_3layer.h5')
segmodel=models.load_model(save_pathway+'acc_[0.7284375, 0.8287241, 0.8243243, 0.8475, 0.8799497, 0.8758642, 0.8796875, 0.87963545, 0.8659375, 0.88874924]_model.h5')


def runCNN(img,model):
    pre_x=get_inputs([img])
    pre_y=model.predict(pre_x)
    image_pathway = "/to/your/path/of/BIA4/TB_Chest_Radiography_Database/"
    train_dir=image_pathway+"train"
    train_pic_gen=ImageDataGenerator(rescale=1./255)
    train_flow=train_pic_gen.flow_from_directory(train_dir,(512,512),batch_size=16,class_mode='binary')
    output=put_prey(pre_y,list(train_flow.class_indices.keys()))
    result=output[0][0] #class
    P=output[0][1] #probability
    print("result:",result)
    print("P:",P)
    return result,P


###GUI

window = tk.Tk()
window.title('Tuberculosis identification system')
window.geometry('450x300')

#label
sub_tit1 = tk.Label(window, text='Image pathway', font=('Arial', 12), width=30, height=2)
sub_tit2 = tk.Label(window, text='Result', font=('Arial', 12), width=30, height=2)

sub_tit1.place(x=130,y=50,anchor='center')
sub_tit2.place(x=130,y=160,anchor='center')

#image pathway input box
e=tk.Entry()
e.pack()
def getvalue():
    img=e.get()
    print (img)
    return img
e.place(x=300,y=50,anchor='center')
selected = tk.IntVar()
#radio button to select original or semented images as input
rad1 = tk.Radiobutton(window, text="Original", value=1, variable=selected)
rad2 = tk.Radiobutton(window, text="Segmented", value=2, variable=selected)
def clicked():
    print(selected.get())
    return selected.get()

rad1.place(x=200,y=110,anchor='center')
rad2.place(x=300,y=110,anchor='center')


#result box
var = tk.StringVar()
l = tk.Label(window, textvariable=var,relief="ridge",borderwidth = 1,bg='white',font=('Arial',12),width=15,height=2) #window上设置label
l.place(x=300,y=160,anchor='center')


#submit button
on_hit = False
def hit_me():
    global on_hit
    img=getvalue()
    print("?",img)
    modeltype=clicked()
    if modeltype==1:
        result,P=runCNN(img,originalmodel)
    if modeltype==2:
        result,P=runCNN(img,segmodel)
    if result=="Tuberculosis":
        on_hit = True
        var.set('Tuberculosis')
    else:
        on_hit=False
        var.set('Normal')

b2 = tk.Button(window,text='Start Classification',width=15,height=2,command=hit_me) #按钮用于提取image pathway
b2.place(x=225,y=250,anchor='center')
window.mainloop()


