#import necessary packages 
import numpy as np 
import tensorflow as tf
from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np 
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
from cv2 import imread, createCLAHE 
import cv2
from glob import glob
import matplotlib.pyplot as plt


#Define the path, train files and test_files
train_dir='./Lung Segmentation/CXR_png/'
mask_dir='./Lung Segmentation/masks/'
images = os.listdir(train_dir)
mask0 = os.listdir(mask_dir)
mask0 = [i.split(".png")[0] for i in mask0]
mask = [i for i in mask0 if "mask" in i]
print("Total mask that has modified name:",len(mask))

testing_files = set(os.listdir(train_dir)) & set(os.listdir(mask_dir))
training_files = mask

#Prepare the array for test and training dataset.
def getData(x, flag = "test"):
    im_array = []
    mask_array = []
    
    if flag == "test":
        for i in tqdm(testing_files): 
            im = cv2.resize(cv2.imread(os.path.join(train_dir,i)),(x,x))[:,:,0]
            ma = cv2.resize(cv2.imread(os.path.join(mask_dir,i)),(x,x))[:,:,0]
            
            im_array.append(im)
            mask_array.append(ma)
        
        return im_array,mask_array
    if flag == "train":
        for i in tqdm(training_files): 
            im = cv2.resize(cv2.imread(os.path.join(train_dir,i.split("_mask")[0]+".png")),(x,x))[:,:,0]
            ma = cv2.resize(cv2.imread(os.path.join(mask_dir,i+".png")),(x,x))[:,:,0]

            im_array.append(im)
            mask_array.append(ma)

        return im_array,mask_array
    
#perform sanity check
def plotMask(x,y):
    sample = []
    
    for i in range(6):
        left = x[i]
        right = y[i]
        combined = np.hstack((left,right))
        sample.append(combined)
        
        
    for i in range(0,6,3):

        plt.figure(figsize=(25,10))
        
        plt.subplot(2,3,1+i)
        plt.imshow(sample[i])
        
        plt.subplot(2,3,2+i)
        plt.imshow(sample[i+1])
        
        
        plt.subplot(2,3,3+i)
        plt.imshow(sample[i+2])
        
        #plt.savefig('./pdf/test.pdf')

# Load training and testing data
dim = 256*2
x_train,y_train = getData(dim,flag="train")
x_test, y_test = getData(dim)
#plotMask(x_train,y_train)
plotMask(x_test,y_test)

#combine the train and test dataset and further use them as a unified dataset.
x_train = np.array(x_train).reshape(len(x_train),dim,dim,1)
y_train = np.array(y_train).reshape(len(y_train),dim,dim,1)
x_test = np.array(x_test).reshape(len(x_test),dim,dim,1)
y_test = np.array(y_test).reshape(len(y_test),dim,dim,1)
assert x_train.shape == y_train.shape
assert x_test.shape == y_test.shape
images = np.concatenate((x_train,x_test),axis=0)
mask  = np.concatenate((y_train,y_test),axis=0)

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred) 


#define the model function based on Unet structure
def build_unet(input_size=(512,512,1),start_neurons=16):
        inputs = Input(input_size)
            
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


        conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
                                        
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool3) 
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(conv6)
      
        up7 = concatenate([Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(conv7)

                                                                                                    
        up8 = concatenate([Conv2DTranspose(start_neurons * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(start_neurons * 1, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
                                                          
        return Model(inputs=[inputs], outputs=[conv10])
                                                       
        model = build_unet(input_size=(512,512,1),start_neurons = 16)

        model.summary()



#builld the model 
model = build_unet(input_size=(512,512,1))

#compile the model
history = model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,metrics=[dice_coef, 'binary_accuracy'])
model.summary()



#settings of callback functions
#earlystopping
earlystopping = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15)
# checkpoint
#filepath="{}_weights.best.hdf5".format('cxr_reg')
#checkpoint= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list= [earlystopping]


#split the train,valid and test dataset

train_X, validation_X, train_Y, validation_Y = train_test_split((images-127.0)/127.0,
                                                            (mask>127).astype(np.float32),
                                                            test_size = 0.1,random_state = 2018)

train_X, test_X, train_Y, test_Y = train_test_split(train_X,train_Y,
                                                            test_size = 0.1,
                                                            random_state = 2018)
#we fit the model with training and validation data 
history = model.fit(x = train_X,
           y = train_Y,
           steps_per_epoch = train_X.shape[0]//16, 
           batch_size =16,
           epochs = 1,
           validation_data =(validation_X,validation_Y) ,
           callbacks=callbacks_list)
#save the model 
model.save('seg_eval.h5')



import matplotlib.pyplot as plt
#have a brief view of the highest performance level
#train_max_acc=max(history.history['binary_accuracy'])
#validation_max_acc=max(history.history['val_binary_accuracy'])
#validation_min_loss=min(history.history['val_loss'])
#train_min_loss=min(history.history['loss'])


#we save the accuracy and loss accuracy for further plotting
import pickle
train_acc=history.history['binary_accuracy']
validation_acc=history.history['val_binary_accuracy']
train_loss=history.history['loss']
validation_loss=history.history['val_loss']

#acc_list_file = open('train_acc.pickle','wb')
#pickle.dump(train_acc,acc_list_file)
#acc_list_file.close()

#val_acc_list_file = open('validation_acc.pickle','wb')
#pickle.dump(validation_acc,val_acc_list_file)
#val_acc_list_file.close()

#loss_list_file = open('train_loss.pickle','wb')
#pickle.dump(train_loss,loss_list_file)
#loss_list_file.close()

#val_loss_list_file = open('val_loss.pickle','wb')
#pickle.dump(validation_loss,val_loss_list_file)
#val_loss_list_file.close()

#frame=[train_acc,validation_acc,train_loss,validation_loss];
#matrix=np.matrix(frame)
#np.savetxt('dt.csv', matrix, fmt='%s', delimiter = ',')

#plotting for the training,vaildation accuracy and loss
fig,ax =plt.subplots(1,2,figsize=(10,5))
ax[0].plot(history.history['binary_accuracy'],label="accuracy",color="orange")
ax[0].plot(history.history['val_binary_accuracy'],label="val_binary_accuracy",color="blue")
ax[0].set_title("Accuracy_Plot")
ax[0].legend()

ax[1].plot(history.history['loss'],label="loss",color="orange")
ax[1].plot(history.history['val_loss'],label="val_loss",color="blue")
ax[1].set_title("Loss")
ax[1].legend()
#plt.show()
#plt.savefig('./png/acc_loss2.png')

#evaluate the model
#loss,dice_coef,accuracy = model.evaluate(test_X,test_Y)


#test the model
pred_candidates = np.random.randint(1,test_X.shape[0],10)
preds = model.predict(test_X)


plt.figure(figsize=(20,10))

for i in range(0,9,3):
    plt.subplot(3,3,i+1)
            
    plt.imshow(np.squeeze(test_X[pred_candidates[i]]))
    plt.xlabel("Base Image")
                                               
    plt.subplot(3,3,i+2)
    plt.imshow(np.squeeze(test_Y[pred_candidates[i]]))
    plt.xlabel("Mask")
                                            
    plt.subplot(3,3,i+3)
    plt.imshow(np.squeeze(preds[pred_candidates[i]]))
    plt.xlabel("Prediction")

plt.show()
#plt.savefig('./png/test2.png')

# Calculate the intersection over union (IoU) and Dice coefficient (F-score) for the test results. 

#iou_list=[]
#y_true_f = keras.flatten (test_Y)
#y_pred_f = keras.flatten (preds)
#for i in y_true_f:
#    for j in y_pred_f:
#        intersection = keras.sum(i*j)
#        union = keras.sum(i) + keras.sum(j) - intersection
#        iou = intersection/union
#iou_list.append(iou)

###Define intersection over union and f1-score.
def iou_test(test_Y, preds):
    y_true_f = keras.flatten (test_Y)
    y_pred_f = keras.flatten (preds)
    for i in y_true_f:
          for j in y_pred_f:
                intersection = keras.sum(i*j)
                union = keras.sum(i) + keras.sum(j) - intersection
                iou = intersection/union
    return iou


def f1_score(test_Y, preds):
    y_true_f = keras.flatten (test_Y)
    y_pred_f = keras.flatten (preds)
    for i in y_true_f:
          for j in y_pred_f:
                intersection = keras.sum(i*j)
                f1_score = (2.*intersection) / (keras.sum(i) + keras.sum(j))
    return f1_score


iou_list=[]
f1_score_list=[]
for i in range(test_X.shape[0]//16):
    pre_img=preds[i:i+16]
    truth=test_Y[i:i+16]
    iou_score=iou_test(truth,pre_img)
    f1_score_list.append(f1_score(truth,pre_img))
    iou_list.append(iou_score)

iou_acc_list_file = open('iou_acc.pickle','wb')
pickle.dump(iou_list,iou_acc_list_file)
iou_acc_list_file.close()
pickle.dump(f1_score_list,f1_acc_list_file)
f1_acc_list_file.close()




