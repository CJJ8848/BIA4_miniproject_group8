from os.path import dirname
from PIL.Image import Image
from keras import models
import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import TensorBoard
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#input the image
def get_inputs(imagesDirectory):
    pre_x = []
    for img_name in os.listdir(imagesDirectory):
        img_path = imagesDirectory + img_name  # pathway of each image
        print(img_path)
        input = cv2.imread(img_path)
        input = cv2.resize(input, (512, 512))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input) 
    pre_x = np.array(pre_x) / 255.0
    return pre_x
#predict y and match the label
def put_prey(pre_y):
    output=[]
    for y in pre_y:
        output.append(y[0])
    return output
#model pathway
prompt = "> "
print(f"please type the pathway of image data")
print("example: /to/your/path/of/h5model/")
model = models.load_model(prompt)
#image pathway
print(f"please type the pathway of image data")
print("example: /to/your/path/of/BIA4/TB_Chest_Radiography_Database/")
image_pathway = input(prompt)

test_dir1=image_pathway+"test/Normal/"
test_dir2=image_pathway+"test/Tuberculosis/"
pre_x1=get_inputs(test_dir1)
pre_x2=get_inputs(test_dir2)

pre_y1=model.predict(pre_x1)
pre_y2=model.predict(pre_x2)

train_dir=image_pathway+"train"
test_dir=image_pathway+"test"

train_pic_gen=ImageDataGenerator(rescale=1./255)
test_pic_gen=ImageDataGenerator(rescale=1./255)


train_flow=train_pic_gen.flow_from_directory(train_dir,(512,512),batch_size=16,class_mode='binary')

test_flow=test_pic_gen.flow_from_directory(test_dir,(512,512),batch_size=16,class_mode='binary')

#get the prediction result
output1=put_prey(pre_y1)
output2=put_prey(pre_y2)
output=output1+output2
#print("pre_y,",output)
#give the y test label
y_test=[0]*350+[1]*70
#run the roc curve get the fp rate and tp rate and auc value
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, output)
auc_keras = auc(fpr_keras, tpr_keras)

#plot and save

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(dirname(__file__)+"/ROC_curve/"+str(auc_keras)+"_.png")
plt.show()
