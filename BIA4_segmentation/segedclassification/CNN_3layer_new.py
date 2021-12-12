#-*-coding:utf-8-*-
from os.path import dirname

import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout

#input of the image pathway and save pathway of h5 model
prompt = "> "
print(f"please type the pathway of image data")
print("example: /to/your/path/of/BIA4/TB_Chest_Radiography_Database/")
image_pathway = input(prompt)
print("please type the pathway of model saving")
#print("example: Discolimus,Rhbiditis,Miconchus,Eudorylaimus,Axonchium,Pratylenchus,Amplimerlinius,Dorylaimus,Ditylenchus,Acrobeloides,Panagrolaimus,Aphelenchoides,Acrobeles,Pristionchus,Xenocriconema,Aporcelaimus,Mylonchulus,Helicotylenchus,Mesodorylaimus")
print("save_pathway: /to/your/path/of/BIA4/BIA4_project_classification/h5/")
save_pathway = input(prompt)

#data input and preprocessing
train_dir=image_pathway+"train"
test_dir=image_pathway+"test"
val_dir=image_pathway+"val"
train_pic_gen=ImageDataGenerator(rescale=1./255)
test_pic_gen=ImageDataGenerator(rescale=1./255)
val_pic_gen=ImageDataGenerator(rescale=1./255)
train_batch_size=32
train_flow=train_pic_gen.flow_from_directory(train_dir,(512,512),batch_size=train_batch_size,class_mode='binary')
test_flow=test_pic_gen.flow_from_directory(test_dir,(512,512),batch_size=16,class_mode='binary')
val_flow=val_pic_gen.flow_from_directory(val_dir,(512,512),batch_size=16,class_mode='binary')

# print(train_flow.class_indices)
#model construction
model=Sequential([
    Convolution2D(32,4,4,input_shape=(512,512,3),activation='relu'),
    MaxPool2D(pool_size=(4,4)),
    Convolution2D(64,4,4,activation='relu'),
    MaxPool2D(pool_size=(4,4)),
    Convolution2D(128,4,4,activation='relu'),
    MaxPool2D(pool_size=(4,4)),
    Flatten(),
    Dense(1024,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.summary()
op="sgd"
model.compile(optimizer=op,loss='binary_crossentropy',metrics=['accuracy'])

hist=model.fit_generator( train_flow,steps_per_epoch=4900//train_batch_size,epochs=10,validation_data=val_flow,validation_steps=50)
#hist=model.fit_generator(train_flow,steps_per_epoch=2,epochs=2,validation_data=val_flow,validation_steps=50)
score = model.evaluate(test_flow,steps=25)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#print(hist.history.keys())
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
model.save(save_pathway+'acc_'+str(score[1])+"_batch"+str(train_batch_size)+"_op_"+str(op)+'_model_3layer_seged.h5')
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')

plt.legend()
plt.figure()
plt.savefig(dirname(__file__)+"/acc_and_loss/"+str(score[1])+"_batch"+str(train_batch_size)+"_op_"+str(op)+"_acc_3layer_seged.png")
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.savefig(dirname(__file__)+"/acc_and_loss/"+str(score[1])+"_batch"+str(train_batch_size)+"op_"+str(op)+"_loss_3layer_seged.png")
plt.show()
