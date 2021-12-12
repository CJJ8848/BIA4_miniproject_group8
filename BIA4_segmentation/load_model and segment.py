import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from cv2 import imread, createCLAHE, imwrite, imshow
import os
import cv2


def dice_coef(y_true, y_pred):
    '''
    y_true:batch of ground truth image
    p_pred:batch of prediction image 

    the function return the dice accruacy to measure the segmentation effeciency
    '''
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

#load the pretrained model
loaded_model = keras.models.load_model("seg_3.h5",custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
loaded_model.summary()

#implementation of the ostu algorithm to seperate threshold of mask 
def otsu(img):
    #caluculate each pixel's weight
    mean_weight=1/(img.shape[0] * img.shape[1])
    counts, bins = np.histogram(img,bins=range(0,256))

    #initialize intra-class variance(mas)
    inter_var=-1
    final_thresh=-1

    for bin in bins[1:-1]:
        #compute class probability
        pc1=np.sum(counts[:bin])
        pc2=np.sum(counts[bin:])
        #class mean
        m1=pc1 * mean_weight
        m2=pc2 * mean_weight
        #calculate class mean = weighted-class p / class p
        mc1=np.sum(i * counts[i] for i in range(0,bin)) / m1
        mc2=np.sum(i * counts[i] for i in range(bin,255)) / m2
        final_value=pc1*pc2*(mc1-mc2)**2

        #iterate for the maximun inter-class variance
        if final_value > inter_var:
            final_thresh = bin
            inter_var=final_value

    return final_thresh


def seg(maskdir,originaldir,savedir):
    '''
    seg function is defined to segment the images using pretrained model
    -------------------------------------------------------------------
    maskdir: directory of the mask images
    originaldir: directory of the images needed to be segmented
    savedir: the filepath where the segmented image will be stored
    '''
    for img_name in os.listdir(originaldir):
        img_path = originaldir + img_name  # pathway of each image
        print(img_path)
        input = cv2.resize(cv2.imread(img_path),(512,512))[:,:,0]
        input = np.array(input).reshape(1,512,512,1) / 255.0
        #normalize the image into (0,1)

        pred = loaded_model.predict(input)
        #predict the image 
        pred = np.array(pred).reshape(512,512,1) *255.0
        #turn the image value into (0,255)
        print("pred:",pred.shape)
        imgmask=maskdir+str(img_name[:-4])+"_mask.png"
        imwrite(imgmask,pred)
        #save the predict mask images 
        pic = imread(img_path)
        mask=imread(imgmask)
        threshold=otsu(mask)
        #use otsu function to separate the theshold of mask 
        segmented_lung=np.zeros(mask.shape)
        print("shape:",mask.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # the value of the pix of the mask is bigger than the threshold,
                # we consifered it as the lung;therefore fill in the segmented_lung with
                # the original image value. 
                if mask[i][j][0]>=threshold:
                    segmented_lung[i][j]=pic[:,:,0][i][j]
        imwrite(savedir+str(img_name[:-4])+"_final.png",segmented_lung)

#run
n_maskdir="/Users/cuijiajun/Desktop/BIA4/BIA4_seg/TB_Chest_Radiography_Database/n_mask/"
n_originaldir="/Users/cuijiajun/Desktop/BIA4/BIA4_seg/TB_Chest_Radiography_Database/Normal/"
n_savedir="/Users/cuijiajun/Desktop/BIA4/BIA4_seg/TB_Chest_Radiography_Database/normal_final/"
#normal set
seg(n_maskdir,n_originaldir,n_savedir)

tb_maskdir="/Users/cuijiajun/Desktop/BIA4/BIA4_seg/TB_Chest_Radiography_Database/tb_mask/"
tb_originaldir="/Users/cuijiajun/Desktop/BIA4/BIA4_seg/TB_Chest_Radiography_Database/Tuberculosis_old/"
tb_savedir="/Users/cuijiajun/Desktop/BIA4/BIA4_seg/TB_Chest_Radiography_Database/tb_final/"
#train set
seg(tb_maskdir,tb_originaldir,tb_savedir)