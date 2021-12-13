import numpy as np
import tensorflow as tf
from tensorflow import keras
from cv2 import imread, createCLAHE, imwrite 
import os
import cv2
import argparse


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
# Create the parser
parser = argparse.ArgumentParser()

loaded_model = keras.models.load_model("seg_3.h5",custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})

#input directory
parser.add_argument('--input', type=str, required=True)
#output directory
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
    
def load_image(images_dir, img_size,suffix='jpeg',dtype=np.uint8):
    images_path = []
    for fname in os.listdir(images_dir):
        if fname.endswith(suffix) and not fname.startswith('.'):
            images_path.append(os.path.join(images_dir, fname))
    images_path = sorted(images_path)
    images_num = len(os.listdir(images_dir))
    images = []
    #print(images[0].shape)
    for i, path in enumerate(images_path):
        #judge the dimension of input image
        if len(imread(path).shape) ==3:
           img = cv2.resize(cv2.imread(path),img_size)[:,:,0]
           images.append(img)
        if len(imread(path).shape) ==2:
           img = cv2.resize(cv2.imread(path),img_size)
           images.append(img)

    #if len(images)<16 == True:
     #   print("No enough png files for prediction.")
    images = np.array(images).reshape(len(images),512,512,1) / 255.0
    #normalization image to (0,1)
    return images, images_path

if os.path.exists(args.input) == False:
    print("Input directory doesn't exist.")
if os.path.exists(args.output) == False:
    print("Output directory doesn't exist.")
else: img_size = (512,512)
input_images,input_path= load_image(images_dir=args.input, img_size=img_size,suffix='.png')
preds=loaded_model.predict(input_images)
preds=preds*255


#load the pretrained model
loaded_model = keras.models.load_model("seg_3.h5",custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
loaded_model.summary()

#implementation of the ostu algorithm to seperate threshold of mask
#it fails here due to the version collision
#def otsu(img):
    #caluculate each pixel's weight
   # mean_weight=1/(img.shape[0] * img.shape[1])
    #counts, bins = np.histogram(img,bins=range(0,256))

    #initialize intra-class variance(mas)
   # inter_var=-1
    #final_thresh=-1

   # for bin in bins[1:-1]:
        #compute class probability
      #  pc1=np.sum(counts[:bin])
      #  pc2=np.sum(counts[bin:])
        #class mean
       
      #  m1=pc1 * mean_weight
      #  m2=pc2 * mean_weight
        #calculate class mean = weighted-class p / class p
      #  mc1=np.sum(i * counts[i] for i in range(0,bin)) / m1
      #  mc2=np.sum(i * counts[i] for i in range(bin,255)) / m2
      #  final_value=pc1*pc2*(mc1-mc2)**2

        #iterate for the maximun inter-class variance
     #   if final_value > inter_var:
     #       final_thresh = bin
     #       inter_var=final_value

    #return final_thresh

def otsu_new(gray_img):
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    threshold_t = 0
    max_g = 0
    for t in range(255):
        n0 = gray_img[np.where(gray_img < t)]
        n1 = gray_img[np.where(gray_img>=t)]
        w0 = len(n0)/(h*w)
        #w1: the prportion of target in the image
        w1 = len(n1)/(h*w)
        #w1:the proportion of background in the image
        u0 = np.mean(n0) if len(n0)>0 else 0
        u1 = np.mean(n1) if len(n1)>0 else 0
    
        g = w0*w1*(u0-u1)**2
        if g > max_g :
            max_g = g
            threshold_t = t
    #print ('biggest difference between class：',threshold_t)
    gray_img[gray_img<threshold_t] = 0
    gray_img[gray_img>threshold_t] = 255
    return gray_img,threshold_t


for i in range(input_images.shape[0]):
    predi = preds[i]
    segmented_lung,threshold=otsu_new(predi)
    #instead of juding ever pixes, using 0/255 characteristics of segmented lung 
    img_2 = segmented_lung*input_images[i]
    #for j in range(segmented_lung.shape[0]):
     #   for k in range(segmented_lung.shape[1]):
      #      if segmented_lung[j][k]>=threshold:
       #         segmented_lung[j][k] == ori_img[j][k]
    cv2.imwrite(args.output+"/"+"{0}".format(input_path[i].split('/')[-1]),img_2)


