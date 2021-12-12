#-*-coding:utf-8-*-
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt

image_pathway = "/to/your/path/of/BIA4/TB_Chest_Radiography_Database/"
label_list ="Normal,Tuberculosis"
print("start...")
def process_image_channels(image):
    process_flag = False
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r,g,b))
        process_flag = True
    # process the channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
    else: image=None
    return image

if __name__ == '__main__':
    # only used at the first time to initialize the tfrecord data from raw data
    # in OS system, .DS_Store files need to be deleted first with the command below.
    # sudo find /to/your/path/of/BIA4/BIA4_seg/  -name ".DS_Store" -depth -exec rm {} \;

    cwd1 = image_pathway + 'test/'
    cwd2 = image_pathway + 'train/'
    cwd3 = image_pathway + 'val/'
    classes = label_list.split(",")
    for cwd in [cwd1, cwd2, cwd3]:

      for index, name in enumerate(classes):
        imagesDirectory = cwd + name + '/'
        for img_name in os.listdir(imagesDirectory):
            img_path = imagesDirectory + img_name  # pathway of each image
            img = Image.open(img_path)
            img=process_image_channels(img)
            if img!=None:

                img.save(imagesDirectory + img_name,"png")
                print(img_name)
                print("shape:" + str(np.shape(img)))  # (224, 224, 3)
            #plt.imshow(img)
            #plt.show()