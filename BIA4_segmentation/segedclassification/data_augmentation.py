# -*-coding:utf-8-*-
import os

from PIL import Image, ImageChops

#translation
def ImgOfffSet(Img,xoff,yoff):
    width, height = Img.size
    c = ImageChops.offset(Img,xoff,yoff)
    c.paste((0,0,0),(0,0,xoff,height))
    c.paste((0,0,0),(0,0,width,yoff))
    return c
# main function to augment the data
def data_aug(image_pathway):
    for img_name in os.listdir(image_pathway):
        img_path = image_pathway + img_name  # pathway of each image
        img = Image.open(img_path)
        out1 = img.rotate(-20)
        out2 = img.rotate(10)
        out3 = img.rotate(30)
        out4 = ImgOfffSet(img,20,0)
        #rotate -20 degree
        out1.save(image_pathway+str(img_name)+"-ro-20.png")
        #rotate 10 degree
        out2.save(image_pathway+str(img_name)+"-ro10.png")
        #rotate 30 degree
        out3.save(image_pathway+str(img_name)+"-ro30.png")
        #translation of 20
        out4.save(image_pathway+str(img_name)+"-off20.png")
train_tb_image_pathway="/to/your/path/of/BIA4/BIA4_seg/TBtest2/Normal/"
data_aug(train_tb_image_pathway)