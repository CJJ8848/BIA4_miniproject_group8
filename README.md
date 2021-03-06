# BIA4

# BIA project
**Group8**
 
## Introduction

Welcome! Our project focused to identify the tuberculosis based on chest radiography images.This folder contains the project report, source codes, trained Unet model and a CNN model for Tuberculosis image classification and a command line executable file allow users to use our pre-trained model to classify their chest radiography images. 

**The language version used for segmentaion codes is python 3.8 with tensorflow=2.7.0, keras=2.7.0.**
**The language version used for classification codes is python 3.7 with tensorflow=1.13.1, keras=2.3.1.**


All the data and classification h5 model files are stored in the ZJE institute server (because of the upload file size limitation of github),
you can run the command 'scp -r 3180110107bit@10.105.100.153:/public/workspace/3180110107bit/BIA4remote to/your/path/' and then type the password '111111' to download the data and models.


-   **BIA4 group 8 group report.docx**: Project report

1. Segmentation
-	**seg_3.h5**: The trained Unet model for Lung segmentation in X-ray
-	**self_model.py**: Main source code for Lung segmentation, contains codes and discriptions of image preprocessing, model construction, accuracy evaluation & visaulization
-	**load_model and segment.py**: Codes to load the pretrained Unet model to produce images of the segmented lungs using classification dataset
-	**Segmentation.py**: a tool for users to segment their own dataset using our model.

2. Classification
-	**train_test_val.py**: divide the dataset into train(0.7),test(0.1) and val(0.2) subsets.
-	**gray2rgb.py**: used to tranform the format of images into RGB.
-	**data_augmentation.py**: the file used to perform data augmentation in TB training set.
-	**CNN_3layer.py**: used to train the best model
-	**ROC.py**: plot the ROC curve
-	**GUI.py**: the GUI 
-	**GUI documentation.doc**: the GUI operating guide
## Getting Started 

### Segmentation

To segment your images with our model, you can simply run the file 'Segmentation.py' by:
```text
python Segmentation.py --input /directory_of_images --output /directory_to_save
```
### Classification
running example of the "CNN_3layer.py" scripts:

  just run the commend 'python /path/of/CNN_3layer.py' and then follow the tips to input image dataset pathway and the save pathway of h5 model.

### GUI
Operating guide of GUI:

To open the GUI, you can type in ???python3 /to/your/path/of/GUI.py??? into command line.

Details of operating guide can be found in 'GUI documentation.doc'.
