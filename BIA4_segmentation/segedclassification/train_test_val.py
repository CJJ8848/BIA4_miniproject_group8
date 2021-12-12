import os, random, shutil


#set the pathway
base_dir = '/to/your/path/of/BIA4/TB_Chest_Radiography_Database/'
# remake the train, vaild and test dataset
Tuberculosis_dir = base_dir + 'Tuberculosis/'
Normal_dir = base_dir + "Normal/"
train_dir = base_dir + "train/"
valid_dir = base_dir + "val/"
test_dir = base_dir + "test/"

#train test validation divide
def moveFile(fileDir, tarDir, rate):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    picknumber = int(filenumber * rate)  # extract images with a curtain rate
    sample = random.sample(pathDir, picknumber)  # random pick

    if not os.path.exists(tarDir):
        os.makedirs(tarDir)

    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
if __name__ == '__main__':

    moveFile(Tuberculosis_dir, train_dir+"Tuberculosis/", 0.7)
    # for the remaining we put 2/3 into vaild and 1/3 into test
    moveFile(Tuberculosis_dir, valid_dir+"Tuberculosis/", 2 / 3)
    moveFile(Tuberculosis_dir, test_dir+"Tuberculosis/", 1)
    moveFile(Normal_dir, train_dir+"Normal/", 0.7)
    moveFile(Normal_dir, valid_dir+"Normal/", 2 / 3)
    moveFile(Normal_dir, test_dir+"Normal/", 1)