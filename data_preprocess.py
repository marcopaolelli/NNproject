import sys
import glob
import os
import pickle
import pandas as pd
import numpy as np
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import gc
from PIL import Image
from random import shuffle

parent_dir = 'Dataset'
sub_dirs = ['fold-0','fold-1','fold-2','fold-3','fold-4']
specPath = 'Images/Spectrograms/'
slicesPath = 'Images/Slices/'
datasetPath = '/media/marco/Dati/Dataset3/'
genres = [0,1,2,3,4]
sliceSize = 128
validationRatio = 0.2
testRatio = 0.1

def createMelSpec(path):
    signal,sr = lb.load(path)
    signal = np.asarray(signal)
    spec = lb.feature.melspectrogram(y=signal,n_mels=128)
    melspec = lb.logamplitude(spec**2, ref_power=1.0)
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(melspec.shape[1],128)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    specTitle = os.path.splitext(os.path.basename(path))[0]+'_'+path.split('/')[1].split('-')[1]+'.png'
    lb.display.specshow(lb.power_to_db(spec,ref=np.max))
    fig.savefig('Images/Spectrograms/'+specTitle,dpi=1)
    fig.clf()
    plt.close()
    gc.collect()
    #plt.savefig('Images/Spectrograms/test.png', bbox_inches='tight', pad_inches=0)
    
    #plotMelspectrogram(signal)
    return specTitle

def plotMelspectrogram(signal):
    spec = lb.feature.melspectrogram(y=signal, n_mels=128)
    melspec = lb.logamplitude(spec**2, ref_power=1.0)
    plt.figure()
    lb.display.specshow(lb.power_to_db(spec,ref=np.max))
    plt.show()
"""
def sliceSpec(specTitle,size):
    #genre = os.path.splitext(os.path.basename(specTitle))[0].split("_")[-1]
    img = Image.open(specPath+specTitle)

    width, height = img.size
    nSamples = int(width/size)

    for i in range(nSamples):
        startPixel = i*size
        imgTmp = img.crop((startPixel, 1, startPixel + size, size + 1))
        imgTmp.save(slicesPath+"{}_{}.png".format(specTitle[:-4],i))
    return

"""

"""
folders = os.listdir(parent_dir)
for f in folders:
    filenames = os.listdir(parent_dir+'/'+f)
    filenames = [filename for filename in filenames if filename.endswith('.mp3')]
    for filename in filenames:
        ms = createMelSpec('Dataset/'+f+'/'+filename)
        sliceSpec(ms,128)
"""




def getDataset(mode):
    if not os.path.isfile(datasetPath+"train1_X_"+'Dataset2'+".p"):
        createDatasetFromSlices() 
    else:
        print("[+] Using existing dataset")
    
    return loadDataset(mode)

def loadDataset(mode):
    #Load existing
    datasetName = 'Dataset2'
    if mode == "train":
        print("[+] Loading training and validation datasets... ")
        train1_X = pickle.load(open("{}train1_X_{}.p".format(datasetPath,datasetName), "rb" ))
        train1_y = pickle.load(open("{}train1_y_{}.p".format(datasetPath,datasetName), "rb" ))
        train2_X = pickle.load(open("{}train2_X_{}.p".format(datasetPath,datasetName), "rb" ))
        train2_y = pickle.load(open("{}train2_y_{}.p".format(datasetPath,datasetName), "rb" ))
        validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath,datasetName), "rb" ))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Training and validation datasets loaded!")
        return train1_X, train1_y, train2_X, train2_y, validation_X, validation_y

    else:
        print("[+] Loading testing dataset... ")
        test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath,datasetName), "rb" ))
        test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Testing dataset loaded! ")
        return test_X, test_y

def saveDataset(train1_X, train1_y, train2_X, train2_y, validation_X, validation_y, test_X, test_y):
     #Create path for dataset if not existing
    
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc: # Guard against race condition
            #if exc.errno != errno.EEXIST:
                #raise

    #SaveDataset
    print("[+] Saving dataset... ")
    datasetName = 'Dataset2'
    pickle.dump(train1_X, open("{}train1_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(train1_y, open("{}train1_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(train2_X, open("{}train2_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(train2_y, open("{}train2_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(test_X, open("{}test_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(test_y, open("{}test_y_{}.p".format(datasetPath,datasetName), "wb" ))
    print("    Dataset saved! ")

def createDatasetFromSlices():
    data = []
    filenames = os.listdir(specPath)
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    shuffle(filenames)
    for filename in filenames:
        img = Image.open(specPath+filename)
        img = img.resize((217,64), resample=Image.ANTIALIAS)
        img.save('Images/prova/bbb'+filename)
        imgData = np.asarray(img, dtype=np.uint8).reshape(217,64,4)
        imgData = imgData/255.
        sliceGenre = int (os.path.splitext(os.path.basename(filename))[0].split("_")[-1])
        label = [1. if sliceGenre == g else 0. for g in genres]
        data.append((imgData,label))

    X,y = zip(*data)
        #Split data
    validationNb = int(len(X)*validationRatio)
    testNb = int(len(X)*testRatio)
    trainNb = (len(X)-(validationNb + testNb))    

    #print(X)
    #print(X.shape)
    #print(y)
    #print(y.shape)
    #Prepare for Tflearn at the same time

    print(validationNb)
    print(testNb)
    print (trainNb)

    train1_X = np.array(X[:trainNb/2]).reshape([-1, 217, 64, 4])
    train1_y = np.array(y[:trainNb/2])
    train2_X = np.array(X[trainNb/2:trainNb]).reshape([-1, 217, 64, 4])
    train2_y = np.array(y[trainNb/2:trainNb])
    validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, 217, 64, 4])
    validation_y = np.array(y[trainNb:trainNb+validationNb])
    test_X = np.array(X[-testNb:]).reshape([-1, 217, 64, 4])
    test_y = np.array(y[-testNb:])
    print("    Dataset created! ")
        
    #Save
    saveDataset(train1_X, train1_y, train2_X, train2_y, validation_X, validation_y, test_X, test_y)

    return train1_X, train1_y, train2_X, train2_y, validation_X, validation_y, test_X, test_y

#createDatasetFromSlices()