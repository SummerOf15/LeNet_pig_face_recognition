#coding:utf-8
'''
Created on 2017年11月25日

@author: Administrator
'''
import cv2
import numpy as np
import os
import base

class DataSet(object):

    def __init__(self,images,labels,dtype=np.float32,reshape=True,):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples=images.shape[0]
        if reshape:
            images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
#             images = images.reshape(images.shape[0],images.shape[1] * images.shape[2]*images.shape[3])
        if dtype==np.uint32:
        # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 :
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]  
          
          
          
def extract_images(img_dir):
    num_images=0
    fs=[]
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            fs.append(file)
    num_images=len(fs)
    print (num_images)
    i=0
#     data=np.ndarray((num_images,60,60,3),dtype=np.uint8)
    data=np.ndarray((num_images,60,60),dtype=np.uint8)
    for img_name in fs:
        img=cv2.imread(img_dir+img_name,0)
        if img.size<1:
            print('file %s not existed'%(img_dir+img_name))
            break
#         data[i,:,:,:]=img
        data[i,:,:]=img
        i+=1
    return data
def extract_labels(img_dir):
    num_images=0
    fs=[]
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            fs.append(file)
    num_images=len(fs)
    labels=np.ndarray((num_images,1),dtype=np.uint8)
    for i in range(num_images):
        dir=img_dir.split('/')
        labels[i]=int(dir[1])-1
    return labels

def read_data_set(train_dir,validation_dir):
    dirs=[]
    for file in os.listdir(train_dir):  
        file_path = os.path.join(train_dir, file)  
        if os.path.isdir(file_path):  
            dirs.append(file_path+'/')
    num_dirs=len(dirs)
#     train_images=np.ndarray((0,60,60,3),dtype=np.uint8)
    train_images=np.ndarray((0,60,60),dtype=np.uint8)
    train_labels=np.ndarray((0,1),dtype=np.uint8)
    for i in range(num_dirs):
        image_data=extract_images(str(dirs[i]))
        image_labels=extract_labels(str(dirs[i]))
        train_images=np.vstack((train_images,image_data))
        train_labels=np.vstack((train_labels,image_labels))
    dirs=[]
    for file in os.listdir(validation_dir):  
        file_path = os.path.join(validation_dir, file)  
        if os.path.isdir(file_path):  
            dirs.append(file_path+'/')
#     validation_images=np.ndarray((0,60,60,3),dtype=np.uint8)
    validation_images=np.ndarray((0,60,60),dtype=np.uint8)
    validation_labels=np.ndarray((0,1),dtype=np.uint8)
    for i in range(num_dirs):
        image_data=extract_images(str(dirs[i]))
        image_labels=extract_labels(str(dirs[i]))
        validation_images=np.vstack((validation_images,image_data))
        validation_labels=np.vstack((validation_labels,image_labels))
#     if not 0 <= validation_size <= len(train_images):
#         raise ValueError(
#             'Validation size should be between 0 and {}. Received: {}.'
#             .format(len(train_images), validation_size))
    
#     validation_images = train_images[:validation_size]
#     validation_labels = train_labels[:validation_size]
#     train_images = train_images[validation_size:]
#     train_labels = train_labels[validation_size:]
    
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    return base.Datasets(train=train,validation=validation)
if __name__=='__main__':
    print('start!')
    read_data_set('images/','validation/')
    print('Done!')