#coding:utf-8
'''
Created on 2017年12月4日

@author: Administrator
'''
import time
import tensorflow as tf
import pigs_inference
import numpy as np
import pigs_train
import os
import cv2
import csv

PREDICT_BATCH=10
def predict(testImg,imgname):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32, [PREDICT_BATCH,pigs_inference.IMAGE_SIZE,pigs_inference.IMAGE_SIZE,
                                      pigs_inference.NUM_CHANNELS],name='x-input')
        reshaped_xs=np.reshape(testImg, (PREDICT_BATCH,
                                   pigs_inference.IMAGE_SIZE,
                                   pigs_inference.IMAGE_SIZE,
                                   pigs_inference.NUM_CHANNELS))
        feed_dict1={x:reshaped_xs}
        y=pigs_inference.inference(x, False, None)
        variable_averages=tf.train.ExponentialMovingAverage(pigs_train.MOVING_AVERAGE_DECAY)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore)
        
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(pigs_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                a=sess.run([y],feed_dict=feed_dict1)
                save_to_csv(imgname,a)
        
def save_to_csv(imgName,possibility):
    csvfile=open('resultB.csv','a',encoding='utf8',newline='')
    writer=csv.writer(csvfile)
    assert len(imgName)==PREDICT_BATCH,('error: length of list imgName')
    assert len(possibility)==1,('error about possibility')
    r=0
    
    for t in range(len(imgName)):
        for i in range(30):
            poss=possibility[0][r,:]
            minNum=min(poss)
            if(minNum<0):
                offset=abs(minNum)
                poss[:]=poss[:]+offset
            sums=sum(poss)
            poss[:]=poss[:]/sums
            row=[imgName[t],i+1,poss[i]]
            writer.writerow(row)
        r+=1
    csvfile.close()
    
def main(argv=None):
    img_data=np.ndarray((0,60*60*1),dtype=np.uint8)
    num_img=0
    imgname=[]
    print('start to write data to csv')
    for i in range(1,6001):
        if i%500==0:
            print(i)
        image_name=str(i)+'.JPG'
        image_path=os.path.join('test/testB1/',image_name)
        if os.path.exists(image_path):
            num_img+=1
            img=cv2.imread(image_path,0)
            #assert img.shape[3]==1,('img channels are not equal to one')
            img1=np.reshape(img,(1,img.shape[0]*img.shape[1]*1))
            img_data=np.vstack((img_data,img1))
            imgname.append(i)
            if num_img%PREDICT_BATCH==0:
                predict(img_data,imgname)#error: just for once
                imgname.clear()
                img_data
                img_data=np.ndarray((0,60*60*1),dtype=np.uint8)
    print('Done')
if __name__=='__main__':
    tf.app.run()