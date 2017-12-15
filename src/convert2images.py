#coding:utf-8
#Bin GAO

'''
This script build a function for converting the video into images
'''

import os
import cv2

#convert video to image
#u can use time.sleep(xx), if u just want to take 5 or 6.. images/s.
def convert_to_images(input_video_file,img_path,timesleep):
    cam = cv2.VideoCapture(input_video_file)
    counter = 0
    t=0;
    while True:
        flag, frame = cam.read()
        t+=1
        if flag:
            if t%timesleep==0:
                resImg=cv2.resize(frame,(int(60*1280/720),60),interpolation=cv2.INTER_AREA)
                subImg=resImg[:,20:80]
                cv2.imwrite(os.path.join(img_path, str(counter) + '.jpg'),subImg)
                counter = counter + 1
        else:
            break
        if cv2.waitKey(1) == 27:
            break
                # press esc to quit
    cv2.destroyAllWindows()
def generate_train_datasets():
    print('generate train datasets')
    #video sequences name
    for i in range(30,31):
        input='videos/%d.mp4'%i
        outputdir='images/%d/'%i
        if(not os.path.exists(outputdir)):
            os.mkdir(outputdir)
        if(not os.path.exists(input)):
            print ('%s is not exited'%(input))
            break
        timesleep=40
        convert_to_images(input, outputdir,timesleep)
def generate_validation_datasets():
    for i in range(10,31):
        input='videos/%d.mp4'%i
        outputdir='validation/%d/'%i
        if(not os.path.exists(outputdir)):
            os.mkdir(outputdir)
        if(not os.path.exists(input)):
            print ('%s is not exited'%(input))
            break
        timesleep=200
        convert_to_images(input, outputdir,timesleep)
def extract_test_images(input_img_dir,output_img_dir):
    for i in range(1,6001):
        image_name=str(i)+'.JPG'
        image_path=os.path.join(input_img_dir,image_name)
        save_path=os.path.join(output_img_dir,image_name)
        if os.path.exists(image_path):
            img=cv2.imread(image_path)
            if img.shape[0]>img.shape[1]:
                resImg=cv2.resize(img,(int(60*img.shape[0]/img.shape[1]),60),interpolation=cv2.INTER_AREA)
            else:
                resImg=cv2.resize(img,(60,int(60*img.shape[1]/img.shape[0])),interpolation=cv2.INTER_AREA)
            reshapeX=int(resImg.shape[0]/2)
            reshapeY=int(resImg.shape[1]/2)
            subImg=resImg[reshapeX-30:reshapeX+30,reshapeY-30:reshapeY+30]
            cv2.imwrite(save_path,subImg)
if __name__ == '__main__':
    print('Process start...')
    generate_train_datasets()
    generate_validation_datasets()
    extract_test_images('test/testB/','test/testB1/')
    print('Done')