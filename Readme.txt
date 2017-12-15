This is a project about pig face recognition. It is based on Lenet and trained on CPU. 

Datasets: http://jddjr.jd.com/item/4

Environments:
python3.5
opencv3.3.1
tensorflow 1.4.0
six 1.11.0
numpy 1.13.3

datasets path:
you should do: 1.put videos under videos/  2.put test images under dir test/testB/
train videos:videos/
test datasets:test/testB/
the following data is generated auto create
train datasets preprocess results: images/
test datasets preprocess results:test/testB1/
validation datasets:validation/

how to run:
1、generate images:
python conver2images.py
2、train
python pigs_restore_train.py
3、predict
python pigs_predict.py