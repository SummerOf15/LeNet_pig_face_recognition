#coding:utf-8
'''

@author: zhangkai
'''
import tensorflow as tf

#define network parameters
INPUT_NODE=3600  #all pixels input
OUTPUT_NODE=30  #classes
IMAGE_SIZE=60
NUM_CHANNELS=1
NUM_LABELS=30
#first conv layer
CONV1_DEEP=32
CONV1_SIZE=5
#second conv layer
CONV2_DEEP=64
CONV2_SIZE=5
#full connected nodes
FC_SIZE=512

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer !=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable('weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.summary.histogram('layer1-conv1/weights', conv1_weights)
        conv1_biases=tf.get_variable('bias',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #size=5*5; stride=1; fill with 0;feature=32
        tf.summary.histogram('layer1-conv1/bias', conv1_biases)
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        #pool layer:output 30*30*32
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
        
    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.summary.histogram('layer3-conv2/weights', conv2_weights)
        conv2_biases=tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('layer2-conv1/bias', conv2_biases)
        #size=5*5 deep=64 stride=1
        conv2=tf.nn.conv2d(norm1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
        
    with tf.name_scope('layer4-pool2'):
        #output:15*15*64
        norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
        pool2=tf.nn.max_pool(norm2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[3]*pool_shape[1]*pool_shape[2]
    reshaped=tf.reshape(pool2, [pool_shape[0],nodes])
    
    with tf.variable_scope('layer5-fc1'):
        #output:
        fc1_weights=tf.get_variable('weight',[nodes,FC_SIZE],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases=tf.get_variable('bias',[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: fc1=tf.nn.dropout(fc1, 0.5)
        
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable('weight',[FC_SIZE,NUM_LABELS],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable('bias',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
    
    return logit
        
        