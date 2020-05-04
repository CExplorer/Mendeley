# -*- coding: utf-8 -*-
"""
Created on Fri May 31 2019
@author: Ruoyu Chen
The Inceptionv4 networks
"""
import tensorflow as tf
 
BATCH_SIZE = 10
 
 
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)
 
 
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)
 
 
def conv2d(input, filter, strides, padding="SAME", name=None):
    # filters with shape [filter_height * filter_width * in_channels, output_channels]
    # Must have strides[0] = strides[3] =1
    # For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]
    '''
    Args:
        input: A Tensor. Must be one of the following types: float32, float64.
        filter: A Tensor. Must have the same type as input.
        strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        use_cudnn_on_gpu: An optional bool. Defaults to True.
        name: A name for the operation (optional).
    '''
    return tf.nn.conv2d(input, filter, strides, padding=padding, name=name)  # padding="SAME"用零填充边界

def Conv(input, name, filter_size, bias_size, stride, padding = 'SAME'):
    with tf.name_scope(name):
        with tf.name_scope('Variable'):
            filters = weight_variable(filter_size, name='filter')
            bias = weight_variable(bias_size, name='bias')
        with tf.name_scope("Convolution"):
            layer = tf.nn.relu(conv2d(input, filters, strides=stride, padding = padding) + bias)
    return layer

def stem(image):
    # input shape(299,299,3)
    with tf.name_scope('steam'):
        net = Conv(input=image, name = 'Layer1', filter_size = [3,3,3,32], bias_size = [32],padding = 'VALID', stride = [1,2,2,1])
        net = Conv(input=net, name = 'Layer2', filter_size = [3,3,32,32], bias_size = [32], padding = 'VALID', stride = [1,1,1,1])
        net = Conv(input=net, name = 'Layer3', filter_size = [3,3,32,64], bias_size = [64], padding = 'SAME', stride = [1,1,1,1])
        with tf.name_scope('MaxPool_1'):
            maxpool = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID",name='maxpool')
        net = Conv(input=net, name = 'Layer4', filter_size = [3,3,64,96], bias_size = [96], padding="VALID", stride = [1,2,2,1])
        with tf.name_scope('Filter_concat_1'):
            net = tf.concat([maxpool, net], 3)
        net_1 = Conv(input=net, name = 'Layer5_1', filter_size = [1,1,160,64], bias_size = [64], padding="SAME", stride = [1,1,1,1])
        net_1 = Conv(input=net_1, name = 'Layer6_1', filter_size = [3,3,64,96], bias_size = [96], padding="VALID",stride = [1,1,1,1])
        net_2 = Conv(input=net, name = 'Layer5_2', filter_size = [1,1,160,64], bias_size = [64], stride = [1,1,1,1])
        net_2 = Conv(input=net_2, name = 'Layer6_2', filter_size = [7,1,64,64], bias_size = [64], stride = [1,1,1,1])
        net_2 = Conv(input=net_2, name = 'Layer7_2', filter_size = [1,7,64,64], bias_size = [64], stride = [1,1,1,1])
        net_2 = Conv(input=net_2, name = 'Layer8_2', filter_size = [3,3,64,96], bias_size = [96], padding="VALID", stride = [1,1,1,1])
        with tf.name_scope('Filter_concat_2'):
            net = tf.concat([net_1, net_2], 3)
        net_1 = Conv(input=net, name = 'Layer9', filter_size = [3,3,192,192], bias_size = [192], padding="VALID", stride = [1,2,2,1])
        with tf.name_scope('MaxPool_2'):
            maxpool = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID",name='maxpool')
        with tf.name_scope('Filter_concat_3'):
            net = tf.concat([net_1, maxpool], 3)
    return net

def inception_A(input, name):
    input_shape=input.get_shape().as_list()
    with tf.name_scope(name):
        with tf.name_scope('Average_Pool'):
            avg = tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME",name='avgpool')
        net1 = Conv(input=avg, name = 'Layer1_1', filter_size = [1,1,input_shape[3],96], bias_size = [96],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=input, name = 'Layer2_1', filter_size = [1,1,input_shape[3],96], bias_size = [96],padding = 'SAME', stride = [1,1,1,1])
        net3 = Conv(input=input, name = 'Layer3_1', filter_size = [1,1,input_shape[3],64], bias_size = [64],padding = 'SAME', stride = [1,1,1,1])
        net3 = Conv(input=net3, name = 'Layer3_2', filter_size = [3,3,64,96], bias_size = [96],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=input, name = 'Layer4_1', filter_size = [1,1,input_shape[3],64], bias_size = [64],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_2', filter_size = [3,3,64,96], bias_size = [96],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_3', filter_size = [3,3,96,96], bias_size = [96],padding = 'SAME', stride = [1,1,1,1])
        with tf.name_scope('Filter_concat'):
            net = tf.concat([net1, net2, net3, net4], 3)
    return net

def Reduction_A(input, n, k, I, m):
    input_shape=input.get_shape().as_list()
    with tf.name_scope('ReductionA'):
        with tf.name_scope('MaxPool'):
            maxpool = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2,2, 1], padding="VALID",name='MaxPool')
        net1 = Conv(input=input, name = 'Layer1', filter_size = [3,3,input_shape[3],n], bias_size = [n],padding = 'VALID', stride = [1,2,2,1])
        net2 = Conv(input=input, name = 'Layer2_1', filter_size = [1,1,input_shape[3],k], bias_size = [k],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=net2, name = 'Layer2_2', filter_size = [3,3,k,I], bias_size = [I],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=net2, name = 'Layer2_3', filter_size = [3,3,I,m], bias_size = [m],padding = 'VALID', stride = [1,2,2,1])
        with tf.name_scope('Filter_concat'):
            net = tf.concat([maxpool ,net1, net2], 3)
    return net

def inception_B(input, name):
    input_shape=input.get_shape().as_list()
    with tf.name_scope(name):
        with tf.name_scope('Average_Pool'):
            avg = tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME",name='avgpool')
        net1 = Conv(input=avg, name = 'Layer1_1', filter_size = [1,1,input_shape[3],128], bias_size = [128],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=input, name = 'Layer2_1', filter_size = [1,1,input_shape[3],384], bias_size = [384],padding = 'SAME', stride = [1,1,1,1])
        net3 = Conv(input=input, name = 'Layer3_1', filter_size = [1,1,input_shape[3],192], bias_size = [192],padding = 'SAME', stride = [1,1,1,1])
        net3 = Conv(input=net3, name = 'Layer3_2', filter_size = [1,7,192,224], bias_size = [224],padding = 'SAME', stride = [1,1,1,1])
        net3 = Conv(input=net3, name = 'Layer3_3', filter_size = [7,1,224,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=input, name = 'Layer4_1', filter_size = [1,1,input_shape[3],192], bias_size = [192],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_2', filter_size = [1,7,192,192], bias_size = [192],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_3', filter_size = [7,1,192,224], bias_size = [224],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_4', filter_size = [1,7,224,224], bias_size = [224],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_5', filter_size = [7,1,224,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        with tf.name_scope('Filter_concat'):
            net = tf.concat([net1, net2, net3, net4], 3)
    return net

def Reduction_B(input):
    input_shape=input.get_shape().as_list()
    with tf.name_scope('ReductionB'):
        with tf.name_scope('MaxPool'):
            maxpool = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2,2, 1], padding="VALID",name='MaxPool')
        net1 = Conv(input=input, name = 'Layer1_1', filter_size = [1,1,input_shape[3],192], bias_size = [192],padding = 'SAME', stride = [1,1,1,1])
        net1 = Conv(input=net1, name = 'Layer1_2', filter_size = [3,3,192,192], bias_size = [192],padding = 'VALID', stride = [1,2,2,1])
        net2 = Conv(input=input, name = 'Layer2_1', filter_size = [1,1,input_shape[3],256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=net2, name = 'Layer2_2', filter_size = [1,7,256,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=net2, name = 'Layer2_3', filter_size = [7,1,256,320], bias_size = [320],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=net2, name = 'Layer2_4', filter_size = [3,3,320,320], bias_size = [320],padding = 'VALID', stride = [1,2,2,1])
        with tf.name_scope('Filter_concat'):
            net = tf.concat([maxpool, net1, net2], 3)
    return net

def inception_C(input, name):
    input_shape=input.get_shape().as_list()
    with tf.name_scope(name):
        with tf.name_scope('Average_Pool'):
            avg = tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME",name='avgpool')
        net1 = Conv(input=avg, name = 'Layer1_1', filter_size = [1,1,input_shape[3],256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net2 = Conv(input=input, name = 'Layer2_1', filter_size = [1,1,input_shape[3],256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net3 = Conv(input=input, name = 'Layer3_1', filter_size = [1,1,input_shape[3],384], bias_size = [384],padding = 'SAME', stride = [1,1,1,1])
        net3_1 = Conv(input=net3, name = 'Layer3_2_1', filter_size = [1,3,384,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net3_2 = Conv(input=net3, name = 'Layer3_2_2', filter_size = [3,1,384,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=input, name = 'Layer4_1', filter_size = [1,1,input_shape[3],384], bias_size = [384],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_2', filter_size = [1,3,384,448], bias_size = [448],padding = 'SAME', stride = [1,1,1,1])
        net4 = Conv(input=net4, name = 'Layer4_3', filter_size = [3,1,448,512], bias_size = [512],padding = 'SAME', stride = [1,1,1,1])
        net4_1 = Conv(input=net4, name = 'Layer4_4_1', filter_size = [3,1,512,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        net4_2 = Conv(input=net4, name = 'Layer4_4_2', filter_size = [1,3,512,256], bias_size = [256],padding = 'SAME', stride = [1,1,1,1])
        with tf.name_scope('Filter_concat'):
            net = tf.concat([net1, net2, net3_1, net3_2, net4_1, net4_2], 3)
    return net

def layer(input, name, size):
    input_shape=input.get_shape().as_list()
    with tf.name_scope(name):
        with tf.name_scope('Variable'):
            W = weight_variable([input_shape[1], size], name='Weight')
            bias = weight_variable([size], name='bias')
        with tf.name_scope("layer"):
            layer = tf.nn.relu(tf.matmul(input, W, name='layer') + bias)
    return layer

def Inception_v4(input, keep_prob=0.8):
    '''参考论文: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'''
    # input_size:(299,299,3)
    net = stem(input)
    net = inception_A(net, 'Inception_A1')
    net = inception_A(net, 'Inception_A2')
    net = inception_A(net, 'Inception_A3')
    net = inception_A(net, 'Inception_A4')
    net = Reduction_A(net, n = 96, k = 96, I = 224, m = 256)
    net = inception_B(net, 'Inception_B1')
    net = inception_B(net, 'Inception_B2')
    net = inception_B(net, 'Inception_B3')
    net = inception_B(net, 'Inception_B4')
    net = inception_B(net, 'Inception_B5')
    net = inception_B(net, 'Inception_B6')
    net = inception_B(net, 'Inception_B7')
    net = Reduction_B(net)
    net = inception_C(net, 'Inception_C1')
    net = inception_C(net, 'Inception_C2')
    net = inception_C(net, 'Inception_C3')
    with tf.name_scope('Average_Pooling'):
        net = tf.nn.avg_pool(net, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID",name='avgpool')
        net = tf.reshape(net,[-1,1536])
    with tf.name_scope('Dropout'):
        net = tf.nn.dropout(net, keep_prob)
    with tf.name_scope('Softmax'):
        net = layer(net, name='Layer', size = 1000)
    return net
 
def backward(datasets, label, test_data, test_label):
    with tf.name_scope('Input_data'):
        X = tf.placeholder(tf.float32, [None, 299, 299, 3], name="Input")
        Y_ = tf.placeholder(tf.float32, [None, 1], name='Estimation')
    LEARNING_RATE_BASE = 0.00001  # 最初学习率
    LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
    LEARNING_RATE_STEP = 1000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
    gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps, LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY, staircase=True)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    y = VGG16(X)
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    with tf.name_scope('loss'):
        loss_mse = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(y), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(init_op)
        # 训练模型。
        STEPS = 500001
        min_loss = 1
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % len(datasets)
            end = start + BATCH_SIZE
            if i % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_str, step, _ = sess.run([merged, gloabl_steps, train_step],
                                                feed_dict={X: datasets[start:end], Y_: label[start:end], keep_prob:1.0},
                                                options=run_options, run_metadata=run_metadata)
                writer.add_summary(summary_str, i)
                writer.add_run_metadata(run_metadata, 'step%d' % (i))
                test_accuracy = accuracy.eval(feed_dict={X: test_data, Y_: test_label, keep_prob: 1.0})
                print("After %d training step(s), accuracy is %g" % (i, test_accuracy))
                saver.save(sess, './logs/variable', global_step=i)
            else:
                summary_str, step, _ = sess.run([merged, gloabl_steps, train_step],
                                                feed_dict={X: datasets[start:end], Y_: label[start:end]})
                writer.add_summary(summary_str, i)
 
 
def main():
    with tf.name_scope('Input_data'):
        X = tf.placeholder(tf.float32, [None, 299, 299, 3], name="Input")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    y = Inception_v4(X, keep_prob)
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs", sess.graph)
    writer.close()
 
 
if __name__ == '__main__':
    main()