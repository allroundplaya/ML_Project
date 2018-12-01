import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from settings import *

learning_rate = 0.01
train_epochs = 3
batch_size = 100

X = tf.placeholder(tf.float32, [200, 200, 3])
Y = tf.placeholder(tf.float32, [None, 29])

"""
합성곱이나 풀링 층을 넣을 때, stride 를 [1, x, y, 1] 이런식으로 넣는 것을 볼 수 있는데
stride[0]과 stride[3]은 무시해도 된다고 합니다.
한 번에 filter 가 x 방향으로 얼마나 이동하는지, y 방향으로 얼마나 이동하는지만 명시해주면 된다고 합니다.
"""

# first Layer
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]), stddev=0.01)  # filter(kernel) 의 weight, 임의로 설정해 줍니다.

L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')  # vector of weighted sum
L1 = tf.nn.relu(L1)  # activation
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pooling

# second Layer
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # filter(kernel) 의 weight, 임의로 설정해 줍니다.

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')  # vector of weighted sum
L2 = tf.nn.relu(L2)  # activation
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pooling

# Fully Connected Layer (아마 softmax?)
L2 = tf.reshape(L2, [-1, 7*7*64])  # 앞의 -1은 2차원 벡터를 flatten 하기 위해 있는 것 n*(7*7*64) 차원으로 만드는데 여기서 아마 n이 1일 거임.
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer = tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))  # 편향

hypothesis = tf.matmul(L2, W3) + b

# Softmax 함수를 직접 사용하는 대신에 softmax_cross_entropy_with_logits을 사용할 수 있다.
# 인자로 logits과 label을 전달해주면 된다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_withlogits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# GD를 이전까지 썼지만
# 좀더 학습성과가 뛰어나다고 알려져 있는 Adam Optimizer를 사용한다.

sess = tf.Session()  # TensorFlow session
sess.run(tf.global_variables_initializer())  # 초기화를 합니다.

print('Learning started. It takes sometime.')
for epoch in range(train_epochs):
    avg_cost = 0
    # total_batch = (example 수 / batch_size)
    # for i in range(total_batch):
    #     feed_dict = {X:batch_xs, Y:batch_ys}
    #     c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    #     avg_cost += c / total_batch
    total_batch = 29
    for char in LETTERS:
        feed_dict = {X:np.load('./data/train_X_'+char+'.npy'), Y:np.load('./data/train_y_'+char+'.npy')}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print("Epoch:", "%04d" (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("learning finished")
