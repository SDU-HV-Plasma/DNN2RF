import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
import csv

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#Normalization

#0-1
dfn1 = pd.read_csv('Normalization 0-1.CSV',encoding='gbk')
xn1=dfn1[['Cycles','Current density']]
x1_scaler=MinMaxScaler(feature_range=(0,1))
Xn1=x1_scaler.fit_transform(xn1)

#1-2
dfn2 = pd.read_csv('Normalization 1-2.CSV',encoding='gbk')
xn2=dfn2[['Cycles','Current density']]
x2_scaler=MinMaxScaler(feature_range=(0,1))
Xn2=x2_scaler.fit_transform(xn2)

#Train set

df = pd.read_csv('Train set.CSV',encoding='gbk')

#0-1
a=df[df.Cycles>=0]
dfa=a[a.Cycles<=1]
xa=dfa[['Cycles','Current density']]
xaa=x1_scaler.transform(xa)
xa_data=np.array(xaa,dtype='float32')
ya=dfa[['Voltage']]
ya_data=np.array(ya,dtype='float32')

#1-2
b=df[df.Cycles>=1]
dfb=b[b.Cycles<=2]
xb=dfb[['Cycles','Current density']]
xbb=x2_scaler.transform(xb)
xb_data=np.array(xbb,dtype='float32')
yb=dfb[['Voltage']]
yb_data=np.array(yb,dtype='float32')

#Test set

dft = pd.read_csv('Test set 112.5mA.CSV',encoding='gbk')

#0-1
at=dft[dft.Cycles>=0]
dfat=at[at.Cycles<1]
xat=dfat[['Cycles','Current density']]
xaat=x1_scaler.transform(xat)
xat_data=np.array(xaat,dtype='float32')
yat=dfat[['Voltage']]
yat_data=np.array(yat,dtype='float32')

#1-2
bt=dft[dft.Cycles>=1]
dfbt=bt[bt.Cycles<=2]
xbt=dfbt[['Cycles','Current density']]
xbbt=x2_scaler.transform(xbt)
xbt_data=np.array(xbbt,dtype='float32')
ybt=dfbt[['Voltage']]
ybt_data=np.array(ybt,dtype='float32')

# Input

xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,1])
xs2 = tf.placeholder(tf.float32, [None,2])
ys2 = tf.placeholder(tf.float32, [None,1])

# Hidden layer

l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

l5 = add_layer(xs2, 2, 30, activation_function=tf1.nn.relu)
l6 = add_layer(l5, 30, 30, activation_function=tf1.tanh)
l7 = add_layer(l6, 30, 30, activation_function=tf1.tanh)
l8 = add_layer(l7, 30, 30, activation_function=tf1.sigmoid)

# Output

prediction1 = add_layer(l4, 30, 1, activation_function=None)
prediction2 = add_layer(l8, 30, 1, activation_function=None)

# Train loss

loss1 = tf.reduce_mean(tf.square(ys1 - prediction1))
loss2 = tf.reduce_mean(tf.square(ys2 - prediction2))

train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)

# Computation loss

lossc1 = 100*tf.reduce_mean(tf.abs((ys1 - prediction1)/ys1))
lossc2 = 100*tf.reduce_mean(tf.abs((ys2 - prediction2)/ys2))

# Train

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    is_train = False
    is_mod = False
    saver = tf.train.Saver(max_to_keep=1)

    if is_train:
      if is_mod:
        model_file1 = tf1.train.latest_checkpoint('save1/')
        saver.restore(sess, model_file1)
        for i in range(100001):
         sess.run(train_step1, feed_dict={xs1: xa_data, ys1: ya_data})
         if i % 100 == 0:
          print(sess.run(loss1, feed_dict={xs1: xa_data, ys1: ya_data}))

        saver.save(sess, 'save1/model1', global_step=i + 1)
      else:
           model_file2 = tf1.train.latest_checkpoint('save2/')
           saver.restore(sess, model_file2)
           for i in range(100001):
               sess.run(train_step2, feed_dict={xs2: xb_data, ys2: yb_data})
               if i % 100 == 0:
                   print(sess.run(loss2, feed_dict={xs2: xb_data, ys2: yb_data}))

           saver.save(sess, 'save2/model2', global_step=i + 1)

# Computation
    else:
        model_file1=tf1.train.latest_checkpoint('save1/')
        saver.restore(sess,model_file1)
        print(sess.run(lossc1, feed_dict={xs1: xat_data, ys1: yat_data}))
        with open("Voltage 112.5mA.csv","w",newline='') as f:
         b_csv = csv.writer(f)
         b_csv.writerows(sess.run(prediction1, feed_dict={xs1: xat_data}))

         model_file2 = tf1.train.latest_checkpoint('save2/')
         saver.restore(sess, model_file2)
         print(sess.run(lossc2, feed_dict={xs2: xbt_data, ys2: ybt_data}))
         b_csv.writerows(sess.run(prediction2, feed_dict={xs2: xbt_data}))