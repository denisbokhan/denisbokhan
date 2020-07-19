#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:24:12 2020

@author: denis
"""
#-----------------------Импорт модулей и отключение отладчика-------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
#-------------------------------------------------------------------------

#----------Размер мини-батча и число эпох и число картинок----------------
batch_size = 100
num_steps = 1000
n_samples = 60000
n_test = 10000
#-------------------------------------------------------------------------

#-------Загружаем изображения вещей - тренировачный и тестовый наборы------
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

fig1 = x_train[0]
fig2 = x_train[1]
fig3 = x_train[2]
fig4 = x_train[17]
fig5 = x_train[4]
fig6 = x_train[5]
fig7 = x_train[6]
fig8 = x_train[7]
fig9 = x_train[15]
fig10 = x_train[9]


x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)



#-------------------Для примера изобразим 10 цифр на графиках------------


#print(fig1.shape)

plt.figure()
plt.imshow(fig1)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(fig2)
plt.colorbar()
plt.grid(False)
plt.show()


plt.figure()
plt.imshow(fig3)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(fig4)
plt.colorbar()
plt.grid(False)
plt.show()


plt.figure()
plt.imshow(fig5)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(fig6)
plt.colorbar()
plt.grid(False)
plt.show()


plt.figure()
plt.imshow(fig7)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(fig8)
plt.colorbar()
plt.grid(False)
plt.show()


plt.figure()
plt.imshow(fig9)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(fig10)
plt.colorbar()
plt.grid(False)
plt.show()
#-------------------------------------------------------------------------

ttt  = tf.reshape(x_train,[-1,784])
uuu =  tf.reshape(x_test,[-1,784])
print(ttt.shape)
print(uuu.shape)

#----------------------Выделение памяти для тренировачных данных----------
#----------------------(картинка 28*28 пикселов = 784)--------------------
x = tf.compat.v1.placeholder(tf.float32, shape=(None,784))
#--------------а также исходной разметки картинок в one-hot формате-------
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None,10))
#-------------------------------------------------------------------------

#----------------------------Переменные, для слооёв нейросети--------
#-------------------------------------------------------------------------

#--------------------Веса и сдвиг первого слоя--------------------------------

W1 = tf.Variable(tf.compat.v1.truncated_normal([784,100], stddev=0.1))
b1 = tf.Variable(tf.compat.v1.truncated_normal([100],stddev=0.1))

#---------------------Веса и сдвиг второго слоя------------------------------
W2 = tf.Variable(tf.compat.v1.truncated_normal([100, 100], stddev=0.1))
b2 = tf.Variable(tf.compat.v1.truncated_normal([100], stddev=0.1))

#---------------------Веса и сдвиг третьего слоя------------------------------
W3 = tf.Variable(tf.zeros([100, 10]))
b3 = tf.Variable(tf.zeros([10]))

#--------------------Параметры перенормировочного слоя----------------------
beta = tf.Variable(tf.zeros([100]))
scale = tf.Variable(tf.ones([100]))
#-------------------------------------------------------------------------
beta1 = tf.Variable(tf.zeros([100]))
scale1 = tf.Variable(tf.ones([100]))
#-------------------------------------------------------------------------


#----------------------Массивы для проверочных данных---------------------
y_one_hot = np.zeros((batch_size,10),np.float32)
y_one_hot_test = np.zeros((n_test,10),np.float32)
test_indices = np.zeros(n_test,np.int32)
#-------------------------------------------------------------------------

#-----------Задаём слои нейросети с нормализацией по мини-батчам---------
h1 = tf.compat.v1.nn.tanh(tf.compat.v1.matmul(x, W1) + b1)
#------------Вычисляем среднее и дисперсию на выходе первого слоя---------
batch_mean, batch_var = tf.compat.v1.nn.moments(h1, [0])
#------------Нормировочный слой-------------------------------------------
h1_bn = tf.compat.v1.nn.batch_normalization(h1, batch_mean, batch_var, 
                                            beta, scale, 0.001)
#-------------------------------------------------------------------------
h2 = tf.compat.v1.nn.tanh(tf.compat.v1.matmul(h1_bn, W2) + b2)
#------------Вычисляем среднее и дисперсию на выходе второго слоя---------
batch_mean1, batch_var1 = tf.compat.v1.nn.moments(h2, [0])
#------------Нормировочный слой-------------------------------------------
h2_bn = tf.compat.v1.nn.batch_normalization(h2, batch_mean1, batch_var1, 
                                            beta1, scale1, 0.001)
#-------------------------------------------------------------------------
y_logit = tf.compat.v1.matmul(h2_bn, W3) + b3
#-------------------------------------------------------------------------

#---------------Функция активации третьего слоя нейросети------------------
y = tf.nn.softmax(tf.compat.v1.matmul(h2_bn, W3) + b3)
#-------------------------------------------------------------------------

#----------------------Функция потерь - перекрёстная энтропия-------------
#------------------------с усреднением по мини-батчу---------------------- 
cross_entropy = tf.reduce_mean(
    tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(y_,y_logit))
#-------------------------------------------------------------------------

#---------------Выбор метода оптимизации и скорости обучения (0.5)----------- 
#train_type = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train_type = tf.compat.v1.train.AdamOptimizer()
#----------------------------------------------------------------------------

#-----------------Минимизация кросс-энтропии как шаг оптимизации модели------
train_step = train_type.minimize(cross_entropy)
#----------------------------------------------------------------------------

#------------Преобразование тензоров в массивы-------------------------------
vvv = tf.compat.v1.Session().run(ttt)
www = tf.compat.v1.Session().run(uuu)
#----------------------------------------------------------------------------

#------------------Проверка точности построенной модели----------------------
#------------------В обученную сеть подаём тестовый набор картинок----------- 
#------------------и сравниваем ответ с настоящей меткой---------------------
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#---------------------------------------------------------------------------


#------------------------Запускаем сессию Tensorflow - на этом этапе 
#-------------------------пойдёт работа с графом вычислений-------------------
with tf.compat.v1.Session() as sess:
#-----------------------------------------------------------------------------    
    
#------------------------Первая сессия - инициализация глобальных переменных--
    sess.run(tf.compat.v1.global_variables_initializer())

#---------------------------------------------------------------------------- 
#-----------------------Цикл по эпохам обучения------------------------------    
    for m in range(num_steps):

#-----------Выборка случайных индексов и построение мини-батчей-------------- 
#----------тренировачных данных: интенсивности нормируем на единицу----------
        indices = np.random.choice(n_samples, batch_size)
        x_batch = np.array([vvv[j] for j in indices]) /255.0
#-----------------Создаём проверочный вектор y в one-hot кодировке----------
        y_aux = y_train[indices]
        l = -1
        for i in y_aux:
            l = l +1
            for k in range(0,10):
                if k==i:
                    y_one_hot[l,k] = 1.0
                else:
                    y_one_hot[l,k] = 0.0
#-------------------------Запускаем сессию оптимизации весов----------------
        sess.run(train_step, feed_dict={x: x_batch, y_: y_one_hot})
#---------------------------------------------------------------------------

#----После обучения модели проводим тест точности распознавания-----------


#-------------------Подготовка массивов тестовых картинок-------------------
    x_batch_test = www / 255.0
    for s in range(0,n_test):
        test_indices[s] = s
#---------Создаём проверочный вектор y_test в one-hot кодировке-------------
    l = -1    
    y_aux_test = y_test[test_indices]
    for i in y_aux_test:
        l = l +1
        for k in range(0,10):
            if k==i:
                y_one_hot_test[l,k] = 1.0
            else:
                    y_one_hot_test[l,k] = 0.0
    
#-------------------Запускаем сессию проверки построеной модели -------------
#------------------- на тестовом наборе цыфр---------------------------------
    print("Accuracy of identification: %s" % 
          sess.run(accuracy, feed_dict={x: x_batch_test, y_: y_one_hot_test}))

#----------------------------------------------------------------------------


