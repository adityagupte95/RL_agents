# import tensorflow as tf
# from tensorflow import keras as k
# import gym
#
#
# model = k.Sequential()
# model.add(k.layers.Dense(32, activation=tf.keras.activations.relu, input_dim=35))
# model.add(k.layers.Dense(32, activation=tf.keras.activations.relu))
# model.add(k.layers.Dense(10))
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam')
#
# # model.summary()
# buffer=[]
#
# obs=[12,13,14]
# action=[12,127,124,433]
# reward=[1]
#
# buffer.append([obs, action, reward])
# print(buffer)


temp=[1,2,3,4,5,6,7,8]
temp1=[]
p=0
for i in reversed(temp):
    p=p*0.9+i
    temp1.insert(0,p)
print(temp1)