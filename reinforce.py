import gym
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras as k

class ReinforceAgent:

    def __init__(self,state_size,action_size):
        self.state_space=state_size
        self.action_size=action_size
        self.buffer=[]
        self.reward=[]
        self.discounted_reward=[]
        self.model= self.policy_network()

        self.model = k.Sequential()
        self.model.add(k.layers.Dense(32, activation=tf.keras.activations.relu, input_dim=self.state_space))
        self.model.add(k.layers.Dense(32, activation=tf.keras.activations.relu))
        self.model.add(k.layers.Dense(self.action_size, activation= tf.keras.activations.softmax))
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam')

        self.model.summary()

    def take_action(self,state):
        action = np.random.choice([0,1], p=self.model.predict(state).flatten())
        return action


    def save_experience(self,obs,action,reward):
        self.buffer.append([obs,action,reward])
        self.reward.append(reward)

    def discount_rewards(self):
        discounted_reward=[]
        for i in reversed(self.reward):
            temp=temp*self.discount_factor+i
            discounted_reward.insert(0,temp)
        return discounted_reward

    def train_agent(self):
        rewards=discounted_reward()


        self.model.fit(x,y)




if __name__=='__main__':
    # env=gym.make('HalfCheetah-v2')
    # print(env.action_space)
    agent = ReinforceAgent
    env1=gym.make('CartPole-v1')
    print(env1.action_space)
    observation=env1.reset()
    for i in range (100000):

        for i in range(1000):
            env1.render()
            action = agent.take_action(observation)
            observation_, reward, done, _ = env1.step(action)
            agent.save_experience(observation,action,reward)
            observation=observation_

            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                break
        discountedsreward=agent.discounted_reward

        env1.close()










