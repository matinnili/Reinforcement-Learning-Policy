#!/usr/bin/env python
# coding: utf-8

# # Welcome!
# Below, we will learn to implement and train a policy to play atari-pong, using only the pixels as input. We will use convolutional neural nets, multiprocessing, and pytorch to implement and train our policy. Let's get started!
# 
# (I strongly recommend you to try this notebook on the Udacity workspace first before running it locally on your desktop/laptop, as performance might suffer in different environments)

# In[1]:


# install package for displaying animation
#!pip install JSAnimation

# custom utilies for displaying animation, collecting rollouts and more
import pong_utils

get_ipython().run_line_magic('matplotlib', 'inline')

# check which device is being used. 
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils.device
print("using device: ",device)


# In[2]:


import numpy as np


# In[3]:


# render ai gym environment
#!pip install gym[atari]
import gym
import time

# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment
env = gym.make('PongDeterministic-v4')

print("List of available actions: ", env.unwrapped.get_action_meanings())

# we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
# the 'FIRE' part ensures that the game starts again after losing a life
# the actions are hard-coded in pong_utils.py


# # Preprocessing
# To speed up training, we can simplify the input by cropping the images and use every other pixel
# 
# 

# In[4]:


import matplotlib
import matplotlib.pyplot as plt

# show what a preprocessed image looks like
env.reset()
_, _, _, _ = env.step(0)
# get a frame after 20 steps
for _ in range(20):
    frame, _, _, _ = env.step(1)

print(frame.shape)
plt.subplot(1,2,1)
plt.imshow(frame)
plt.title('original image')

plt.subplot(1,2,2)
plt.title('preprocessed image')

f = pong_utils.preprocess_single(frame)
print(f.shape)
# 80 x 80 black and white image
plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
plt.show()


# # Policy
# 
# ## Exercise 1: Implement your policy
#  
# Here, we define our policy. The input is the stack of two different frames (which captures the movement), and the output is a number $P_{\rm right}$, the probability of moving left. Note that $P_{\rm left}= 1-P_{\rm right}$

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F



# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        
    ########
    ## 
    ## Modify your neural network
    ##
    ########
        
        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride 
        # (round up if not an integer)

        # output = 20x20 here
        self.conv1 = nn.Conv2d(2, 4, kernel_size=4, stride=4)
        #size becomes 20*20*8
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=4)
        self.size=8*5*5
        
        # 2 fully connected layer
        self.fc1 = nn.Linear(self.size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        
    ########
    ## 
    ## Modify your neural network
    ##
    ########
    
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten the tensor
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sig(x)


# run your own policy!
#policy=Policy().to(device)
policy=Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
import torch.optim as optim
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


# # Game visualization
# pong_utils contain a play function given the environment and a policy. An optional preprocess function can be supplied. Here we define a function that plays a game and shows learning progress

# In[8]:


pong_utils.play(env, policy, time=100,preprocess=pong_utils.preprocess_single) 
# try to add the option "preprocess=pong_utils.preprocess_single"
# to see what the agent sees


# # Rollout
# Before we start the training, we need to collect samples. To make things efficient we use parallelized environments to collect multiple examples at once

# In[57]:


old_probs, states, actions, rewards =         pong_utils.collect_trajectories(envs, policy, tmax=100)


# # Function Definitions
# Here you will define key functions for training. 
# 
# ## Exercise 2: write your own function for training
# (this is the same as policy_loss except the negative sign)
# 
# ### REINFORCE
# you have two choices (usually it's useful to divide by the time since we've normalized our rewards and the time of each trajectory is fixed)
# 
# 1. $\frac{1}{T}\sum^T_t R_{t}^{\rm future}\log(\pi_{\theta'}(a_t|s_t))$
# 2. $\frac{1}{T}\sum^T_t R_{t}^{\rm future}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}$ where $\theta'=\theta$ and make sure that the no_grad is enabled when performing the division

# In[8]:


import numpy as np

def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):

    ########
    ## 
    ## WRITE YOUR OWN CODE HERE
    ##
    ########
    
    discounts=torch.tensor([discount**i for i in range(len(reward))]).view(100,1)
    rewards=torch.tensor(rewards)
    rewards=rewards*discounts
    rewards=(rewards-torch.mean(rewards,dim=0))/torch.std(rewards,dim=0)
    new_reward=torch.Tensor.cumsum(torch.flip(rewards,[0]),0)
    new_reward=torch.flip(new_reward,[0])
    actions=torch.tensor(actions)
    states=torch.cat(states)
    probs=policy(states).view(rewards[0],rewards[1])*(1-actions)+(1-policy(states).view(rewards[0],rewards[1]))*actions
    
    loss=torch.sum(torch.log(probs)*new_reward)
    
    
    seq_len = len(rewards)
    
    #calculate the future rewards
    discounts = discount**np.arange(seq_len)
    """
    the following codes is equal to:
        discount_rewards = np.asarray(rewards)
        for row in range(seq_len):
            discount_rewards[row, :] = discount_rewards[row, :]*discounts[n]
    
    it is used to calculate the point multiply by row elements 
    """
    '''discount_rewards = np.asarray(rewards)*discounts[:, np.newaxis]
    future_rewards = discount_rewards[::-1].cumsum(axis=0)[::-1]'''
    """batch normalization of future_rewards"""
    '''means = np.mean(future_rewards, axis=1)
    stds = np.std(future_rewards, axis=1)+ 1e-10
    normalized_future_rewards = (future_rewards - means[:, np.newaxis]) / stds[:, np.newaxis]
    
#     """
#     convert all to torch.tensor to calculate in gpu
#     """
#     #loss = torch.tensor(normalized_future_rewards*log_pro)    
#     actions = torch.tensor(actions, dtype=torch.int8, device=device)
#     normalized_future_rewards = torch.tensor(normalized_future_rewards, dtype=torch.float, device=device)
#     '''
#     the following old_probs is generated by collect_trajectories, it is a 
#     numpoy array detached from gpu which has no gradient info, so can not used to calculate the gradient 
#     '''
#     #old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
#     #log_pro = torch.tensor(log_pro, dtype=torch.float, device=device)
    
#     # convert states to policy (or probability)
#     new_probs = pong_utils.states_to_prob(policy, states)
#     new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

#     # include a regularization term
#     # this steers new_policy towards 0.5
#     # which prevents policy to become exactly 0 or 1
#     # this helps with exploration
#     # add in 1.e-10 to avoid log(0) which gives nan
#     entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
#         (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

#     '''
#     `normalized_future_rewards*torch.log(new_probs)` is useful to caculate the policy gradient, 
#     but seems not good as 
#     `normalized_future_rewards*(new_probs/old_probs)`, where old_probs is generated by collect_trajectories and it is not tensor in GPU
#     maybe it is caused by using torch.log(new_probs) to simulate the original prob (eg. old_probs), but it is not the same as original.     
#     '''
#     return torch.mean(normalized_future_rewards*torch.log(new_probs) + beta*entropy)
#     #return torch.mean(normalized_future_rewards*(new_probs/old_probs) + beta*entropy)
  
    
    return loss
    #torch.mean(normalized_future_rewards*torch.log(old_probs+1.0e-10))

#Lsur= surrogate(policy, prob, state, action, reward)

#print(Lsur)


# # Training
# We are now ready to train our policy!
# WARNING: make sure to turn on GPU, which also enables multicore processing. It may take up to 45 minutes even with GPU enabled, otherwise it will take much longer!

# In[11]:


from parallelEnv import parallelEnv
import numpy as np
# WARNING: running through all 800 episodes will take 30-45 minutes

# training loop max iterations
# episode = 500
episode = 800


# widget bar to display progress
get_ipython().system('pip install progressbar')
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
beta = .01
tmax = 320

# episode = 100
# tmax = 10

# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories
    old_probs, states, actions, rewards =         pong_utils.collect_trajectories2(envs, policy, tmax=tmax)
        
    total_rewards = np.sum(rewards, axis=0)

    # this is the SOLUTION!
    # use your own surrogate function
#     L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    
    L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    del L
        
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta*=.995
    
    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    
    # display some progress every 20 iterations
    if (e+1)%20 ==0 :
        print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
        print(total_rewards)
        
    # update progress widget bar
    timer.update(e+1)
    
timer.finish()
    


# In[ ]:


# play game after training!
pong_utils.play(env, policy, time=2000) 


# In[ ]:


# save your policy!
torch.save(policy, 'REINFORCE.policy')

# load your policy if needed
# policy = torch.load('REINFORCE.policy')

# try and test out the solution!
# policy = torch.load('PPO_solution.policy')


# In[ ]:




