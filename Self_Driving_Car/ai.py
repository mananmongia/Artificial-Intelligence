# here we implement DQN class to select right action each time play

# import librareies
import numpy as np
import random
import os 
import torch # can handle dynamic graphs better than Tensorflow
import torch.nn as nn
import torch.nn.functional as F # contain different funtion to create Neural network (loss function we use will be uber loss)
import torch.optim as optim # to optimzer for stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import Variable # convertion form tensors to variable that containing gradient and tensor

# Creatting architecture of Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        #input_size is input neureons & nb_cation is action which can take palce here three
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size,30) # full connection B/W input layer - hidden - output layer
        self.fc2 = nn.Linear(30,nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# implement Experience replay
class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
    def push(self,event):
        # event has four elements (last State @ st, new state @ st+1, last action @ at,last reward @ rt)
        self.memory.append(event)
        if (len(self.memory) > self.capacity):
            del self.memory[0]
                
    def sample(self,batch_size):
            # to get random sample from memory /// to optimize we use pytorch 
            samples = zip(*random.sample(self.memory, batch_size)) 
            '''
            zip() is like reshape function for example
            list = ((1,2,3),(4,5,6)) =>(action,state,reward) =>((a1,s1,r1),(a2,s2,r2))
            zip(*list) = ((1,4),(2,5),(3,6)) => ((a1,a2),(s1,s2),(r1,r2))
            we need to do this because algo is wirtten like that
            we will rap these batches into pytorch variable which conatin both tensor and variable
            '''
            return map(lambda x: Variable(torch.cat(x, 0)),samples)
            '''
            lambda is function which takes samples as x and apply variable() function contatinate 
            them in 0 axis and then return
            '''
# implement Deep Q Learning 
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
     
        ''' last_state is last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, 
        orientation, -orientation] i.e. a vector, for pytorch we need more than a vector i.e.
        torchTensor with one dimension (fake dimension) that coresspond to the batch
        
        network can only accept observation batch
        
        '''
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # .sequeeze(index)
        self.last_action = 0 # action2rotation = [0,20,-20]
        self.last_reward = 0
        
    # select right action 
    def select_action(self, state):
        #softmax 
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        # Temperature = 7 /// more temperature, more chance to choose wining q value in softmax function
        # softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3) = [0, 0.02, 0.98]
        # volatile = True because we will not use gradient in Torch Tensor as input is converted to Torch Tensor in __init__ @ line 72
        action = probs.multinomial(num_samples = 1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # we don't consider the transitions by seris of tuple as sample() in replaymemory class creates batches
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # self.model(batch_state) gives output of all possible action , we want action decided by neural Network so we gather index 1
        # .unsqueeze(1) is to make diminsion similar to batch state as in line 73
        # .sequeeze(1) because we only need batch_action to be removed with fake dimension
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #we have several states so we detach /// max w.r.t actions so max(1) /// next state @ [0]
        target = self.gamma* next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() # reinitialize optimizer at each iteration of loop 
        td_loss.backward()
        # to free memory so retain_variable = True
        self.optimizer.step() # to update the weigths
        
    def update(self,reward, new_signal):
        # at line 131 map.py : action = brain.update(last_reward, last_signal)
                                # reward update : line 140 in mapy.py
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # state is a signal itself \\\ last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
       #now we need to select action to perform
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: #1st memory @ line 63 ||| 2nd memory of class ReplayMemory
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            #taking samples from memory via method in object (named memory of class ReplayMemoery) called sample(batch size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            #learn from samples
        self.last_action = action       #  --|
        self.last_state = new_state     #    | = updates variable
        self.last_reward = reward        # --| 
        #move reward window
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    },'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=>  CheckPoint . . . .")
            checkpoint = torch.load('last_brain.pth')
            # update self.model & self.optimizer
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done !")
        else:
            print("No CheckPoint Found")
                
        
            