"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import pwd
import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE

def f(q, p, t, c, u_a):
    
    if p == 1 :
        return p*math.log(p/q) - (math.log(t) + c* math.log(math.log(t)))/u_a

    elif p == 0 :
        return (1-p)*math.log((1-p)/(1-q)) - (math.log(t) + c* math.log(math.log(t)))/u_a

    else:
        
        return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q)) - (math.log(t) + c* math.log(math.log(t)))/u_a

def Bis_me(tol, p, l_lim, q, c, t, u_a):


    if p+q == 2:
        return 0

    else:
        r_lim = q
        while(np.abs(f((l_lim+r_lim)/2, p, t, c, u_a)) > tol):
            if (f((l_lim+r_lim)/2, p, t, c, u_a) * f(l_lim, p, t, c, u_a) < 0):
                r_lim = (l_lim+r_lim)/2
            else:
                l_lim = (l_lim+r_lim)/2

    return (l_lim+r_lim)/2
                



# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.n = num_arms
        self.check = 0
        self.index_pull = -1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)
    
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE

        if self.check == 0:
            if sum(self.counts) == self.num_arms:
                self.check = 1
                return np.argmax(self.ucb_values)
            else:
                self.index_pull +=1 
                return self.index_pull
        
        else:
            return np.argmax(self.ucb_values)
    
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        u_a = self.counts[arm_index]
        value = self.values[arm_index]
        
        

        if self.check == 1:
            for i in range(self.n):
                self.ucb_values[i] = self.values[i] + math.sqrt(2*math.log(sum(self.counts))/self.counts[i])

        
        new_value = ((u_a - 1) / u_a) * value + (1 / u_a) * reward
        self.values[arm_index] = new_value
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.n = num_arms
        self.check = 0
        self.index_pull = -1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.kl_ucb_values = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        if self.check == 0:
            if sum(self.counts) == self.num_arms:
                
                self.check = 1
                # print(self.values)
                for i in range(self.n):    
                    self.kl_ucb_values[i] = Bis_me(self.tol, self.values[i], self.values[i], 1, self.t, self.c, self.counts[i])

                return np.argmax(self.kl_ucb_values)
            else:
                self.index_pull +=1 
                return self.index_pull
        
        else:
            print(self.counts)
            # print(self.values)
            for i in range(self.n):    
                self.kl_ucb_values[i] = Bis_me(self.tol, self.values[i], self.values[i], 1, self.t, self.c, self.counts[i])
            return np.argmax(self.kl_ucb_values)
    
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):

        # START EDITING HERE
        self.counts[arm_index] += 1
        u_a = self.counts[arm_index]
        self.t = sum(self.counts)
        self.c = 3
        self.tol = 1e-10
        value = self.values[arm_index]
        new_value = ((u_a - 1) / u_a) * value + (1 / u_a) * reward
        self.values[arm_index] = new_value
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.n = num_arms
        self.success_counts = np.zeros(num_arms)
        self.failure_counts = np.zeros(num_arms)
        self.betadr_values = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(self.n):
            self.betadr_values[i] = np.random.beta(self.success_counts[i]+1, self.failure_counts[i]+1)
        return np.argmax(self.betadr_values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.success_counts[arm_index] += 1
        else:
            self.failure_counts[arm_index] += 1
        # END EDITING HERE
