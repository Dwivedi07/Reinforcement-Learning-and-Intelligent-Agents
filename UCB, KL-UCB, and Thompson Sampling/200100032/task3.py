"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

from random import randint
import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        # START EDITING HERE
        self.start = 0
        self.epsilon = 0.01
        self.reward_uptill = []
        for i in range(num_arms):
            self.reward_uptill.append([])

        self.counts = np.zeros(num_arms)
        self.indexes = [i for i in range(num_arms)]
        self.values = np.zeros(num_arms)
        self.pulled_arm_indices = []
        # You can add any other variables you need here
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.start == 0:
            self.start = 1
            self.pulled_arm_indices.append(np.random.choice(self.indexes))
            return self.pulled_arm_indices[-1]
        else:
            if self.values[self.pulled_arm_indices[-1]] >= (1-self.epsilon)*(1-1/self.num_arms):
                return self.pulled_arm_indices[-1]
            else:
                self.indexes.remove(self.pulled_arm_indices[-1])
                self.pulled_arm_indices.append(np.random.choice(self.indexes))
                return self.pulled_arm_indices[-1]
            

        
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.counts[arm_index] += 1
        self.reward_uptill[arm_index].append(reward)
        n = self.counts[arm_index]
        value = self.values[arm_index]

        new_value = ((n - 1) / n) * value + (1 / n) * reward


        self.values[arm_index] = new_value

        # END EDITING HERE
