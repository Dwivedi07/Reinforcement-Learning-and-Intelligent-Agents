"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.   
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"

        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.eps = 0.1
        self.success_counts = np.zeros(num_arms)
        self.failure_counts = np.zeros(num_arms)
        self.betadr_values = np.zeros(num_arms)
        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        arm_index = []
        number_of_pulls = np.zeros(self.num_arms)
        a = []
        pull_so_left = self.batch_size

        if np.random.random() < self.eps:

            for i in range(self.num_arms):

                if pull_so_left == 0:
                    break
        
                else:
                    if i == self.num_arms-1:
                        number_of_pulls[i] = pull_so_left
                        pull_so_left = self.batch_size - sum(number_of_pulls)
                        arm_index.append(i)
                        a.append(number_of_pulls[i])
                        

                    else:
                        number_of_pulls[i] = np.random.randint(0,pull_so_left+1)
                        pull_so_left = self.batch_size - sum(number_of_pulls)

                        
                        if number_of_pulls[i] != 0:
                            arm_index.append(i)
                            a.append(number_of_pulls[i])

            
            return (arm_index, np.array(a, dtype=int))

        else:


            # sorted_values_asc = np.sort(self.values)
            # # arm_index_aq_values = np.argsort(self.values)

            # # a_ind = [arm_index_aq_values[i] for i in [-1,-2,-3]]
            # req_values = [iv for iv in self.values if iv >= np.mean(sorted_values_asc)]
            # req_indices =[]
            # z = list(self.values)       

            # for i in req_values:
            #     req_indices.append(z.index(i))
            
            # l = len(req_indices)
            # # print(l)
            # number_of_pulls = np.zeros(l)
            # pull_so_left = self.batch_size

            
            # arm_index = []
            # a = []

            # for i in range(l):

            #     if pull_so_left == 0:
            #         break
        
            #     else:
            #         if i == l-1:
            #             number_of_pulls[i] = pull_so_left
            #             pull_so_left = self.batch_size - sum(number_of_pulls)
            #             arm_index.append(req_indices[i])
            #             a.append(number_of_pulls[i])
        
            #         else:
            #             number_of_pulls[i] = np.random.randint(0,pull_so_left+1)
            #             pull_so_left = self.batch_size - sum(number_of_pulls)

            #             if number_of_pulls[i] != 0:
            #                 arm_index.append(req_indices[i])
            #                 a.append(number_of_pulls[i])

            # return (arm_index, np.array(a, dtype=int))

            ############Thompson Sampling ###########
            
            # for i in range(self.num_arms):
            #     self.betadr_values[i] = np.random.beta(self.success_counts[i]+1, self.failure_counts[i]+1)

            # return ([np.argmax(self.betadr_values)], [self.batch_size])

        
            return ([np.argmax(self.values)], [self.batch_size])

        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        rewards = np.zeros(self.num_arms)
        arm_indexes = arm_rewards.keys()
        current_counts = np.zeros(self.num_arms)
        

        for i in arm_indexes:
            current_counts[i] = len(arm_rewards.get(i))
            self.counts[i] += current_counts[i]
            self.success_counts[i] += sum(arm_rewards.get(i))
            self.failure_counts[i] += current_counts[i] - sum(arm_rewards.get(i))
            rewards[i] = sum(arm_rewards.get(i))/current_counts[i]

        
        for i in arm_indexes:
            self.values[i] = ((self.counts[i] - current_counts[i]) / self.counts[i]) * self.values[i] + (current_counts[i] / self.counts[i]) * rewards[i]
        
        # END EDITING HERE