from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse
import numpy as np

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.exit_x = 350
        self.exit_y = 0
        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        car_x,car_y,car_velocity,car_angle = state[0],state[1],state[2],state[3]
        r = (0-(car_y)/(self.exit_x-car_x))
        req_head_angle = (180/math.pi)*math.atan(r)
        if car_angle>=180:
            car_angle = 360-car_angle

        angle_to_be_rotated = req_head_angle-car_angle
        

        if abs(abs(car_angle)-abs(req_head_angle)) >=3:
            if angle_to_be_rotated >0:
                action_steer = 2
                action_acc = 2
            else:
                action_steer = 0
                action_acc = 2
        else:
            action_steer = 1
            action_acc = 4

        # action_steer = 2
        # action_acc = 2
        
        action = np.array([action_steer, action_acc])
        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.i = 1
        self.threshold = 2
        self.exit_x = 355
        self.exit_y = 0

        super().__init__()

    def Set_params(self,list):
        l = len(self.rx)
        self.delta = round(0.2*l)

        self.y = [list[0][1]+50,list[0][1]-50,
            list[1][1]+50,list[1][1]-50,
            list[2][1]+50,list[2][1]-50,
            list[3][1]+50,list[3][1]-50]
    
        self.y = np.sort(self.y)
        self.y_thr_min = self.y[3]
        self.y_thr_max = self.y[4]
    
   
    def calc_index_target(self, car_x, car_y):
        if self.delta < self.threshold:
            return -1
        
        dis = []
        for m in range(len(self.rx)):
            dis.append(math.hypot(car_x-self.rx[m], car_y-self.ry[m]))

        min_index = np.argsort(dis)[0] 
        if min_index+4>=len(self.rx):
            return -1

        return min_index+4
    
    def check_clear_path(self,car_x,car_y):
        a = 0
        b = 0

        for i in range(len(self.rx)):
            if self.rx[i]>=car_x:
                b+=1
                if self.ry[i]>self.y_thr_min+15 and self.ry[i]< self.y_thr_max-15:
                    a = a+1

        return (a==b)
        


    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        car_x,car_y,car_velocity,car_angle = state[0],state[1],state[2],state[3]
        
        if self.check_clear_path(car_x,car_y)==False:
            self.i= self.calc_index_target(car_x,car_y)
            if self.rx[self.i] != car_x:
                r = ((self.ry[self.i] - car_y)/(self.rx[self.i] - car_x))
                
                req_head_angle = (180/math.pi)*math.atan(r)
                if car_angle>=180:
                    car_angle = -(360-car_angle)
                if req_head_angle>=180:
                    req_head_angle = -(360-req_head_angle)

                angle_to_be_rotated = (req_head_angle-car_angle)
                if angle_to_be_rotated>180:
                    angle_to_be_rotated=360-angle_to_be_rotated
                elif angle_to_be_rotated<-180:
                    angle_to_be_rotated=360+angle_to_be_rotated

                # print('req head',req_head_angle, 'to be rotated',angle_to_be_rotated,'car angle', car_angle)
                if abs(angle_to_be_rotated) >=3:
                    if angle_to_be_rotated > 0:
                        action_steer = 2
                        action_acc = 2
                    else:
                        action_steer = 0
                        action_acc = 2

                else:
                    action_steer = 1
                    if car_velocity>8:
                        action_acc = 0
                    else:
                        action_acc = 3

                action = np.array([action_steer, action_acc])  

                return action
            else:
                if self.ry[self.i]>car_y:
                    req_head_angle = 90
                else:
                    req_head_angle = -90
                
                
                if car_angle>=180:
                    car_angle = -(360-car_angle)
                if req_head_angle>=180:
                    req_head_angle = -(360-req_head_angle)

                angle_to_be_rotated = (req_head_angle-car_angle)
                if angle_to_be_rotated>180:
                    angle_to_be_rotated=360-angle_to_be_rotated
                elif angle_to_be_rotated<-180:
                    angle_to_be_rotated=360+angle_to_be_rotated

                # print('req head',req_head_angle, 'to be rotated',angle_to_be_rotated,'car angle', car_angle)
                if abs(angle_to_be_rotated) >=3:
                    if angle_to_be_rotated > 0:
                        action_steer = 2
                        action_acc = 2
                    else:
                        action_steer = 0
                        action_acc = 2

                else:
                    action_steer = 1
                    if car_velocity>8:
                        action_acc = 0
                    else:
                        action_acc = 3

                action = np.array([action_steer, action_acc])  

                return action
        
        
        else:
            
            r = (self.exit_y-(car_y)/(self.exit_x-car_x))
            req_head_angle = (180/math.pi)*math.atan(r)
            if car_angle>=180:
                car_angle = -(360-car_angle)

            if req_head_angle>=180:
                req_head_angle = -(360-req_head_angle)
            angle_to_be_rotated = req_head_angle-car_angle
        
            # print('req head',req_head_angle, 'to be rotated',angle_to_be_rotated,'car angle', car_angle)
            if abs(angle_to_be_rotated) >=3:
                if angle_to_be_rotated > 0:
                    action_steer = 2
                    action_acc = 2
                else:
                    action_steer = 0
                    action_acc = 2
            else:
                action_steer = 1
                action_acc = 4
            
            action = np.array([action_steer, action_acc])
            return action
            


    class PPF:

        def __init__(self, x_c_obs, y_c_obs, resolution, rb):

            self.resolution = resolution
            self.rb = rb
            self.min_x, self.min_y = 0, 0
            self.max_x, self.max_y = 0, 0
            self.obstacle_map = None
            self.span_in_x, self.span_in_y = 0, 0
            self.motion = self.search_vectors()
            self.calc_obstacle_map(x_c_obs, y_c_obs)

        def Node(self, x, y, cost, pi):
            node_info = [x,y,cost,pi]
            return node_info

        def calc_final_path(self, goal_node, Explored_node):
            rx, ry = [self.Gpos(goal_node[0], self.min_x)], [
                self.Gpos(goal_node[1], self.min_y)]
            parent_index = goal_node[3]
            while parent_index != -1:
                n = Explored_node[parent_index]
                rx.append(self.Gpos(n[0], self.min_x))
                ry.append(self.Gpos(n[1], self.min_y))
                parent_index = n[3]

            return rx, ry

        def calc_h_cost(self,n1, n2):
            d =  math.hypot(n1[0] - n2[0], n1[1] - n2[1])
            return d

        def Gpos(self, index, min_position):
            position = index * self.resolution + min_position
            return position

        def Indexing_of_xy(self, position, min_pos):
            return round((position - min_pos) / self.resolution)

        def GIn(self, node):
            return (node[1] - self.min_y) * self.span_in_x + (node[0] - self.min_x)

        def verify_node(self, node):
            px = self.Gpos(node[0], self.min_x)
            py = self.Gpos(node[1], self.min_y)
            if px < self.min_x:    
                return False
            elif py < self.min_y:
                return False
            elif px >= self.max_x:
                return False
            elif py >= self.max_y:
                return False
            if self.obstacle_map[node[0]][node[1]]:       #obstacke zone check
                return False

            return True

        def calc_obstacle_map(self, x_c_obs, y_c_obs):

            self.min_x = round(min(x_c_obs))
            self.min_y = round(min(y_c_obs))
            self.max_x = round(max(x_c_obs))
            self.max_y = round(max(y_c_obs))
            self.span_in_x = round((self.max_x - self.min_x) / self.resolution)
            self.span_in_y = round((self.max_y - self.min_y) / self.resolution)
           
            self.obstacle_map = [[False for _ in range(self.span_in_y)]
                                for _ in range(self.span_in_x)]
            for ix in range(self.span_in_x):
                x = self.Gpos(ix, self.min_x)
                for iy in range(self.span_in_y):
                    y = self.Gpos(iy, self.min_y)
                    for iox, ioy in zip(x_c_obs, y_c_obs):
                        d = math.hypot(iox - x, ioy - y)
                        if d <= self.rb:
                            self.obstacle_map[ix][iy] = True
                            break

        def search_vectors(self):
            self.possible_search = [[1, 0, 1],
                                [0, 1, 1],
                                [-1, 0, 1],
                                [0, -1, 1],
                                [-1, -1, math.sqrt(2)],
                                [-1, 1, math.sqrt(2)],
                                [1, -1, math.sqrt(2)],
                                [1, 1, math.sqrt(2)]]

            return self.possible_search

        def planning(self, sx, sy, gx, gy):

            start_node = self.Node(self.Indexing_of_xy(sx, self.min_x),
                                self.Indexing_of_xy(sy, self.min_y), 0.0, -1)
            goal_node = self.Node(self.Indexing_of_xy(gx, self.min_x),
                                self.Indexing_of_xy(gy, self.min_y), 0.0, -1)
            
            Exploring_node, Explored_node = dict(), dict()
            Exploring_node[self.GIn(start_node)] = start_node

            while True:
                c_id = min(Exploring_node,key=lambda o: Exploring_node[o][2] + self.calc_h_cost(goal_node,Exploring_node[o]))
                current = Exploring_node[c_id]

                # Checkcing if we reached goal
                if current[0] == goal_node[0] and current[1] == goal_node[1]:
                    goal_node[3] = current[3]
                    goal_node[2] = current[2]
                    break

                # Remove the item from the exproing set and adding it to the Expored set
                del Exploring_node[c_id]                
                Explored_node[c_id] = current

                
                for i, _ in enumerate(self.motion):
                    node = self.Node(current[0] + self.motion[i][0], current[1] + self.motion[i][1], current[2] + self.motion[i][2], c_id)
                    n_id = self.GIn(node)

                    if n_id in Explored_node:
                        continue
                    if not self.verify_node(node):
                        continue
                    if n_id not in Exploring_node:
                        Exploring_node[n_id] = node
                    else:
                        if Exploring_node[n_id][2] > node[2]:
                            Exploring_node[n_id] = node

            rx, ry = self.calc_final_path(goal_node, Explored_node)

            return rx, ry

    def Pathplanner(self,state,ran_cen_list):
        sx = state[0]  # [m]
        sy = state[1]  # [m]
        gx = 340 # [m]
        gy = 0.0  # [m]
        resolve = 10.0  # [m]
        reach_buffer = 12.0  # [m]

        # Wall points added in obstacle zone
        x_c_obs, y_c_obs = [], []
        for i in range(-350, 350):
            x_c_obs.append(i)
            y_c_obs.append(350.0)
        for i in range(-350, 350):
            x_c_obs.append(-350)
            y_c_obs.append(i)
        for i in range(-350, 350):
            x_c_obs.append(i)
            y_c_obs.append(-350.0)
        for i in range(-350, -250):
            x_c_obs.append(350)
            y_c_obs.append(i)
        for i in range(250, 350):
            x_c_obs.append(350)
            y_c_obs.append(i)

        #pits points added in obst zone
        for l in range(4):
            for i in range(ran_cen_list[l][0]-80, ran_cen_list[l][0]+80):
                y_c_obs.append(ran_cen_list[l][1]+80)
                x_c_obs.append(i)
                y_c_obs.append(ran_cen_list[l][1]-80)
                x_c_obs.append(i)
            
            for i in range(ran_cen_list[l][1]-80, ran_cen_list[l][1]+80):
                x_c_obs.append(ran_cen_list[l][0]+80)
                y_c_obs.append(i)
                x_c_obs.append(ran_cen_list[l][0]-80)
                y_c_obs.append(i)
        


        path_algo = self.PPF(x_c_obs, y_c_obs, resolve, reach_buffer)
        rx, ry = path_algo.planning(sx, sy, gx, gy)
        rx = rx[::-1]
        ry = ry[::-1]
        return rx, ry

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False
            x = [state[0],state[1]]
            self.rx, self.ry = self.Pathplanner(x,ran_cen_list)
            # self.rx, self.ry = Pathplanner(x,ran_cen_list)
            self.list_cen = [ran_cen_list[i][0] for i in range(4)]
            self.Set_params(ran_cen_list)

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    
    # for random_seed in random_seeds:
    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
