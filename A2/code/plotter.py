import numpy as np
import random,argparse,sys,subprocess,os,time
parser = argparse.ArgumentParser()
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser.add_argument("--text", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--vary-runs", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--vary-balls", type=str, required=False, help='Give path here for states file')
    args = parser.parse_args()

    if args.text != None:
        rf = open(args.text,'r')
        lines = rf.readlines()
        Proabab_win = []
        q = []
        for i in range(len(lines)):
            info = lines[i].split(' ')
            mu = np.zeros(len(lines))
            Proabab_win.append(float(info[1]))
            q.append(float(info[0]))

        plt.plot(q,Proabab_win)
        plt.xlabel('q(Weakness of Player B')
        plt.ylabel('Winning Probability')
        plt.show()
        rf.close()

    if args.vary_runs != None:
        print('Generating Plots for constant Balls=10 and Runs changing from 20 to 1')
        rf = open(args.vary_runs,'r')
        lines = rf.readlines()
        MAx_Runs = int(lines[0].split(' ')[0][2:])
        States = []
        Proabab_win = []
        runs_left = [20-i for i in range(20)]
        for i in range(MAx_Runs):
            info = lines[i].split(' ')
            Proabab_win.append(float(info[2]))

        # print(balls,Proabab_win)
        plt.plot(runs_left, Proabab_win)
        plt.xlabel('Runs to Score')
        plt.ylabel('Winning Probability')
        plt.show()
        rf.close()


    if args.vary_balls != None:
        print('Generating Plots for constant Runs=10 and Balls changing from 15 to 1')
        rf = open(args.vary_balls,'r')
        lines = rf.readlines()
        States = []
        Proabab_win = []
        balls = [15-i for i in range(15)]
        for i in range(15):
            info = lines[10*i+1].split(' ')
            Proabab_win.append(float(info[2]))

        # print(runs,Proabab_win)
        plt.plot(balls,Proabab_win)
        plt.xlabel('Balls Left')
        plt.ylabel('Winning Probability')
        plt.show()
        rf.close()

    