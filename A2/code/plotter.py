import numpy as np
import random,argparse,sys,subprocess,os,time
parser = argparse.ArgumentParser()
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser.add_argument("--textrp", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--textop", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--vary-runsrp", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--vary-runsop", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--vary-ballsrp", type=str, required=False, help='Give path here for states file')
    parser.add_argument("--vary-ballsop", type=str, required=False, help='Give path here for states file')
    args = parser.parse_args()

    if args.textrp != None and args.textop != None:
        rfrp = open(args.textrp,'r')
        rfop = open(args.textop,'r')
        lines = rfrp.readlines()
        lines1 = rfop.readlines()
        Proabab_win = []
        Probab_win_o =[]
        q = []
        for i in range(len(lines)):
            info = lines[i].split(' ')
            info1 = lines1[i].split(' ')
            Proabab_win.append(float(info[1]))
            Probab_win_o.append(float(info1[1]))
            q.append(float(info[0]))
        print(q, Proabab_win)       
        plt.plot(q,Proabab_win)
        plt.plot(q,Probab_win_o)
        plt.xlabel('q(Weakness of Player B')
        plt.ylabel('Winning Probability')
        plt.legend(["Random Policy", "Optimal Policy"], loc ="upper right")
        plt.show()
        rfrp.close()
        rfop.close()

    if args.vary_runsrp != None and args.vary_runsop != None:
        print('Generating Plots for constant Balls=10 and Runs changing from 20 to 1')
        rfrp = open(args.vary_runsrp,'r')
        rfop = open(args.vary_runsop,'r')
        lines = rfrp.readlines()
        lines1 = rfop.readlines()
        MAx_Runs = int(lines[0].split(' ')[0][2:])
        States = []
        Proabab_win = []
        Proabab_win_o = []
        runs_left = [20-i for i in range(20)]
        for i in range(MAx_Runs):
            info = lines[i].split(' ')
            info1 = lines1[i].split(' ')
            Proabab_win.append(float(info[2]))
            Proabab_win_o.append(float(info1[2]))

        print(runs_left, Proabab_win)
        plt.plot(runs_left, Proabab_win)
        plt.plot(runs_left, Proabab_win_o)
        plt.xlabel('Runs to Score')
        plt.ylabel('Winning Probability')
        plt.legend(["Random Policy", "Optimal Policy"], loc ="upper right")
        plt.show()
        rfrp.close()
        rfop.close()


    if args.vary_ballsrp != None:
        print('Generating Plots for constant Runs=10 and Balls changing from 15 to 1')
        rfrp = open(args.vary_ballsrp,'r')
        rfop = open(args.vary_ballsop,'r')
        lines = rfrp.readlines()
        lines1 = rfop.readlines()
        States = []
        Proabab_win = []
        Probab_win_o = []
        balls = [15-i for i in range(15)]
        for i in range(15):
            info = lines[10*i+1].split(' ')
            Proabab_win.append(float(info[2]))
            info1 = lines1[10*i+1].split(' ')
            Probab_win_o.append(float(info1[2]))

        print(balls, Proabab_win)
        plt.plot(balls,Proabab_win)
        plt.plot(balls,Probab_win_o)
        plt.xlabel('Balls Left')
        plt.ylabel('Winning Probability')
        plt.legend(["Random Policy", "Optimal Policy"], loc ="upper right")
        plt.show()
        rfrp.close()
        rfop.close()

    