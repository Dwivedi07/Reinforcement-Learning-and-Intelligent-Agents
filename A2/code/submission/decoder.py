import argparse, time
import numpy as np
import os
parser = argparse.ArgumentParser()

def Extract_states_info(file_path):
    f_s = open(file_path)
    lines=f_s.readlines()
    States = []
    for l in range(len(lines)):
        States.append(lines[l][:len(lines[l])-1])
    
    States.append(str(0).zfill(4))
    
    return States


def Extract_Value_Policy(path):
    fvp = open(path)
    lines = fvp.readlines()
    VP=[]
    for l in range(len(lines)):
        l = lines[l].split(" ")
        if int(l[1])==3:
            VP.append([4, float(l[0])])
        elif int(l[1])==4:
            VP.append([6, float(l[0])])
        else:
            VP.append([int(l[1]), float(l[0])])

    return VP

if __name__ == "__main__":
    parser.add_argument("--value-policy", type=str, required=True, help='Give path here for value and policy')
    parser.add_argument("--states",type=str, required=True, help = 'Give the states file path')
    args = parser.parse_args()

    States = Extract_states_info(args.states)
    Value_Policy = Extract_Value_Policy(args.value_policy)

    with open('policyfile.txt','w') as wf:
        for l in range(len(States)-1):
            print(f'{States[l]} {Value_Policy[l][0]} {Value_Policy[l][1]}')
            wf.write(f'{States[l]} {Value_Policy[l][0]} {Value_Policy[l][1]}\n')



