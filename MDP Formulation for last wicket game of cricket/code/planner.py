import argparse, time
import numpy as np
import os
from pulp import *
parser = argparse.ArgumentParser()

def ExtractData(file_path):
    f = open(file_path)
    lines=f.readlines()
    NumStates = int(lines[0].split(" ")[-1])
    NumActions = int(lines[1].split(" ")[-1])
    mdptype = lines[-2].split(" ")[-1]
    Gamma = float(lines[-1].split(" ")[-1])
    ends = lines[2].split(" ")
    Terminal_states = [int(i) for i in ends[1:]]

    # action *state1 *state 2 == dimensions of array
    Trans_probab = np.zeros((NumActions, NumStates, NumStates))
    Reward = np.zeros((NumActions, NumStates, NumStates))
    for i in range(3,len(lines)-2):
        info = lines[i].split(" ")
        s1,a,s2,r,p = int(info[1]), int(info[2]), int(info[3]), float(info[4]), float(info[5])
        Trans_probab[a,s1,s2]= p
        Reward[a,s1,s2] = r

    return Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma


def Bellman_op_op(F, Terminal_states, NumStates, NumActions , Trans_probab, Reward, mdptype, Gamma):
    Value_function = np.zeros(NumStates)
    Policy = np.zeros(NumStates)
    for s1 in range(NumStates):        
        Int_Val = 0
        for a in range(NumActions):
            Int_Val_new = 0
            for s2 in range(NumStates):
                Int_Val_new += Trans_probab[a,s1,s2]*(Reward[a,s1,s2] + Gamma*F[s2]) 
            if Int_Val_new > Int_Val:
                Int_Val = Int_Val_new
                Value_function[s1] = Int_Val_new
                Policy[s1] = a
        
    return Value_function, Policy

def Infi_norm(F1, F2):
    norm = 0    
    #Infi Norm
    diff  = F1 - F2
    for i in range(len(diff)):
        diff[i] = abs(diff[i])

    norm = np.amax(diff)
    return norm

def Calculate_Policy(V, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma):
    policy = []
    for s1 in range(NumStates):
        err_a_pol = []
        for a in range(NumActions):
            lam = 0
            for s2 in range(NumStates):
                lam += Trans_probab[a,s1,s2]*(Reward[a,s1,s2] + Gamma*V[s2])

            err_a_pol.append(abs(lam - V[s1]))
        policy.append(np.argsort(err_a_pol)[0])
    return policy


def Linear_Solver(Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma):
    V1 = []
    prob=LpProblem("The Optimal Value Function", LpMinimize)
    StateValues = []

    for i in range(NumStates):
        if i<10:
            StateValues.append(f'0{i}')
        else:
            StateValues.append(i)

    # A dictionary called SVvars is created to contain the referenced Variables
    SVvars=LpVariable.dicts("V",StateValues)


    # The objective function is added to 'prob' first
    prob+= lpSum([SVvars[StateValues[sT]]  for sT in range(NumStates)])

    #Adding Constraints for Terminal States
    if (mdptype[0] == 'e'):
        for sT in Terminal_states:
            prob += SVvars[StateValues[sT]] == 0, f'Terminal state{sT} value function constraint'
            
    #Adding Constraints
    for s1 in range(NumStates):
        for a in range(NumActions):
            prob += SVvars[StateValues[s1]] >= lpSum((Trans_probab[a,s1,s2]*(Reward[a, s1, s2] + Gamma * SVvars[StateValues[s2]])) for s2 in range(NumStates)) 


    prob.solve(PULP_CBC_CMD(msg=False))


    for v in prob.variables():
    
        V1.append(v.varValue)

    return V1

def Generate_Random_Policy(NumStates, NumActions):
    Rd_pol = []
    for i in range(NumStates):  
        Rd_pol.append(np.random.randint(0,1)) 

    return Rd_pol

def Improvable_Actions(V_PI, Trans_probab, Reward, Gamma, NumStates, NumActions, Terminal_states):
    A = []
    for i in range(NumStates):
        A.append([])
    
    for s1 in range(NumStates):
        q_a = []
        ac = []
        for a in range(NumActions):
            Q_PI=0
            Q_PI = sum([Trans_probab[a,s1,s2]*(Reward[a,s1,s2] + Gamma*V_PI[s2]) for s2 in range(NumStates)])
            if Q_PI > V_PI[s1] + 1e-8:
                q_a.append(Q_PI)
                ac.append(a)
        if len(q_a) != 0:
            A[s1] = ac[np.argmax(q_a)]
    
    return A

def Improved_Policy(IA_S, Rd_pol):
    for s1 in range(len(IA_S)):
        if IA_S[s1] != []:
            Rd_pol[s1] = IA_S[s1]
    
    return Rd_pol

def check(IA_S):
    a = 0
    for i in range(len(IA_S)):
        if IA_S[i] == []:
            a+=1
    
    return (a != len(IA_S))
    
def check_V(A,B):
    k = 0
    for i in range(len(A)):
        if A[i] == B[i]:
            k = k+1
    
    return (k == len(A))


def Evaluate_VI(MDP):
    Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma  = ExtractData(MDP)
    
    #setting initial guess of Value function as V0 and the tolerance vealue for stopping condition
    V0=[]
    epsilon = 1e-10
    for i in range(NumStates):
        V0.append(1)

    #initial guess in case of episodic tasks
    if (mdptype[0] == 'e'):
        for i in Terminal_states:
            V0[i] = 0
        
    V1= Bellman_op_op(V0, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)[0]
  
    while (Infi_norm(V0, V1) > epsilon):
        V0 = V1
        V1, pi = Bellman_op_op(V0, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)
 
    return V1, pi

    
def Evaluate_HPI(MDP):
    Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma  = ExtractData(MDP)

    '''
    I: We will initialize an arbitrary policy
    II: For this policy we will calculate Q(S,A) for each possible action at each state and compare it with the value function of given policy
    III: This will help us in determining IA and IS
    IV: We will Randomly choose the next policy by picking up one if IA's in respective IS's
    V: We will repeat II,III,IV till IS == phi
    '''
    Rd_pol = Generate_Random_Policy(NumStates, NumActions)
    

    V_PI = Evaluate_VF(Rd_pol, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)
    IA_S = Improvable_Actions(V_PI, Trans_probab, Reward, Gamma, NumStates, NumActions, Terminal_states) 
 
    
    while check(IA_S) == 1:

        V_n = Improved_Policy(IA_S, Rd_pol)
        if check(IA_S) != 1:
            break
        
        Rd_pol = V_n
        V_PI = Evaluate_VF(Rd_pol, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)
        IA_S = Improvable_Actions(V_PI, Trans_probab, Reward, Gamma, NumStates, NumActions, Terminal_states)

    return V_PI, Rd_pol

def Evaluate_LP(MDP):
    Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma  = ExtractData(MDP)
    
    #We will first solve for the optimal value then actions by using those optimal values
    V1 = Linear_Solver(Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)
    pi = Calculate_Policy(V1, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)
    
    return V1, pi
    
def Evaluate_VF(Policy, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma):
    # Ax = b form for n variable n equations
    A = np.zeros((NumStates, NumStates))
    b = np.zeros(NumStates)
    for s1 in range(NumStates):
        b[s1] = -sum([Trans_probab[Policy[s1],s1,s2]*Reward[Policy[s1],s1,s2] for s2 in range(NumStates)])

    for s1 in range(NumStates):
        for s2 in range(NumStates):
            if s2 == s1:
                A[s1,s2] = Trans_probab[Policy[s1],s1,s2]*Gamma - 1
            else:
                A[s1,s2] = Trans_probab[Policy[s1],s1,s2]*Gamma

    return np.dot(np.linalg.inv(A), b) 

if __name__ == "__main__":
    parser.add_argument("--mdp", type=str, required=True, help='Give path here for the mdp')
    parser.add_argument("--algorithm",type=str, required=False, default = "hpi", help = 'Give the algo --accepetable inputs are vi, hpi, and lp')
    parser.add_argument("--policy", type=str, required=False, help='Give the file name for the policy to use for evaluation')
    args = parser.parse_args()

    if (args.policy != None):

        #as per example given in website page below should be taken, but i am commenting it as per the required format used in autograder
        # path_policy_file = os.path.join('/'.join(args.mdp.split('/')[:len(args.mdp.split('/'))-1]), args.policy)

        path_policy_file = args.policy
        f = open(path_policy_file)
        lines=f.readlines()

        Policy = []
        
        #Note the directory has been changed by TAs hence i am adding these lines for those types of policy file in which states and policy given
        if len(lines[0].split(" "))==2:
            for j in range(len(lines)):
                info = lines[j].split(" ")
                if int(info[1])== 4:
                    Policy.append(3)
                elif int(info[1])== 6:
                    Policy.append(4)
                else:
                    Policy.append(int(info[1]))
            #Apending the policy for the terminal state
            Policy.append(0)
            
        else:  #for normal episodic and continuing tasks
            for j in range(len(lines)):
                Policy.append(int(lines[j]))
        
        Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma  = ExtractData(args.mdp)
        V_PI = Evaluate_VF(Policy, Terminal_states, NumStates, NumActions ,Trans_probab, Reward, mdptype, Gamma)
        for i in range(len(V_PI)):
            print(f"{V_PI[i]} {Policy[i]}")
        with open('value_and_policy_file.txt','w') as wf:
            for i in range(len(V_PI)):
                wf.write(f"{V_PI[i]} {int(Policy[i])}\n")
        
    
    elif args.algorithm == 'vi':
        V_star, PI_star = Evaluate_VI(args.mdp)

        #Autograder want print
        for i in range(len(V_star)):
            print(f"{V_star[i]} {int(PI_star[i])}")


    elif args.algorithm == 'hpi':
        V_star, PI_star = Evaluate_HPI(args.mdp)
        for i in range(len(V_star)):
            print(f"{V_star[i]} {int(PI_star[i])}")

        #AS DEFAULT ALGO FOR SOLVINF SO CRICKET PROBLEM WILL NEED  VALUE AND POLICY FILE
        with open('value_and_policy_file.txt','w') as wf:
            for i in range(len(V_star)):
                wf.write(f"{V_star[i]} {int(PI_star[i])}\n")
    
    elif args.algorithm == 'lp':
        V_star, PI_star = Evaluate_LP(args.mdp)
        for i in range(len(V_star)):
            print(f"{V_star[i]} {PI_star[i]}")

            
        

