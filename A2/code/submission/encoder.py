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

def Parameters_extract(file_path):
    fp=open(file_path)
    lines=fp.readlines()
    Probab_a_r = np.zeros((5,8))   

    for l in range(1,len(lines)):
        linedata = lines[l].split(" ")
        for i in range(1,len(linedata)+1):
            if i<=6:
                Probab_a_r[l-1][i-1]=float(linedata[i])
            elif i == 7:                                          #Note no 5 runs possoble but here i added for my ease to use it in functions
                Probab_a_r[l-1][i-1]=0
            elif i == 8:
                Probab_a_r[l-1][i-1]=float(linedata[-1])

    return Probab_a_r

def Probab_A(bb1, bb2, rr1, rr2, a, Probab_ac_run):
    if bb1%6 == 1:                                              #If A at the end of over to be in strike should score either 1 or 3 
        if rr1-rr2 in [1,3]:
            return Probab_ac_run[a][rr1-rr2+1]
        else:
            return 0
    else:
        if (rr1-rr2<=6):                                          #A in mid of over so it cannot reach next state by scoring 1 or 3
            if (rr1-rr2 == 1 or rr1-rr2 == 3):
                return 0
            else:
                return Probab_ac_run[a][rr1-rr2+1]
        
        else:
            return 0

def P_ev(a,r,Probab_ac_run):
    if r<=6:
        return Probab_ac_run[a][r+1]
    else:
        return 

def over(num):
    if 0<=num<=6:
        return 0
    elif 7<=num<=12:
        return 1
    else:
        return 2

def Expec_probab(bb1, bb2, rr1, rr2, a, Probab_ac_run):
    P=0
    delb = bb1-1-bb2
    Runs = rr1 - rr2
    if bb1 == 13 or bb1 == 7:
        Aexp = [0,2,4,6]                   #A is at end of the over so B will get strike from the brgiining of over hence A can score only 0,2,4,6
        Bexp = [Runs-0,Runs-2,Runs-4,Runs-6]
        if delb <=5:
            if (1 in Bexp and Runs-1 in Aexp):                     #A again get the strike within that over
                P+=P_ev(a,Runs-1,Probab_ac_run)
        elif delb == 6 and bb1 != 7:            #A get the strike at the begining of next over basically maeden over from B
            if (0 in Bexp and Runs in Aexp):
                P+=P_ev(a,Runs,Probab_ac_run)
        elif delb >= 7 and bb1 != 7:            #A get strike in next over 
            if (2 in Bexp and Runs-2 in Aexp):
                P+=P_ev(a,Runs-2,Probab_ac_run)

    else:                                       #A at the mid of an over changes strike so he can score either 1 or 3
        Bexp = [Runs-1, Runs-3] 
        Aexp = [1,3]   
        over_i = over(bb1-1) 
        over_f = over(bb2+1)     
        #bb1-1//6 over number when B got strike and bb2+1//6 over number when B played last ball
        if Runs > 6:
            P = 0
        else:
            if (bb2 % 6 != 0):                    #if the A is not getting strike at the begining of the next over
                if (over_i - over_f == 2):          #Switch of 2 overs
                    if (3 in Bexp and Runs-3 in Aexp):
                        P += P_ev(a,Runs-3,Probab_ac_run)
                elif (over_i - over_f == 1):        #Switch of 1 over
                    if (2 in Bexp and Runs-2 in Aexp):
                        P+= P_ev(a,Runs-2, Probab_ac_run)
                elif (over_i - over_f == 0):        #Switch within the same over
                    if (1 in Bexp and Runs-1 in Aexp):
                        P+= P_ev(a,Runs-1,Probab_ac_run)
            
            else:                                  #A getting strike next time at the begining of the next over
                if (over_i - over_f == 1):         #Switch of 1 over
                    if (1 in Bexp and Runs-1 in Aexp):
                        P+= P_ev(a,Runs-1, Probab_ac_run)
                elif (over_i - over_f == 0):       #Switch within the same over
                    if (0 in Bexp and Runs in Aexp):
                        P+= P_ev(a,Runs,Probab_ac_run)
    
    return P

def End_Expec_Probab(bb1, bb2, rr1, rr2, a, Probab_ac_run, q):
    P=0
    #Directly score by A and win
    P+=EndProbab_A(rr1,rr2,a,Probab_ac_run)           #Adding the Probab of A direct hitting shot led to win
    Runs = rr1 - rr2
    #Invovling strike change to B
    if bb1==13 or bb1==7:                                #A at end of over so B got strike at starting
        Aexp = [r for r in [0,2,4,6] if Runs - r >0]
        Bexp = [Runs - rb for rb in Aexp]
        for run_sc_A in Aexp:
            if Runs-run_sc_A in Bexp:
                if Runs-run_sc_A == 1:
                    P+=P_ev(a,run_sc_A,Probab_ac_run)*(sum([((1-q)/2)**(i+1) for i in range(6)]))
                elif Runs-run_sc_A == 2 and bb1 ==13:
                    P+=P_ev(a,run_sc_A,Probab_ac_run)*(((1-q)/2)**6)*(sum([((1-q)/2)**(i+1) for i in range(6)]))
        
            else:
                P+=0
    else:
        Aexp = [mu for mu in [1,3] if Runs-mu>0]       #A can't score anything which will make target of B 0
        Bexp = [Runs-l  for l in Aexp]
        over_iB = (bb1-1)//6
        bbp = (bb1-1)%6
        for run_sc_A in Aexp:
            if Runs - run_sc_A in Bexp:
                if Runs - run_sc_A == 1:
                    P+=P_ev(a,run_sc_A,Probab_ac_run)*(sum([((1-q)/2)**(i+1) for i in range(bbp)]))
                
                elif Runs - run_sc_A == 2 and over_iB != 0:
                    P+=P_ev(a,run_sc_A,Probab_ac_run)*(((1-q)/2)**bbp)*(sum([((1-q)/2)**(i+1) for i in range(6)]))
                
                elif Runs - run_sc_A == 3 and over_iB == 2:
                    P+=P_ev(a,run_sc_A,Probab_ac_run)*(((1-q)/2)**(6+bbp))*(sum([((1-q)/2)**(i+1) for i in range(6)]))
                
            else:
                P+=0


    return P


def EndProbab_A(rr1, rr2, a, Probab_ac_run):
    P=0
    Runs = rr1-rr2
    for i in range(8):  
        if i-1>=Runs:
            P+=Probab_ac_run[a,i]

    return P


def Generate_Transition_probab(path1, path2, q):
    #Extracting DATA 
    States = Extract_states_info(path1)
    Probab_ac_run = Parameters_extract(path2) 
    NumActions = 5
    NumStates =len(States)

    Trans_probab = np.zeros((NumActions, NumStates, NumStates))
    Rewards = np.zeros((NumActions, NumStates, NumStates))

    for a in range(NumActions):
        for s1 in range(NumStates):
            for s2 in range(NumStates):
                bb1 = int(States[s1][:2])
                bb2 = int(States[s2][:2])
                rr1 = int(States[s1][2:])
                rr2 = int(States[s2][2:])
                if bb2!=0:                               #Next state in which A will land will not be winning state
                    if bb1-bb2 > 0 and rr1-rr2>=0:
                        if bb1-bb2 == 1:                 #consecutive states
                            Trans_probab[a][s1][s2] = Probab_A(bb1, bb2, rr1, rr2, a, Probab_ac_run)
                        else:
                            '''when A reach next state with B being involved in between
                               then B should play bb1-bb2-1 balls without being out
                            '''
                            Trans_probab[a][s1][s2] = Expec_probab(bb1, bb2, rr1, rr2, a, Probab_ac_run)*(((1-q)/2)**(bb1-bb2-1)) #error line
                    else:
                        Trans_probab[a][s1][s2] = 0
                else:
                    if bb1-bb2 > 0 and rr1-rr2>=0:
                        if bb1-bb2 == 1:
                            Trans_probab[a][s1][s2] = EndProbab_A(rr1, rr2, a, Probab_ac_run)
                        else:
                            Trans_probab[a][s1][s2] = End_Expec_Probab(bb1, bb2, rr1, rr2, a, Probab_ac_run, q) 
                    else:
                        Trans_probab[a][s1][s2] = 0

    for a in range(NumActions):
        for s1 in range(NumStates):
            for s2 in range(NumStates):
                if s2 == NumStates-1:
                    Rewards[a,s1,s2] = 1
    
    return States, NumStates, NumActions, Trans_probab, Rewards

if __name__ == "__main__":
    parser.add_argument("--states", type=str, required=True, help='Give path here for states file')
    parser.add_argument("--parameters",type=str, required=True, help = 'Give the player A parameters')
    parser.add_argument("--q", type=float, required=True, help='Give the weakness of B')
    args = parser.parse_args()

    q=args.q
    States, NumStates, NumActions, Trans_probab, Rewards = Generate_Transition_probab(args.states, args.parameters, q)
    Bmax = int(States[0][:2])
    Rmax = int(States[0][2:])

    with open('mdp_file.txt','w') as wf:
        wf.write(f'numStates {NumStates}\n')
        wf.write(f'numActions {NumActions}\n')
        wf.write(f'end {len(States)-1}\n')
        print(f'numStates {NumStates}')
        print(f'numActions {NumActions}')
        print(f'end {len(States)-1}')
        for a in range(NumActions):
            for s1 in range(NumStates):
                for s2 in range(NumStates):
                    if Trans_probab[a][s1][s2] != 0:
                        print(f'transition {s1} {a} {s2} {Rewards[a,s1,s2]} {Trans_probab[a,s1,s2]}')
                        wf.write(f'transition {s1} {a} {s2} {Rewards[a,s1,s2]} {Trans_probab[a,s1,s2]}\n')
        
        wf.write(f'mdptype episodic\n')
        wf.write(f'discount 1\n') 
        print(f'mdptype episodic')
        print(f'discount 1')

        
        