import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--balls",required=False,type=str,help="Number of balls", default=15)
parser.add_argument("--runs",required=False,type=str,help="Runs to score", default=30)
args = parser.parse_args()
total_balls = int(args.balls)
score = int(args.runs)

with open(f'/host/A2/code/data/cricket/state_list{total_balls}{score}.txt','w') as wf:
    for balls in range(total_balls):
        for runs in range(score):
            for player in range(1):
                mu = str(total_balls-balls).zfill(2)+  str(score-runs).zfill(2)
                wf.write(f'{mu}\n')
