import random
import matplotlib.pyplot as plt

#Reward and next state functions

def Reward(state,action,terminal_state):
	if(action==1 and state==terminal_state-2):
		return 1;
	else:
		return 0;


def Next_state(state,action,terminal_state):
	if(action==0 or state==terminal_state-1):
		return state;
	else:
		return (state+1);

target = open("data_replacing.txt", "w")

#Initializing variables

A=2
S=5
gamma=0.9
lamda=0.9
alpha=[]
s_dash=0
a_dash=0
rms_error=[]

V=[]
policy=[]


for s in range(0,S):
	V.append(0)

for s in range(0,S):
	policy.append(random.randint(0.0,A-1))

#Code for calculation fo optimal V[s] (Similar to assignment 2)

policy_stable=0
while(policy_stable==0):
	delta=10.0
	while(delta>0.0000000000000001):
		delta=0.0
		for s in range(0,S):
			v=V[s]
			
			r=Reward(s,policy[s],S)
			s_dash=Next_state(s,policy[s],S)
			V[s]=r+gamma*V[s_dash]
			delta=max(delta,abs(v-V[s]))


	
	policy_stable=1
	for s in range(0,S):
		b=policy[s]
		max_sum=0.0 
		max_action=0
		for a in range(0,A):
			r=Reward(s,a,S)
			s_dash=Next_state(s,a,S)
			val=r+gamma*V[s_dash]
			if(val>max_sum):
				max_sum=val
				max_action=a

		
		policy[s]=max_action
		if(b!=policy[s]):
			policy_stable=0



#varying alpha from 0.05 to 0.95 in steps of 0.05

x=0.05
while(x<1.0):
	alpha.append(x)
	x=x+0.05



for k in range(0,19):
	target.write("alpha:")
	target.write(str(alpha[k]))
	target.write('\n')


	Q=[]
	e=[]


	for s in range(0,S):
		Q.append([])



	for s in range(0,S):
		e.append([])


	for s in range(0,S):
		for a in range(0,A):
			e[s].append(0.0);

	for s in range(0,S):
		for a in range(0,A):
			Q[s].append(0.0);

	count=0
	delta=0.0

#Code for calculation of Q[s] using SARSA(lambda)  with replacing traces

	while(count<5000):
		s=0
		a=random.randint(0,A-1)
		while(s!=S-1):
			r=Reward(s,a,S)
			s_dash=Next_state(s,a,S)
			epsilon=random.random()
			if(epsilon<0.2):
				a_dash=random.randint(0,A-1)
			else:
				max_a=0 
				count_a=0;
				for i in range(0,A):
					if ((Q[s_dash][i])>max_a):
						max_a=Q[s_dash][i]
						count_a=i

				a_dash=count_a
				
			delta=r+gamma*(Q[s_dash][a_dash])-Q[s][a]
			e[s][a]=1

			for i in range(0,S):
				for j in range(0,A):
					Q[i][j]=Q[i][j]+(alpha[k]*delta*e[i][j])
					e[i][j]=gamma*lamda*e[i][j]

			s=s_dash
			a=a_dash

		count=count+1;

	target.write("The Q values are: \n")	
	sum_tot=0.0
	for i in range(0,S):
		for j in range(0,A):
			target.write(str(Q[i][j]))

#RMS error calculation


	for i in range(0,S):

		sum_tot=sum_tot+(Q[i][policy[i]]-V[i])*(Q[i][policy[i]]-V[i])
	str1=str(sum_tot)
	target.write(" \n")
	target.write("RMS error: ")
	target.write(str1)
	target.write('\n')
	target.write('\n')
	rms_error.append(sum_tot)

#Plotting error vs alpha
plt.plot(alpha,rms_error)
plt.show()
		







