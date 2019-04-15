import numpy as np
import os
import scipy.io as so
def Inner_Loop(A,B,Q,R):
	p=A.shape[1]
	P=np.zeros((p,p))
	for i in range(1000000):
		P_new=Q+np.dot(np.dot(A.T,P),A)-np.dot(np.dot(np.dot(np.dot(A.T,P),B),np.linalg.inv(R+np.dot(np.dot(B.T,P),B))),np.dot(np.dot(B.T,P),A))
		if np.sum(np.power(P_new-P,2))<1e-10:
			break
		P=P_new
	K=-np.dot(np.linalg.inv(R+np.dot(np.dot(B.T,P),B)),np.dot(np.dot(B.T,P),A))
	return (P,K)
def Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,type_pi='exact',type_init='zero',modified_m=None):
	p_v=B.shape[1]
	p_w=C.shape[1]
	p_x=A.shape[1]
	if type_init=='zero':
		L_init=np.zeros((p_v,p_x))
		P_init=np.zeros((p_x,p_x))
	elif type_init=='random':
		L_init=np.random.randn(p_v,p_x)/10
		P_init=np.random.randn(p_x,p_x)
		P_init=np.dot(P_init.T,P_init)
	P_t=P_init*1
	L_t=L_init*1
	converge=False
	diverge=False
	counter=0
	while not (converge or diverge):
		inner_P,K_t=Inner_Loop(A+np.dot(B,L_t),C,Q_K+np.dot(np.dot(L_t.T,R_KL),L_t),R_KK)
		if type_pi=='modified':
			if modified_m is None:
				modified_m=1
			P_tnow=P_t*1
			for i in range(modified_m):
				P_t1=np.dot(np.dot((A+np.dot(B,L_t)+np.dot(C,K_t)).T,P_tnow),(A+np.dot(B,L_t)+np.dot(C,K_t)))
				P_t1=P_t1+Q_L+np.dot(np.dot(L_t.T,R_LL),L_t)+np.dot(np.dot(K_t.T,R_LK),K_t)
				P_tnow=P_t1*1
			L_t1=np.dot(np.dot(np.dot(np.linalg.inv(R_LL+np.dot(np.dot(B.T,P_t1),B)),B.T),P_t1),(A-np.dot(C,K_t)))
		elif type_pi=='exact':
			P_tnow=P_t*1
			counter_pi=0
			while True:
				P_t1=np.dot(np.dot((A+np.dot(B,L_t)+np.dot(C,K_t)).T,P_tnow),(A+np.dot(B,L_t)+np.dot(C,K_t)))
				P_t1=P_t1+Q_L+np.dot(np.dot(L_t.T,R_LL),L_t)+np.dot(np.dot(K_t.T,R_LK),K_t)
				counter_pi+=1
				if np.sum(np.power(P_tnow-P_t1,2))<1e-5:
					print(counter_pi)
					break
				P_tnow=P_t1*1
			L_t1=np.dot(np.dot(np.dot(np.linalg.inv(R_LL+np.dot(np.dot(B.T,P_t1),B)),B.T),P_t1),(A-np.dot(C,K_t)))
			print('exact square norm of delta p')
			print(np.sum(np.power(P_t1-P_t,2)))
			print('exact eigenvalue of delta p')
			print(np.linalg.eig(P_t1-P_t)[0])
			print('exact leader policy')
			print(L_t1)
		elif type_pi=='not_pi':
			P_t1,L_t1=Inner_Loop(A+np.dot(C,K_t),B,Q_L+np.dot(np.dot(K_t.T,R_LK),K_t),R_LL)
		counter+=1
		if (np.sum(np.power(P_t1-P_t,2)))<1e-20:
			converge=True
			print('converge')
			print('type:'+type_pi)
			print(counter)
		elif (np.sum(np.power(P_t1-P_t,2)))>10000:
			if type_pi=='exact':
				if (np.sum(np.power(P_t1-P_t,2)))>10000000:
					print('diverge')
					print('type:'+type_pi)
					diverge=True
					print(counter)
			elif type_pi=='modified':
				if (np.sum(np.power(P_t1-P_t,2)))>10000000:
					print('diverge')
					print('type:'+type_pi)
					diverge=True
					print(counter)
			else:
				print('diverge')
				print('type:'+type_pi)
				diverge=True
				print(counter)
		P_t=P_t1
		L_t=L_t1


def Exp(p_x=2,p_v=2,p_w=2,type_pi='exact',type_matrix='random',matrix_file=None,modified_m=None):
	if type_matrix=='random':
		if os.path.exists('counter.txt')==True:
			counter_file=open('counter.txt','r')
			nb=[]
			for line in counter_file:
				nb.append(line)
			counter=int(nb[0])+1
		else:
			counter=0
		A=np.random.randn(p_x,p_x)/5
		B=np.random.randn(p_x,p_v)
		C=np.random.randn(p_x,p_w)
		R_LL=np.random.randn(p_v,p_v)
		R_LL=np.dot(R_LL,R_LL.T)
		R_LK=np.random.randn(p_w,p_w)
		R_LK=np.dot(R_LK,R_LK.T)
		R_KL=np.random.randn(p_v,p_v)
		R_KL=np.dot(R_KL,R_KL.T)
		R_KK=np.random.randn(p_w,p_w)
		R_KK=np.dot(R_KK,R_KK.T)
		Q_L=np.random.randn(p_x,p_x)
		Q_L=np.dot(Q_L,Q_L.T)
		Q_K=np.random.randn(p_x,p_x)
		Q_K=np.dot(Q_K,Q_K.T)
		matrix_list=[A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L]
		name_list=['A','B','C','R_KL','R_LL','R_KK','R_LK','Q_K','Q_L']
		matdict=dict(zip(name_list,matrix_list))
		so.savemat('random_matrix_'+str(counter),matdict)
		f_counter=open('counter.txt','w')
		f_counter.write(str(counter))
		f_counter.close()
	if type_matrix=='from_file':
		matdict=so.loadmat(matrix_file)
		A=matdict['A']
		B=matdict['B']
		C=matdict['C']
		R_KL=matdict['R_KL']
		R_KK=matdict['R_KK']
		R_LL=matdict['R_LL']
		R_LK=matdict['R_LK']
		Q_L=matdict['Q_L']
		Q_K=matdict['Q_K']
	print('eigenvalue of A')
	print(np.linalg.eig(A)[0])
	if type_pi=='all':
		Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,type_pi='not_pi',modified_m=None)
		Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,type_pi='modified',modified_m=1)
		Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,type_pi='exact',modified_m=None)
	else:
		Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,type_pi=type_pi,modified_m=modified_m)
if __name__=='__main__':
	Exp(p_x=8,p_v=17,p_w=2,type_pi='all')
	#Exp(p_x=8,p_v=17,p_w=2,type_matrix='from_file',matrix_file='random_matrix_diverge.mat')
	#exp(type_pi='modified')
