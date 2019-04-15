import numpy as np
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
def Modified_Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,k_m=100):
	p_b=B.shape[1]
	p_c=C.shape[1]
	p_x=A.shape[1]
	L_init=np.zeros((p_b,p_x))
	P_init=np.zeros((p_c,p_x))
	P_t=P_init*1
	L_t=L_init*1
	for t in range(k_m):
		inner_P,K_t=Inner_Loop(A+np.dot(B,L_t),C,Q_K+np.dot(np.dot(L_t.T,R_KL),L_t),R_KK)
		P_t1=np.dot(np.dot((A+np.dot(B,L_t)+np.dot(C,K_t)).T,P_t),(A+np.dot(B,L_t)+np.dot(C,K_t)))
		P_t1=P_t1+Q_L+np.dot(np.dot(L_t.T,R_LL),L_t)+np.dot(np.dot(K_t.T,R_LK),K_t)
		L_t1=np.dot(np.dot(np.dot(np.linalg.inv(R_LL+np.dot(np.dot(B.T,P_t1),B)),B.T),P_t1),(A-np.dot(C,K_t)))
		print(np.sum(np.power(P_t1-P_t,2)))
		if (np.sum(np.power(P_t1-P_t,2)))<1e-20:
			break
		P_t=P_t1
		L_t=L_t1
def Exact_Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,k_m=100):
	p_b=B.shape[1]
	p_c=C.shape[1]
	p_x=A.shape[1]
	L_init=np.zeros((p_b,p_x))
	P_init=np.zeros((p_c,p_x))
	P_t=P_init*1
	L_t=L_init*1
	for t in range(k_m):
		inner_P,K_t=Inner_Loop(A+np.dot(B,L_t),C,Q_K+np.dot(np.dot(L_t.T,R_KL),L_t),R_KK)
		P_t1,L_t1=Inner_Loop(A+np.dot(C,K_t),B,Q_L+np.dot(np.dot(K_t.T,R_LK),K_t),R_LL)
		print(np.sum(np.power(P_t1-P_t,2)))
		if (np.sum(np.power(P_t1-P_t,2)))<1e-20:
			break
		P_t=P_t1
		L_t=L_t1
def exp(p_x=2,p_b=2,p_c=2,type_pi='exact'):
	A=np.random.randn(p_x,p_x)/5
	B=np.random.randn(p_x,p_b)
	C=np.random.randn(p_x,p_c)
	R_LL=np.random.randn(p_b,p_b)
	R_LL=np.dot(R_LL,R_LL.T)
	R_LK=np.random.randn(p_c,p_c)
	R_LK=np.dot(R_LK,R_LK.T)
	R_KL=np.random.randn(p_b,p_b)
	R_KL=np.dot(R_KL,R_KL.T)
	R_KK=np.random.randn(p_c,p_c)
	R_KK=np.dot(R_KK,R_KK.T)
	Q_L=np.random.randn(p_x,p_x)
	Q_L=np.dot(Q_L,Q_L.T)
	Q_K=np.random.randn(p_x,p_x)
	Q_K=np.dot(Q_K,Q_K.T)
	if type_pi=='exact':
		Exact_Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,k_m=100)
	elif type_pi=='modified':
		Modified_Policy_Iteration(A,B,C,R_KL,R_LL,R_KK,R_LK,Q_K,Q_L,k_m=100)
if __name__=='__main__':
	exp()
	exp(type_pi='modified')
