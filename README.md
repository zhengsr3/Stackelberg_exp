# Stackelberg_exp
The main programme of this repository is stackelberg_basic_func.py. It contains three function:
Inner_Loop is design to find the solution of standard LQR game
Policy_Iteration take system parameters as input(x'=Ax+Bv+Cw,cost_leader=x^TQ_Lx+v^TR_LLv+w^TR_LKw,and cost_follower similiar).
Its type_pi parameter control the way of its policy evaluation and imporvement
('exact' update leader policy using Inner_Loop,'modified' evaluate the policy using one-step time-difference).
Its type_init parameters control the way of how the initial leader policy generate.
Exp take the design of the experiment as input
p_x,p_v,p_w is the dimension of x,v and w
type_matrix decided whether the matrix is generate randomly or read from file.
