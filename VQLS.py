# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:28:51 2021

@author: fushengzhijushi
"""

import numpy as np
import random
import paddle
from paddle_quantum.circuit import UAnsatz
from scipy.optimize import minimize

# -------------------------------------------------------------

# Toffoli/CCX Gate

def toffoli_gate(circ,a,b,c):
    z0=np.array([0], np.float64)
    z0=paddle.to_tensor(z0)
    pi1=np.array([np.pi],np.float64)
    pi1=paddle.to_tensor(pi1)
    
    circ.h(c)
    circ.cnot([b,c])
    circ.u3(z0,z0,-pi1/4,c)
    circ.cnot([a,c])
    circ.t(c)
    circ.cnot([b,c])
    circ.u3(z0,z0,-pi1/4,c)
    circ.cnot([a,c])
    circ.u3(z0,z0,-pi1/4,b)
    circ.t(c)
    circ.h(c)
    circ.cnot([a,b])
    circ.u3(z0,z0,-pi1/4,b)
    circ.cnot([a,b])
    circ.t(a)
    circ.s(b)
    
    return circ

# ---------------------------------------------------------------------------------

# U|0>=|b>, where U=H H H.

def control_b(auxiliary, qubits,circ):
    
    z0=np.array([0], np.float64)
    z0=paddle.to_tensor(z0)
    pi1=np.array([np.pi],np.float64)
    pi1=paddle.to_tensor(pi1)
    for ia in qubits:
        circ.s(ia)
        circ.h(ia)
        circ.t(ia)
        circ.cnot([auxiliary, ia])
        circ.u3(z0,z0,-pi1/4,ia)
        circ.h(ia)
        circ.u3(z0,z0,-pi1/2,ia)
        
    return circ

# -------------------------------------------------------------

# V(\alpha)

def apply_fixed_ansatz(qubits, parameters,circ):
    parameters = np.array(parameters, np.float64)
    parameters=paddle.to_tensor(parameters)
    for iz in range (0, len(qubits)):
        circ.ry(parameters[0][iz], qubits[iz])

    circ.h(qubits[1])
    circ.cnot([qubits[0], qubits[1]])
    circ.h(qubits[1])
    
    circ.h(qubits[0])
    circ.cnot([qubits[2], qubits[0]])
    circ.h(qubits[0])

    for iz in range (0, len(qubits)):
        circ.ry(parameters[1][iz], qubits[iz])
        

    circ.h(qubits[2])
    circ.cnot([qubits[1], qubits[2]])
    circ.h(qubits[2])
    
    circ.h(qubits[0])
    circ.cnot([qubits[2], qubits[0]])
    circ.h(qubits[0])

    for iz in range (0, len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])
    return circ


# --------------------------------------------------------------------------------

# Compute |<b|psi>|^2

def control_fixed_ansatz(qubits, parameters,auxiliary,circ): 
    parameters = np.array(parameters, np.float64)
    parameters = paddle.to_tensor(parameters)
    # print(parameters)
    z0=np.array([0], np.float64)
    z0=paddle.to_tensor(z0)
    for i in range (0, len(qubits)):
        circ.u3(parameters[0][i]/2.0,z0,z0,qubits[i])
        circ.cnot([auxiliary,qubits[i]])
        circ.u3(-parameters[0][i]/2.0,z0,z0,qubits[i])
        circ.cnot([auxiliary,qubits[i]])

    circ=toffoli_gate(circ,auxiliary, qubits[1], 4)
    circ.h(4)
    circ.cnot([qubits[0],4])
    circ.h(4)
    circ=toffoli_gate(circ,auxiliary, qubits[1], 4)
    
    circ=toffoli_gate(circ,auxiliary, qubits[0], 4)
    circ.h(4)
    circ.cnot([qubits[2],4])
    circ.h(4)
    circ=toffoli_gate(circ,auxiliary, qubits[0], 4)

    for i in range (0, len(qubits)):
        circ.u3(parameters[1][i]/2.0,z0,z0,qubits[i])
        circ.cnot([auxiliary,qubits[i]])
        circ.u3(-parameters[1][i]/2.0,z0,z0,qubits[i])
        circ.cnot([auxiliary,qubits[i]])

    circ=toffoli_gate(circ,auxiliary, qubits[2], 4)
    circ.h(4)
    circ.cnot([qubits[1],4])
    circ.h(4)
    circ=toffoli_gate(circ,auxiliary, qubits[2], 4)
    
    circ=toffoli_gate(circ,auxiliary, qubits[0], 4)
    circ.h(4)
    circ.cnot([qubits[2],4])
    circ.h(4)
    circ=toffoli_gate(circ,auxiliary, qubits[0], 4)

    for i in range (0, len(qubits)):
        circ.u3(parameters[2][i]/2.0,z0,z0,qubits[i])
        circ.cnot([auxiliary,qubits[i]])
        circ.u3(-parameters[2][i]/2.0,z0,z0,qubits[i])
        circ.cnot([auxiliary,qubits[i]])
    return circ

# -------------------------------------------------------------------------------

# Hadamard Test

def hadamard_test(gate_type, qubits, auxiliary_index, parameters,circ):

    circ.h(auxiliary_index)

    circ=apply_fixed_ansatz(qubits, parameters,circ)

    for ie in range (0, len(gate_type[0])):
        if (gate_type[0][ie] == 1):
            circ.h(qubits[ie])
            circ.cnot([auxiliary_index, qubits[ie]])
            circ.h(qubits[ie])

    for ie in range (0, len(gate_type[1])):
        if (gate_type[1][ie] == 1):
            circ.h(qubits[ie])
            circ.cnot([auxiliary_index, qubits[ie]])
            circ.h(qubits[ie])
    
    circ.h(auxiliary_index)
    return circ

# -----------------------------------------------------------------------------------

# 

def special_had_test(gate_type, qubits, auxiliary_index, parameters,circ):

    circ.h(auxiliary_index)

    circ=control_fixed_ansatz(qubits, parameters, auxiliary_index,circ)

    for ty in range (0, len(gate_type)):
        if (gate_type[ty] == 1):
            circ.h(qubits[ty])
            circ.cnot([auxiliary_index, qubits[ty]])
            circ.h(qubits[ty])

    circ=control_b(auxiliary_index, qubits,circ)
    
    circ.h(auxiliary_index)
    return circ


# ----------------------------------------------------------------------------------

# Cost function: 1-|<b|psi>|^2/<psi|psi>

def cost_function(parameters):
    
    overall_sum_1 = 0.0
    
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    parameters = np.array(parameters, np.float64)
    
    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):

            circ = UAnsatz(5)

            multiply = coefficient_set[i]*coefficient_set[j]

            circ=hadamard_test([gate_set[i], gate_set[j]], [1, 2, 3], 0, parameters,circ)
            
            # print(circ)
            
            outputstate=circ.run_state_vector()
            # print(outputstate)
            outputstate=outputstate.numpy()
            outputstate=np.real(outputstate)
            o = outputstate
            
            #print(o)
            
            m_sum = 0
            for l in range (0, len(o)):
                if (l>=16):
                    n = o[l]**2
                    m_sum+=n
            #print(m_sum)

            overall_sum_1+=multiply*(1-(2*m_sum))

    overall_sum_2 = 0.0

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):

            multiply = coefficient_set[i]*coefficient_set[j]
            mult = 1.0
            
            for extra in range(0, 2):

                circ2 = UAnsatz(5)

                if (extra == 0):
                    circ2=special_had_test(gate_set[i], [1, 2, 3], 0, parameters,circ2)
                if (extra == 1):
                    circ2=special_had_test(gate_set[j], [1, 2, 3], 0, parameters,circ2)
                
                
                outputstate=circ2.run_state_vector().numpy()
                outputstate=np.real(outputstate)
                o = outputstate

                m_sum = 0.0
                for l in range (0, len(o)):
                    if (l>=16):
                        n = o[l]**2
                        m_sum+=n
                mult = mult*(1-(2*m_sum))
                #print(mult)
                
            #print(multiply)
            overall_sum_2+=multiply*mult
            
            
    # print(overall_sum_1)
    # print(overall_sum_2)
    print(1-float(overall_sum_2/overall_sum_1))

    return 1-float(overall_sum_2/overall_sum_1)


# ---------------------------------------------------------------------------------

# Main

# A=0.25 I I I+0.75 I I Z

coefficient_set = [0.25, 0.75]
gate_set = [[0, 0, 0], [0, 0, 1]]

out = minimize(cost_function, x0=[float(random.randint(0,3000))/1000.0 for i in range(0, 9)], method="COBYLA", options={'maxiter':200})
print(out)

out_f = [out['x'][0:3], out['x'][3:6], out['x'][6:9]]

# print(out_f)

# ---------------------------------------------------------------------------------

# Solution

circ3 = UAnsatz(3)
circ3 = apply_fixed_ansatz([0, 1, 2], out_f,circ3)
outputstate=circ3.run_state_vector().numpy()
o = outputstate

recnum=o[1]
o[1]=o[4]
o[4]=recnum

recnum=o[3]
o[3]=o[6]
o[6]=recnum

# x, where Ax=b.

print("Solution:")
print(o)

# --------------------------------------------------------------------------------

# Test

# A (coefficient_set gate_set)

a1 = coefficient_set[1]*np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,0,-1,0,0], [0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,-1]])
a2 = coefficient_set[0]*np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
a3 = np.add(a1, a2)

# print(a3)

# b

b = np.array([float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8))])

# print(b)

print((b.dot(a3.dot(o)/(np.linalg.norm(a3.dot(o)))))**2)

# -----------------------------------------------------------------------------------

'''

     fun: 1.295935325718034e-08
   maxcv: 0.0
 message: 'Optimization terminated successfully.'
    nfev: 156
  status: 1
 success: True
       x: array([3.14154187, 0.95194946, 0.89340399, 2.14178952, 1.62493701,
       4.03514298, 0.57095282, 2.2437122 , 0.92722322])
Solution:
[ 0.22360824+0.j  0.22361047+0.j  0.22364906+0.j  0.22355195+0.j
 -0.44723179+0.j -0.44723291+0.j -0.4471858 +0.j -0.44720761+0.j]
(0.9999999870406493+0j)

'''

