# This code is part of kagome-vqe.
#
# (C) Copyright LJSB Enterprises, LLC 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""IBM Open Challenge 2022 KAGOME VQE LATTICE tools


@author: rrodenbusch
"""
# -------------  Lattices and Hamiltonians ----------- #
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import EfficientSU2

import numpy as np
import hamiltonians

# Mapping lattice node to device qubits
qbitmap_3_to_5   = [0,1,2]      # lima, belem, quito, manila
qbitmap_3_to_7   = [0,1,2]      # nairobi oslo lagos perth jakarta

qbitmap_4_to_4   = [0,1,2,3]
qbitmap_4_to_5   = [0,1,2,3]
qbitmap_4_to_7   = [0,1,2,3]
qbitmap_4_to_16  = [1,2,3,4]

qbitmap_5_to_5   = [0,1,2,3,4]
qbitmap_5_to_7   = [0,1,2,3,5]  # nairobi, oslo, jakarta, perth, lagos
qbitmap_5_to_16  = [1,2,3,4,5]  # guadalupe

qbitmap_12_to_16 = [1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4] # for guadalupe and FakeGuadalupe


def list_Ansatz(A):
    def get_Alabel(A,name=None):
        if name is None:
            name = A.name
        return f"{name} nqbits:{A.num_qubits} nparams:{A.num_parameters}"
    if isinstance(A,dict):
        for key,value in A.items():
            print(f"'{key}'\t {get_Alabel(value)} gates:{str(dict(value.count_ops()))}")
    else:
        print(f"{get_Alabel(A)} gates:{str(dict(A.count_ops()))}")

# Build a custom ansatz from scratch with optional starting circuit
def rotationQC(nqubits, r0, name=''):
    qc=QuantumCircuit(nqubits,name=name)
    for idx in range(1,nqubits):
        _=qc.rx(idx*r0,idx)
    return qc

def rotationIsing(num_qubits,qc=None,layers=3,name='rotationIsing'):
    from qiskit.circuit import Parameter
    if qc is None:
        ansatz = QuantumCircuit(num_qubits)
    else:
        ansatz = qc.copy(name=name)
        num_qubits  = qc.num_qubits
    # Mixing Layers
    j=0
    for i in range(num_qubits):
        if layers > 0:
            ansatz.rx(Parameter('θ_' + str(j)), i)
            j += 1
        if layers > 1:
            ansatz.ry(Parameter('θ_' + str(j)), i)
            j += 1
        if layers > 2:
            ansatz.rz(Parameter('θ_' + str(j)), i)
            j += 1
    return ansatz

def customSU2(num_qubits,qc=None,layers=3,name='customSU2'):
    from qiskit.circuit import Parameter
    if qc is None:
        customSU2 = QuantumCircuit(num_qubits)
    else:
        customSU2 = qc.copy(name=name)
        num_qubits = qc.num_qubits
    # Mixing Layers
    ESU2 = EfficientSU2(num_qubits, entanglement='linear',
                        reps=layers, skip_final_rotation_layer=True).decompose()
    customSU2 = customSU2.compose(ESU2)
    return customSU2

def customAnsatz2(num_qubits,qc=None,layers=0,name='ansatz2'):
    from qiskit.circuit import Parameter
    if qc is None:
        ansatz2 = QuantumCircuit(num_qubits)
        ansatz2 = ansatz2.extend(customAnsatz1(num_qubits))
    else:
        ansatz2 = qc.copy(name=name)
        num_qubits = qc.num_qubits
    # Mixing Layers
    j = 1
    for l in range(1,layers):
        for i in range(num_qubits):
            ansatz2.rz(Parameter('θ_' + str(j)), i)
            j += 1
            ansatz2.ry(Parameter('θ_' + str(j)), i)
            j += 1
        ansatz2.cx(range(0, num_qubits-1), range(1, num_qubits))
    return ansatz2

def customAnsatz3(num_qubits, qc=None, reps=3, name='ansatz3', qubits=None,
                  coupling=None, deltas=None, nParams=None):
    from qiskit.circuit import Parameter
    A = QuantumCircuit(num_qubits,name=name) if qc is None else qc.copy(name=name)
    num_qubits = A.num_qubits
    # Default coupling is linear
    if coupling is None:
        coupling = []
        for i in range(num_qubits-1):
            coupling.append([i,i+1])
    # print(f"Coupling {coupling}")
    qubits = list(range(num_qubits)) if num_qubits is None else qubits
    nParams = 2*len(qubits) if nParams is None else nParams
    deltas = [0,0] if deltas is None else deltas
    params = []
    for idx in range(nParams):
        params.append(Parameter(f'θ_{idx}'))

    (cycle,pIdx) = (0,0)
    for r in range(reps):
        for curQbit in qubits:   # Repeat rx(t) sx rx(t)
            A.rz(params[pIdx], curQbit)
            A.sx(curQbit)
            A.rz(params[pIdx+1], curQbit)
            pIdx += 2
            if pIdx >= nParams:   # Reset to start of the cycle
                cycle += 1
                pIdx = 0

        for curMap in coupling:  # Add the cx for each coupled qbit
            if len(curMap) > 1:
                A.cx(curMap[0],curMap[1])

    return A

def customAnsatz4(num_qubits, qubits=None, qc=None, name='ansatz4',debug=False,
                  couplings=None,nParams=None, loops=None):
    from qiskit.circuit import Parameter
    A = QuantumCircuit(num_qubits,name=name) if qc is None else qc.copy(name=name)
    num_qubits = A.num_qubits
    qubits = list(range(num_qubits)) if qubits is None else qubits
    # Default coupling is linear, one pass (Same as 3)
    if couplings is None:
        loops = 1
        coupling = []
        for i in range(num_qubits-1):
            coupling.append([i,i+1])
        couplings = [ [coupling] ]
    elif loops is None:
        loops = 1
        couplings = [ couplings ]
    params = []
    nParams = 2*len(qubits) if nParams is None else nParams
    for pIdx in range(nParams):
        params.append(Parameter(f"θ_{pIdx}"))

    idx = 0
    if debug:
        print(f"nParams {len(params)}\n{params}")
    for curLoop in range(loops):
        curCoupling = couplings[curLoop]
        curQbits = qubits[curLoop]
        if debug:
            print(f"\tLoop {curLoop}")
            print(f"\t\tqbits {curQbits}")
        for coupling in curCoupling:
            if debug:
                print(f"\t\t\tcoupling {coupling}")
            for curQbit in curQbits:   # Repeat rx(t) sx rx(t)
                A.rz(params[idx], curQbit)
                A.sx(curQbit)
                A.rz(params[idx+1], curQbit)
                idx = idx+2 if idx+3 < nParams else 0
                
            for curMap in coupling:  # Add the cx for each coupled qbit
                if len(curMap) > 1:
                    A.cx(curMap[0],curMap[1])

    return A

def customAnsatz1(num_qubits,name='cust1'):
    from qiskit.circuit import Parameter
    ansatz_custom = QuantumCircuit(num_qubits,name=name)
    # build initial state
    ansatz_custom.h(range(0, num_qubits, 2))
    ansatz_custom.cx(range(0, num_qubits-1, 2), range(1, num_qubits, 2))
    # First layer
    j = 0
    for i in range(num_qubits):
        ansatz_custom.rz(Parameter('θ_' + str(j)), i)
        j += 1
        ansatz_custom.ry(Parameter('θ_' + str(j)), i)
        j += 1
    ansatz_custom.cx(range(0, num_qubits-1), range(1, num_qubits))
    return ansatz_custom


def init_ansatze(H=None,backends=None,targets=None,optimization_level=1):

    Anzs = {}
    circuits = {}
    hams = H if H is not None else hamiltonians.init_hams()


    ######## Triangles #########
    E='S3'
    baseName = f"A3_3_SU2_{E}"
    Anzs[baseName] = EfficientSU2(3, entanglement='sca', reps=3, su2_gates=['ry','rz'],
                           name=baseName,skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs[f'A3_a_SU2_{E}']  = transpile(Anzs[baseName], backend=backends['a'], # lima
                                 initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs[f'A3_b_SU2_{E}']  = transpile(Anzs[baseName], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs[f'A3_m_SU2_{E}']  = transpile(Anzs[baseName], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs[f'A3_q_SU2_{E}']  = transpile(Anzs[baseName], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs[f'A3_p_SU2_{E}'] = transpile(Anzs[baseName], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs[f'A3_l_SU2_{E}']  = transpile(Anzs[baseName], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level )

    ######## Triangles #########
    Anzs['A3_3_SU2_F1'] = EfficientSU2(3, entanglement='full', reps=1, su2_gates=['ry','rz'],
                           name='A3_3_SU2_F1',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['A3_a_SU2_F1']  = transpile(Anzs['A3_3_SU2_F1'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_b_SU2_F1']  = transpile(Anzs['A3_3_SU2_F1'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_m_SU2_F1']  = transpile(Anzs['A3_3_SU2_F1'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_q_SU2_F1']  = transpile(Anzs['A3_3_SU2_F1'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['A3_p_SU2_F1'] = transpile(Anzs['A3_3_SU2_F1'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs['A3_l_SU2_F1']  = transpile(Anzs['A3_3_SU2_F1'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level )

    ######## Triangles #########
    Anzs['A3_3_SU2_C4'] = EfficientSU2(3, entanglement='circular', reps=4,
                           name='A3_3_SU2_C4',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['A3_a_SU2_C4']  = transpile(Anzs['A3_3_SU2_C4'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_b_SU2_C4']  = transpile(Anzs['A3_3_SU2_C4'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_m_SU2_C4']  = transpile(Anzs['A3_3_SU2_C4'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_q_SU2_C4']  = transpile(Anzs['A3_3_SU2_C4'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['A3_p_SU2_C4'] = transpile(Anzs['A3_3_SU2_C4'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs['A3_l_SU2_C4']  = transpile(Anzs['A3_3_SU2_C4'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)

    ######## Triangles #########
    Anzs['A3_3_SU2_C1'] = EfficientSU2(3, entanglement='circular', reps=1,
                           name='A3_3_SU2_C1',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['A3_a_SU2_C1']  = transpile(Anzs['A3_3_SU2_C1'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_b_SU2_C1']  = transpile(Anzs['A3_3_SU2_C1'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_m_SU2_C1']  = transpile(Anzs['A3_3_SU2_C1'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_q_SU2_C1']  = transpile(Anzs['A3_3_SU2_C1'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['A3_p_SU2_C1'] = transpile(Anzs['A3_3_SU2_C1'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs['A3_l_SU2_C1']  = transpile(Anzs['A3_3_SU2_C1'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level )

    ######## Triangles #########
    Anzs['A3_3_SU2_L1'] = EfficientSU2(3, entanglement='linear', reps=1,
                           name='A3_3_SU2_L1',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['A3_a_SU2_L1']  = transpile(Anzs['A3_3_SU2_L1'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_b_SU2_L1']  = transpile(Anzs['A3_3_SU2_L1'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_m_SU2_L1']  = transpile(Anzs['A3_3_SU2_L1'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_q_SU2_L1']  = transpile(Anzs['A3_3_SU2_L1'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['A3_p_SU2_L1'] = transpile(Anzs['A3_3_SU2_L1'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs['A3_l_SU2_L1']  = transpile(Anzs['A3_3_SU2_L1'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level )

    ######## Triangles #########
    Anzs['A3_3_SU2_L2'] = EfficientSU2(3, entanglement='linear', reps=2,
                           name='A3_3_SU2_L2',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['A3_a_SU2_L2']  = transpile(Anzs['A3_3_SU2_L2'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_b_SU2_L2']  = transpile(Anzs['A3_3_SU2_L2'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_m_SU2_L2']  = transpile(Anzs['A3_3_SU2_L2'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_q_SU2_L2']  = transpile(Anzs['A3_3_SU2_L2'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['A3_p_SU2_L2'] = transpile(Anzs['A3_3_SU2_L2'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs['A3_l_SU2_L2']  = transpile(Anzs['A3_3_SU2_L2'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level )

    ######## Triangles #########
    Anzs['A3_3_SU2_L3'] = EfficientSU2(3, entanglement='linear', reps=3,
                           name='A3_3_SU2_L3',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['A3_a_SU2_L3']  = transpile(Anzs['A3_3_SU2_L3'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_b_SU2_L3']  = transpile(Anzs['A3_3_SU2_L3'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_m_SU2_L3']  = transpile(Anzs['A3_3_SU2_L3'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    Anzs['A3_q_SU2_L3']  = transpile(Anzs['A3_3_SU2_L3'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_3_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['A3_p_SU2_L3'] = transpile(Anzs['A3_3_SU2_L3'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level)
    Anzs['A3_l_SU2_L3']  = transpile(Anzs['A3_3_SU2_L3'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_3_to_7, optimization_level=optimization_level )



    ########  Squares ##########
    Anzs['Asq_4_SU2_L1'] = EfficientSU2(4, entanglement='linear', reps=1,
                           name='Asq_4_SU2_L1',skip_final_rotation_layer=True).decompose()
    ### 5 qubits
    Anzs['Asq_a_SU2_L1']  = transpile(Anzs['Asq_4_SU2_L1'], backend=backends['a'], # lima
                                     initial_layout=qbitmap_4_to_5, optimization_level=optimization_level)
    Anzs['Asq_b_SU2_L1']  = transpile(Anzs['Asq_4_SU2_L1'], backend=backends['b'], # belem
                                     initial_layout=qbitmap_4_to_5, optimization_level=optimization_level)
    Anzs['Asq_m_SU2_L1']  = transpile(Anzs['Asq_4_SU2_L1'], backend=backends['m'], # manila
                                     initial_layout=qbitmap_4_to_5, optimization_level=optimization_level)
    Anzs['Asq_q_SU2_L1']  = transpile(Anzs['Asq_4_SU2_L1'], backend=backends['q'], # quito
                                     initial_layout=qbitmap_4_to_5, optimization_level=optimization_level)
    ### 7 qubits
    Anzs['Asq_p_SU2_opt'] = transpile(Anzs['Asq_4_SU2_L1'], backend=backends['p'], # perth
                                     initial_layout=qbitmap_4_to_7, optimization_level=optimization_level)
    Anzs['Asq_l_SU2_L1']  = transpile(Anzs['Asq_4_SU2_L1'], backend=backends['l'], # lagos
                                     initial_layout=qbitmap_4_to_7, optimization_level=optimization_level)


    # Anzs['A3_SU2'] = EfficientSU2(3, entanglement='linear', reps=3,
    #                                  name='A3_SU2',
    #                                  skip_final_rotation_layer=True).decompose()
    # Anzs['A3_7_SU2_opt'] = transpile(Anzs['A3_SU2'], backend=backends['7'], initial_layout=qbitmap_3_to_7)
    # Anzs['A3_7_SU2_opt'].name = 'A3_7_SU2_opt'


    Anzs['A4_SU2'] = EfficientSU2(4, entanglement='linear', reps=3,
                                     name='A4_SU2',skip_final_rotation_layer=True).decompose()

    Anzs['A4_7_SU2_opt'] = transpile(Anzs['A4_SU2'], backend=backends['7'], initial_layout=qbitmap_4_to_7)
    Anzs['A4_7_SU2_opt'].name = 'A4_7_SU2_opt'


    Anzs['A5_5_SU2_L1'] = EfficientSU2(5, entanglement='linear', reps=1,
                           name='A5_5_SU2_L1',skip_final_rotation_layer=True).decompose()
    Anzs['A5_7_SU2_L1'] = transpile(Anzs['A5_5_SU2_L1'], backend=backends['7'],
                                    initial_layout=qbitmap_5_to_7 )
    Anzs['A5_l_SU2_L1'] = transpile(Anzs['A5_5_SU2_L1'], backend=backends['l'],
                                    initial_layout=qbitmap_5_to_7 )
    Anzs['A5_p_SU2_L1'] = transpile(Anzs['A5_5_SU2_L1'], backend=backends['p'],
                                    initial_layout=qbitmap_5_to_7 )
    Anzs['A5_n_SU2_L1'] = transpile(Anzs['A5_5_SU2_L1'], backend=backends['n'],
                                    initial_layout=qbitmap_5_to_7 )
    Anzs['A5_o_SU2_L1'] = transpile(Anzs['A5_5_SU2_L1'], backend=backends['o'],
                                    initial_layout=qbitmap_5_to_7 )
    Anzs['A5_j_SU2_L1'] = transpile(Anzs['A5_5_SU2_L1'], backend=backends['j'],
                                    initial_layout=qbitmap_5_to_7 )
    Anzs['A5_16_SU2_L1']= transpile(Anzs['A5_5_SU2_L1'], backend=backends['16'],
                                    initial_layout=qbitmap_5_to_16 )
    Anzs['A5_g_SU2_L1']= transpile(Anzs['A5_5_SU2_L1'], backend=backends['g'],
                                    initial_layout=qbitmap_5_to_16 )

    Anzs['A5_5_SU2_L2'] = EfficientSU2(5, entanglement='linear', reps=2,
                           name='A5_5_SU2_L2',skip_final_rotation_layer=True).decompose()
    Anzs['A5_a_SU2_L2'] = transpile(Anzs['A5_5_SU2_L2'], backend=backends['a'],
                                    initial_layout=qbitmap_5_to_5 )
    Anzs['A5_b_SU2_L2'] = transpile(Anzs['A5_5_SU2_L2'], backend=backends['b'],
                                    initial_layout=qbitmap_5_to_5 )
    Anzs['A5_m_SU2_L2'] = transpile(Anzs['A5_5_SU2_L2'], backend=backends['m'],
                                    initial_layout=qbitmap_5_to_5 )
    Anzs['A5_q_SU2_L2'] = transpile(Anzs['A5_5_SU2_L2'], backend=backends['q'],
                                    initial_layout=qbitmap_5_to_5 )


    Anzs['A5_5_SU2_L3'] = EfficientSU2(5, entanglement='linear', reps=3,
                           name='A5_5_SU2_L3',skip_final_rotation_layer=True).decompose()
    Anzs['A5_a_SU2_L3'] = transpile(Anzs['A5_5_SU2_L3'], backend=backends['a'],
                                    initial_layout=qbitmap_5_to_5 )
    Anzs['A5_b_SU2_L3'] = transpile(Anzs['A5_5_SU2_L3'], backend=backends['b'],
                                    initial_layout=qbitmap_5_to_5 )
    Anzs['A5_m_SU2_L3'] = transpile(Anzs['A5_5_SU2_L3'], backend=backends['m'],
                                    initial_layout=qbitmap_5_to_5 )
    Anzs['A5_q_SU2_L3'] = transpile(Anzs['A5_5_SU2_L3'], backend=backends['q'],
                                    initial_layout=qbitmap_5_to_5 )

    Anzs['A5_16_SU2_L1']= transpile(Anzs['A5_5_SU2_L1'], backend=backends['16'],
                                    initial_layout=qbitmap_5_to_16 )
    Anzs['A5_g_SU2_L1']= transpile(Anzs['A5_5_SU2_L1'], backend=backends['g'],
                                    initial_layout=qbitmap_5_to_16 )
    Anzs['A5_5_SU2_L3'] = EfficientSU2(5, entanglement='linear', reps=3,
                           name='A5_5_SU2_L3',skip_final_rotation_layer=True).decompose()
    Anzs['A5_5_SU2_L4'] = EfficientSU2(5, entanglement='linear', reps=4,
                           name='A5_5_SU2_L4',skip_final_rotation_layer=True).decompose()
    Anzs['A5_5_SU2_L5'] = EfficientSU2(5, entanglement='linear', reps=5,
                           name='A5_5_SU2_L5',skip_final_rotation_layer=True).decompose()

    Anzs['A7_7_SU2_L1'] = EfficientSU2(7, entanglement='linear', reps=1,
                           name='A7_7_SU2_L1',skip_final_rotation_layer=True).decompose()
    Anzs['A7_7_SU2_L2'] = EfficientSU2(7, entanglement='linear', reps=2,
                           name='A7_7_SU2_L2',skip_final_rotation_layer=True).decompose()
    Anzs['A7_7_SU2_L3'] = EfficientSU2(7, entanglement='linear', reps=3,
                           name='A7_7_SU2_L3',skip_final_rotation_layer=True).decompose()
    Anzs['A7_7_SU2_L4'] = EfficientSU2(7, entanglement='linear', reps=4,
                           name='A7_7_SU2_L4',skip_final_rotation_layer=True).decompose()
    Anzs['A7_7_SU2_L5'] = EfficientSU2(7, entanglement='linear', reps=5,
                           name='A7_7_SU2_L5',skip_final_rotation_layer=True).decompose()

    Anzs['A12_SU2'] = EfficientSU2(12, entanglement='linear', reps=3,
                           name='A12_SU2',skip_final_rotation_layer=True).decompose()

    Anzs['A12_SU2_opt'] = transpile(Anzs['A12_SU2'], backend=backends['16'], initial_layout=qbitmap_12_to_16)
    Anzs['A12_SU2_opt'].name = 'A12_SU2_opt'

    Anzs['A3_cust1']         = customAnsatz1(3,name='A3_cust1')
    Anzs['A3_7_cust1']       = transpile(Anzs['A3_cust1'], backend=backends['7'], initial_layout=qbitmap_3_to_7)
    Anzs['A3_7_cust1'].name  = 'A3_7_cust1'

    Anzs['A4_cust1']         = customAnsatz1(4, name='A4_cust1')
    Anzs['A4_7_cust1']       = transpile(Anzs['A4_cust1'], backend=backends['7'], initial_layout=qbitmap_4_to_7)
    Anzs['A4_7_cust1'].name  = 'A4_7_cust1'

    Anzs['A12_cust1']        = customAnsatz1(12,name='A12_cust1')
    Anzs['A12_16_cust1']      = transpile(Anzs['A12_cust1'], backend=backends['16'], initial_layout=qbitmap_12_to_16)
    Anzs['A12_16_cust1'].name = 'A12_16_cust1'

    for key, value in Anzs.items():
        Anzs[key].name = key

    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_I_L3'
    Hlabel = '4_4'
    nqubits=4
    circuits[Alabel]=QuantumCircuit(nqubits,name=Alabel)
    _=circuits[Alabel].x(1)
    _=circuits[Alabel].x(3)
    Anzs[Alabel]  = rotationIsing(4,qc=circuits[Alabel],name='|0101> A4_Ising_L3',layers=3)
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()

    # ------------------------------------------------------------------------------ #
    Alabel = 'A3_I_L1'
    Hlabel = '3_3'
    nqubits=3
    r0 = 2*np.pi/3.0
    circuits[Alabel] = rotationQC(nqubits,r0,name='|R(pi/3)>')
    Anzs[Alabel]  = rotationIsing(nqubits,qc=circuits[Alabel],name='|R(pi/3)> A3_Ising_L1',layers=3)
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    # circuits[Alabel].draw()
    Anzs[Alabel].draw()


    # ------------------------------------------------------------------------------ #
    Alabel = 'A3_I0_L1'
    Hlabel = '3_3'
    nqubits=3
    circuits[Alabel]=QuantumCircuit(nqubits,name='|000>')
    Anzs[Alabel]  = rotationIsing(nqubits,qc=circuits[Alabel],name='|000> A3_Ising_L1',layers=3)
    E = Statevector(circuits[Alabel]).expectation_value(hams['3_3'])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()

    # ------------------------------------------------------------------------------ #
    Alabel = 'A3_I1_L1'
    Hlabel = '3_3'
    nqubits=3
    circuits[Alabel]=QuantumCircuit(nqubits,name='|101>')
    _=circuits[Alabel].x(range(0, nqubits, 2))
    Anzs[Alabel]  = rotationIsing(nqubits,qc=circuits[Alabel],name='|101> A3_Ising_L1',layers=3)
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    # circuits[Alabel].draw()
    Anzs[Alabel].draw()

    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_I4_L1'
    Hlabel = 'sq_4'
    nqubits=4
    r4 = np.pi/2.0
    circuits[Alabel] = rotationQC(nqubits,r4,name='|R(pi/2)>')
    Anzs[Alabel]  = rotationIsing(nqubits,qc=circuits[Alabel],name='|0xy0> A4_Ising_L1',layers=3)
    # angles = getBlochAngles(qc2Statevector(circuits[Alabel]))
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()

    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_I5_L1'
    Hlabel = 'sq_4'
    nqubits=4
    circuits[Alabel] = QuantumCircuit(nqubits)
    Anzs[Alabel] = rotationIsing(nqubits,qc=circuits[Alabel],name='|0000> {Alabel}',layers=3)
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()

    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_I6_L1'
    Hlabel = '4_4_BC1'
    nqubits=4
    print(f"-------- {Alabel} -------------")
    circuits[Alabel] = QuantumCircuit(nqubits,name=Alabel)
    _=circuits[Alabel].x([1,3])
    _=circuits[Alabel].rx(np.pi/2.0,2)
    Anzs[Alabel] = rotationIsing(nqubits,qc=circuits[Alabel],name=f'|0000> {Alabel}',layers=3)
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()
    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_SU2_X0'
    Hlabel = '4_4_BC1'
    nqubits=4
    print(f"-------- {Alabel} -------------")
    circuits[Alabel] = QuantumCircuit(nqubits,name=Alabel)
    Anzs[Alabel] = customSU2(nqubits,qc=circuits[Alabel],layers=3,name=f'|0000> {Alabel}')
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()
    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_SU2_X1'
    Hlabel = '4_4_BC1'
    nqubits=4
    circuits[Alabel] = QuantumCircuit(nqubits,name=Alabel)
    _=circuits[Alabel].x([1,3])
    _=circuits[Alabel].rx(np.pi/2.0,2)
    Anzs[Alabel] = customSU2(nqubits,qc=circuits[Alabel],layers=3,name=f'|1x10> {Alabel}')
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()
    # ------------------------------------------------------------------------------ #
    Alabel = 'A4_SU2_X2'
    Hlabel = 'sq_4'
    nqubits=4
    circuits[Alabel] = QuantumCircuit(nqubits,name=Alabel)
    _=circuits[Alabel].x([1,3])
    Anzs[Alabel] = customSU2(nqubits,qc=circuits[Alabel],layers=3,name=f'|1010> {Alabel}')
    E = Statevector(circuits[Alabel]).expectation_value(hams[Hlabel])
    print(f"-------- {Alabel} -------------")
    print(f"A[{Alabel}].expectation({Hlabel})={np.around(np.real(E),4)} E0={np.around(targets[Hlabel],3)}")
    Anzs[Alabel].draw()

    return Anzs


def stack_couplings(Anzs=None, qubits=None, Adevices=None, couplings=None,
                    backends=None, devices=['a'], optimization_level=3,
                    seed_transpiler=None, nParams=None,loops=1,
                    nNodes=3, nQubits=5, qbitmap=None, debug=False ):
    Anzs = {} if Anzs is None else Anzs
    Adevices = {} if Adevices is None else Adevices
    qbitmap = list(range(nQubits)) if qbitmap is None else qbitmap

    #################################### Loop through Entanglements, Reps, and Devices
    for E, coupling in couplings.items():
        # Anzs[f'A{nNodes}_{nNodes}_{E}_L1'] = baseQC = customAnsatz4(nNodes,qubits=None,
        #                                     qc=None, name=f'A{nNodes}_{nNodes}_{E}_L1',
        #                                     couplings=coupling )
        Anzs[f'A{nNodes}_{nQubits}_{E}_L1'] = baseQC = customAnsatz4(nQubits, qc=None, qubits=qubits,loops=loops, debug=debug,
                                             name=f'A{nNodes}_{nQubits}_{E}_L1', couplings=coupling, nParams=nParams )
        for D in devices:
            Anzs[f'A{nNodes}_{D}_{E}_L1'] = transpile(baseQC, backend=backends[D], initial_layout=qbitmap,
                                                      optimization_level=optimization_level,
                                                      seed_transpiler=seed_transpiler)
            Adevices[f'A{nNodes}_{D}_{E}_L1'] = D
    ####################################

    for key in Anzs.keys():
        Anzs[key].name = key
    return Anzs, Adevices


def local_ansatze(Anzs=None, Adevices=None,
                  couplings={'SA1':[[0,1],[1,2],[1,3],],},
                  reps=[2,1,], qubits=[0,1,2,],
                  backends=None, devices=['a'],
                  nNodes=3,nQubits=5,qbitmap=[0,1,2,], ):
    Anzs = {} if Anzs is None else Anzs
    Adevices = {} if Adevices is None else Adevices

    #################################### Loop through Entanglements, Reps, and Devices
    for E, coupling in couplings.items():
        for R in reps:
            Anzs[f'A{nNodes}_{nNodes}_{E}_L{R}'] = customAnsatz3(np.max(qubits)+1,
                                                qc=None, reps= R,qubits=qubits,
                                                name=f'A{nNodes}_{nNodes}_{E}_L{R}',
                                                coupling=coupling )
            Anzs[f'A{nNodes}_{nQubits}_{E}_L{R}'] = baseQC = customAnsatz3(nQubits, qc=None, reps= R,qubits=qubits,
                                                name=f'A{nNodes}_{nQubits}_{E}_L{R}', coupling=coupling )
            for D in devices:
                Anzs[f'A{nNodes}_{D}_{E}_L{R}'] = transpile(baseQC, backend=backends[D], initial_layout=qbitmap)
                Adevices[f'A{nNodes}_{D}_{E}_L{R}'] = D
    ####################################

    for key in Anzs.keys():
        Anzs[key].name = key
    return Anzs, Adevices

def Anzs_images(Akeys=None,Anzs=None,):
    images = {}
    if Akeys is None or len(Akeys)<1:
        Akeys=list(Anzs.keys())
    for Akey in Akeys:
        images[Akey] = Anzs[Akey].draw()
    return images

def Anzs_schedules( Akeys=None, Anzs={}, backends={} ):
    from qiskit import schedule as build_schedule
    (schedules,images,skip) = ({},{},'No Backend ')
    Akeys = list(Anzs.keys()) if Akeys is None else Akeys
    for Akey in Akeys:
        backend = backends.get(Akey,None)
        if backend is not None:
            schedules[Akey] = build_schedule(Anzs[Akey],backend)
            images[Akey] = schedules[Akey].draw(backend=backend)
        else:
            print(f"{skip} {Akey},", end="")
            skip=''
    return schedules, images

def eval_schedules(schedules,devices,Akeys=None,qubits=[0,1,2,]):
    def min_Decoherence(backend,qubits=None):
        if qubits is None:
            qubits = list(range(backend.configuration().n_qubits))
        properties = backend.properties()
        minT1=None
        minT2=None
        for curBit in qubits:
            if minT1 is None or (minT1 > properties.t1(curBit)):
                minT1 = properties.t1(curBit)
            if minT2 is None or (minT2 > properties.t2(curBit)):
                minT2 = properties.t2(curBit)
        return minT1,minT2
    def max_error_rate(backend,qubits=None):
        if qubits is None:
            qubits = list(range(backend.configuration().n_qubits))
        properties = backend.properties()
        maxRate=None
        for curBit in qubits:
            if maxRate is None or (maxRate < properties.readout_error(curBit)):
                maxRate = properties.readout_error(curBit)
        return maxRate
    import pandas as pd
    Akeys = list(schedules.keys()) if Akeys is None else Akeys
    dataList = []
    for key in Akeys:
        dt = devices[key].configuration().dt
        minT1,minT2=min_Decoherence(devices[key],qubits=qubits)
        maxError = max_error_rate(devices[key],qubits=qubits)
        duration = schedules[key].duration
        T1_ratio = duration*dt / minT1
        T2_ratio = duration*dt / minT2
        T1_decay = np.exp(-1*T1_ratio)
        T2_decay = np.exp(-1*T2_ratio)

        dataList.append( [key, 100*maxError,
                          100*T1_decay,100*T2_decay,
                          len(schedules[key]._parameter_manager.parameters),
                          duration*dt/ns,
                          minT1/us, minT2/us,
                          T1_ratio, T2_ratio ] )

    columns=['Anzs','MeasErr%','T1-Decay','T2-Decay','Parms','t (ns)','T1(us)','T2(us)','t/T1','t/T2',]
    df = pd.DataFrame(dataList, columns=columns)
    def make_pretty(styler):
        styler.set_caption("Schedule Evaluations")
        styler.hide(axis="index")
        styler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        styler.format(precision=2,thousands=",")
        styler.format(subset=['t/T1','t/T2'],precision=4)

        styler.set_properties(subset=["Anzs"], **{'text-align': 'center'})
        return styler
    return df.style.pipe(make_pretty)

ns = 1.0e-9 # Nanoseconds
us = 1.0e-6 # Microseconds


