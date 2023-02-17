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
qbitmap_12_to_16 = [1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4] # for guadalupe and FakeGuadalupe
qbitmap_4_to_7   = [0,1,2,3]    # For nairobi and oslo
qbitmap_3_to_7   = [0,1,2]      # For nairobi and oslo


def list_Ansatz(A):
    def get_Alabel(A,name=None):
        if name is None:
            name = A.name
        return f"{name} n:{3} q:{A.num_qubits} p:{A.num_parameters}"
    if isinstance(A,dict):
        for key,value in A.items():
            print(f"'{key}'\t {get_Alabel(value)} g:{str(dict(value.count_ops()))}")
    else:
        print(f"{get_Alabel(A)} g:{str(dict(A.count_ops()))}")

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


def init_ansatze(H=None,backends=None,targets=None):

    Anzs = {}
    circuits = {}
    hams = H if H is not None else hamiltonians.init_hams()

    Anzs['A3_SU2'] = EfficientSU2(3, entanglement='linear', reps=3,
                                     name='A3_SU2',
                                     skip_final_rotation_layer=True).decompose()
    Anzs['A3_7_SU2_opt'] = transpile(Anzs['A3_SU2'], backend=backends['7'], initial_layout=qbitmap_3_to_7)
    Anzs['A3_7_SU2_opt'].name = 'A3_7_SU2_opt'

    Anzs['A4_SU2'] = EfficientSU2(4, entanglement='linear', reps=3,
                                     name='A4_SU2',skip_final_rotation_layer=True).decompose()

    Anzs['A4_7_SU2_opt'] = transpile(Anzs['A4_SU2'], backend=backends['7'], initial_layout=qbitmap_4_to_7)
    Anzs['A4_7_SU2_opt'].name = 'A4_7_SU2_opt'


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
