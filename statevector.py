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
# -------------  Statevector visualization and transformations ----------- #
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_bloch_multivector

def qc2Statevector(qc):
    from qiskit import Aer
    qc1 = qc.copy()
    sim = Aer.get_backend('aer_simulator')
    qc1.save_statevector()
    result = sim.run(qc1).result()
    state_vector = result.get_statevector()
    return state_vector

def VQE2MiniSvector(custom_vqe):
    from qiskit import Aer
    sim = Aer.get_backend('aer_simulator')
    qc = custom_vqe.get_attr('miniAnsatz').assign_parameters(custom_vqe._result.optimal_parameters)
    return qc2Statevector(qc)

def VQE2Statevector(custom_vqe):
    from qiskit import Aer
    sim = Aer.get_backend('aer_simulator')
    qc = custom_vqe._circuit.assign_parameters(custom_vqe._result.optimal_parameters)
    return qc2Statevector(qc)

def getBlochCoords(state_vector,qubit,num_qubits=None):
    if num_qubits is None:
        num_qubits = len(state_vector.dims())
    if qubit >= num_qubits:
        return None

    from qiskit.quantum_info import SparsePauliOp
    Xop = SparsePauliOp.from_sparse_list([("X", [qubit], 1)], num_qubits=num_qubits)
    Yop = SparsePauliOp.from_sparse_list([("Y", [qubit], 1)], num_qubits=num_qubits)
    Zop = SparsePauliOp.from_sparse_list([("Z", [qubit], 1)], num_qubits=num_qubits)
    x = state_vector.expectation_value(Xop)
    y = state_vector.expectation_value(Yop)
    z = state_vector.expectation_value(Zop)
    return [x,y,z]

def getBlochAngles(state_vector):
    angles = []
    num_qubits = len(state_vector.dims())
    for qbit in range(num_qubits):
        coords = getBlochCoords(state_vector,qbit)
        (r,theta,phi) = cart2bloch(coords)
        angles.append([theta,phi,r])
    return angles

def getCoords(state_vector, num_qubits=None):
    coords = []
    if num_qubits is None:
        num_qubits = len(state_vector.dims())
    for qbit in range(num_qubits):
        coords.append(getBlochCoords(state_vector,qbit,num_qubits=num_qubits))
    return coords

def cart2bloch(pt):
    (x,y,z) = np.real(pt)
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy,z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def bloch2cart(pt):
    (r,theta,phi) = np.real(pt)
    xy = r*np.sin(theta)
    x  = xy*np.cos(phi)
    y  = xy*np.sin(phi)
    z  = r*np.cos(theta)
    return x, y, z

def bloch2circuit(angles,nqubits):
    qc = QuantumCircuit(nqubits)
    if len(angles) < nqubits:
        nqubits = len(angles)
    for nbit in range(nqubits):
        (theta, phi,r) = angles[nbit]
        qc.ry(theta,nbit)
        qc.rz(phi,nbit)
    return qc

def bloch2Statevector(angles,nqubits=None):
    if nqubits is None:
        nqubits = len(angles)
    qc = bloch2circuit(angles,nqubits)
    return Statevector(qc)

def getNormedState(custom_vqe,nqubits=None):
    if isinstance(custom_vqe,Statevector):
        init_svector = custom_vqe
    else:
        init_svector = VQE2Statevector(custom_vqe)
    angles = getBlochAngles(init_svector)
    normed_svector = bloch2Statevector(angles,nqubits=nqubits)
    if nqubits is None:
        nqubits = len(angles)
    ravg = 0
    for idx in range(nqubits):
        (theta, phi, r) = angles[idx]
        ravg += r**2
    ravg = np.sqrt(ravg/nqubits)

    return normed_svector, ravg

def displayNormedData(custom_vqe,H=None,nqubits=None):
    from qiskit.quantum_info import state_fidelity

    svector = VQE2Statevector(custom_vqe)
    if H is None:
        H=custom_vqe.get_attr('H')
    expSvector = np.real(svector.expectation_value(H))
    normed_svector, ravg = getNormedState(custom_vqe)
    coords = np.real(getCoords(normed_svector))
    expNormed = np.real(normed_svector.expectation_value(H))
    fidelity = state_fidelity(svector,normed_svector)
    print(f"{custom_vqe.label}: {np.around(expSvector,5)} "
          f"Initial    State Energy Level [Bloch(r)={np.around(ravg,3)}]")
    print("---------------------")
    print(f"{custom_vqe.label}: {np.around(expNormed,5)} "
          f"Normalized State Energy Level [Fidelity={np.around(fidelity,3)}]")
    if nqubits is not None:
        normed_svector, ravg = getNormedState(custom_vqe,nqubits=nqubits)

    return plot_bloch_multivector(normed_svector)