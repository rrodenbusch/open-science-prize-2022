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
import kagome

# cells = {}
# hams = {}
positions = {}


# Define edge lists for the lattices
t = 1    # All are weighted equally to start
edges_sq = [(0,1,t),(1,2,t),(2,3,t),(3,0,t)]
edges_3 = [(0, 1, t),(0, 2, t),(2, 1, t),]
edges_4 = [(0, 1, t),(0, 2, t),(2, 1, t),(1, 3, t),]
edges_5 = [(0, 1, t),(0, 2, t),(2, 1, t),(1, 3, t),(3, 4, t),(1, 4, t)]
edges_12 = [(0, 1, t), (0, 2, t),  (1, 2, t),  (1, 3, t),   (1, 4, t),  (3, 4, t),
            (4, 5, t), (4, 6, t),  (5, 6, t),  (5, 7, t),   (5, 8, t),  (7, 8, t),
            (7, 9, t), (7, 10, t), (9, 10, t), (10, 11, t), (10, 0, t), (11, 0, t) ]
edges_16 = [(1, 2, t),(2, 3, t),(3, 5, t),(5, 8, t),(8, 11, t),(11, 14, t),(14, 13, t),
            (13, 12, t),(12, 10, t),(10, 7, t),(7, 4, t),(4, 1, t),(4, 2, t),(2, 5, t),
            (5, 11, t),(11, 13, t),(13, 10, t),(10, 4, t),]

def init_positions():
    # Specify node locations for better visualizations
    pos16 = {0:[1,-1], 6:[1.5,-1], 9:[2,-1], 15:[2.5,-1],
              1:[0,-0.8], 2:[-0.6,1], 4:[0.6,1], 10:[1.2,3],
              13:[0.6,5], 11:[-0.6,5], 5:[-1.2,3], 3:[-1.8,0.9],
              8:[-1.8,5.1], 14:[0,6.8], 7:[1.8,0.9], 12:[1.8,5.1]}
    pos_sq  = { 0:pos16[13], 1:pos16[4], 2:pos16[2], 3:pos16[11], }
    pos7_sq = { 0:pos16[13], 1:pos16[4], 2:pos16[2], 3:pos16[11],
                4:[0.5,-1], 5:[1.0,-1], 6:[1.5,-1], }
    pos3    = { 0:pos16[13], 1:pos16[10], 2:pos16[12],
                3:[0.5,-1],  4:[0.75,-1], 5:[1.0,-1],  6:[1.25,-1],}
    pos4    = { 0:pos16[13], 1:pos16[10], 2:pos16[12], 3:pos16[7], }
    pos4a   = { 0:pos16[13], 1:pos16[10], 2:pos16[12], 3:[pos4[1][0],pos4[3][1]], }
    pos5    = { 0:pos16[13], 1:pos16[10], 2:pos16[12], 3:pos16[7],
                4:pos16[4], }
    pos7    = { 0:pos16[13], 1:pos16[10], 2:pos16[12], 3:[pos4[1][0],pos4[3][1]],
                4:[0.5,-1],  5:[0.75,-1], 6:[1.0,-1]}
    pos12   = { 0:pos16[13], 1:pos16[10], 2:pos16[12], 3:pos16[7],
                4:pos16[4],  5:pos16[2],  6:pos16[1],  7:pos16[5],
                8:pos16[3],  9:pos16[8], 10:pos16[11], 11:pos16[14]}
    positions = {'12_16':pos16, '12_12':pos12,
                 '5_5': pos5,
                 '4_7': pos7, '4_4': pos4,
                 '3_7': pos3, '3_3': pos3,
                 'sq_7': pos7_sq, 'sq_4': pos_sq,
                 }
    return positions

def init_cells():
    cells = {}
    cells['12_16'] = kagome.create_lattice(16,edges_16)
    cells['12_12'] = kagome.create_lattice(12,edges_12)
    cells['5_5'] = kagome.create_lattice(5,edges_5)
    cells['4_7'] = kagome.create_lattice(7,edges_4)
    cells['4_4'] = kagome.create_lattice(4,edges_4)
    cells['sq_4'] = kagome.create_lattice(4,edges_sq)
    cells['sq_7']  = kagome.create_lattice(7,edges_sq)
    cells['3_7'] = kagome.create_lattice(7,edges_3)
    cells['3_3'] = kagome.create_lattice(3,edges_3)
    return cells


# Create the addtional terms for the Case 1 Hamiltonian
from qiskit.opflow import X, Y, Z, I
# H1_1_4 = 1.0*(I^I^Z^Z) + 1.0*(I^I^Y^Y) + 1.0*(I^I^X^X)

def BoundaryCondition(qbits,K,num_qubits):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.opflow.primitive_ops import PauliSumOp
    # Start with K = I
    coefs = [1.0, 1.0, 1.0]
    op = SparsePauliOp.from_sparse_list([("XX", qbits, coefs[0]),
                                         ("YY", qbits, coefs[1]),
                                         ("ZZ", qbits, coefs[2])] ,num_qubits=num_qubits)
    H = PauliSumOp(op)
    return H

def init_hamiltonians(cells=None, force=False):
    hams = {}
    hams['12_12']  = kagome.get_hamiltonian(cells['12_12'])
    hams['12_16']  = kagome.get_hamiltonian(cells['12_16'])
    hams['4_7']    = kagome.get_hamiltonian(cells['4_7'])
    hams['4_4']    = kagome.get_hamiltonian(cells['4_4'])
    hams['3_3']    = kagome.get_hamiltonian(cells['3_3'])
    hams['sq_4']   = kagome.get_hamiltonian(cells['sq_4'])
    hams['sq_7']   = kagome.get_hamiltonian(cells['sq_7'])

    # Case 1  K = I
    H4_4_BCS_C1 = BoundaryCondition([0,1],I,4) + BoundaryCondition([0,3],I,4)
    hams['4_4_BC1']  =  hams['4_4'] + H4_4_BCS_C1
    cells['4_4_BC1'] = cells['4_4']

    H4_7_BCS_C1 = BoundaryCondition([0,1],I,7) + BoundaryCondition([3,0],I,7)
    hams['4_7_BC1'] = hams['4_7'] + H4_7_BCS_C1
    cells['4_7_BC1'] = cells['4_4']


    eigenvalue_results = {} if force else kagome.load_object('eigenvalues.dump')
    eigenvalue_results=kagome.compute_eigenvalues(hams,64,force=force,prev_results=eigenvalue_results)
    targets = kagome.list_eigenvalues(eigenvalue_results,cells)
    kagome.save_object(eigenvalue_results,'eigenvalues.dump')

    return hams, eigenvalue_results, targets






