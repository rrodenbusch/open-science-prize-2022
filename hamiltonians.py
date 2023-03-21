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
import numpy as np

# Define edge lists for the lattices
t = 1    # All are weighted equally to start
edges_sq = [(0,1,t),(1,2,t),(2,3,t),(3,0,t)]

edges_3 = [ (0,1,t), (0,2,t), (2,1,t), ]

edges_4 = [ (0, 1, t), (0, 2, t), (2, 1, t),  (1,3,t) ]

edges_5    = [(0,1,t), (0,2,t), (2,1,t), (1,3,t), (1,4,t), (3,4,t), ]
edges_5_7  = [(0,1,t), (0,2,t), (1,2,t), (1,3,t), (1,5,t), (3,5,t) ]
edges_5_7m = [(0,1,t), (0,2,t), (1,2,t), (2,3,t), (2,4,t), (3,4,t) ]

edges_5_16 = [(0,1,t), (0,2,t), (1,2,t), (1,7,t), (1,4,t), (7,4,t) ]
# edges_7 = [(0,1,t), (0,2,t), (1,2,t), (1,7,t), (1,4,t), (7,4,t), (4,5,t), (4,6,t), ]
# edges_7 = [ (0, 1, t), (0, 2, t), (2, 1, t), (1, 3, t), (1, 4, t),  (3, 4, t),
#             (4, 5, t), (4, 6, t), (5,6,t) ]
edges_12_16 = [(1,2,t), (2,3,t), (3,9,t), (9,8,t), (8,11,t), (11,14,t),
               (14,13,t), (13,12,t), (12,10,t), (10,7,t), (7,4,t), (4,1,t),
               (4,2,t), (2,9,t), (9,11,t), (11,13,t), (13,10,t), (10,4,t), ]

edges_4_16  = [ (4,1,t),(4,2,t),(2,1,t),(2,3,t), ]
edges_42_16 = [ (9,8,t),(9,11,t),(11,8,t),(11,14,t), ]
edges_43_16 = [ (13,12,t),(13,10,t),(10,12,t),(10,7,t), ]

edges_5_16 = [ (4,1,t),(4,2,t),(2,1,t),(2,3,t),(2,5,t),(3,5,t) ]

def init_positions():
    node_pos = {}
    extras= [ [0.25, -1],   [0.5,-1],   [0.75,-1],   [1.0,-1],
              [1.25, -1],   [1.5,-1],   [1.75,-1],   [2.0,-1],
              [0.25, -0.5], [0.5,-0.5], [0.75,-0.5], [1.0,-0.5],
              [1.25, -0.5], [1.5,-0.5], [1.75,-0.5], [2.0,-0.5],
              ]
    base = {0:[1,-1], 6:[1.5,-1], 9:[2,-1], 15:[2.5,-1],
            1:[0,-0.8], 2:[-0.6,1], 4:[0.6,1], 10:[1.2,3],
           13:[0.6,5], 11:[-0.6,5], 5:[-1.2,3], 3:[-1.8,0.9],
            8:[-1.8,5.1], 14:[0,6.8], 7:[1.8,0.9], 12:[1.8,5.1]}
    #================== Kagome Lattice ====================================
    node_pos['12_16'] = {1:[0,-0.8],    2:[-0.6,1],  4:[0.6,1],   10:[1.2,3],
                         13:[0.6,5],    11:[-0.6,5], 9:[-1.2,3],   3:[-1.8,0.9],
                         8:[-1.8,5.1], 14:[0,6.8],   7:[1.8,0.9], 12:[1.8,5.1],
                         0:[-1.0,-1.8],   6:[-0.75,-1.8], 5:[-0.5,-1.8], 15:[-0.25,-1.8], }
    node_pos['12_12'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7],
                         4:base[4],  5:base[2],  6:base[1],  8:base[5],
                         7:base[3], 10:base[8],  9:base[11], 11:base[14]}
    #================== Triangle ==========================================
    node_pos['3_3'] = {0:base[13], 1:base[10], 2:base[12],}
    node_pos['3_5'] = {0:base[13], 1:base[10], 2:base[12], 3:extras[0],  4:extras[1], }
    node_pos['3_7'] = {0:base[13], 1:base[10], 2:base[12],
                       3:extras[0],  4:extras[1], 5:extras[2], 6:extras[3],}
    #=================  Square Lattice ===============================
    node_pos['sq_4'] = {0:base[13], 1:base[4], 2:base[2], 3:base[11], }
    node_pos['sq_5'] = {0:base[13], 1:base[4], 2:base[2], 3:base[11], 4:extras[0], }
    node_pos['sq_7'] = {0:base[13], 1:base[4], 2:base[2], 3:base[11],
                        4:extras[0], 5:extras[1], 6:extras[2],}
    #================= 1/3 of Kagome Lattice =====================
    node_pos['4_4'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7], }
    node_pos['4_5'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7],
                       4:extras[0], }
    node_pos['4_7'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7],
                       4:extras[0], 5:extras[1], 6:extras[2] }
    node_pos['4_16'] = {1:[0,-0.8],    2:[-0.6,1],  4:[0.6,1],   10:[1.2,3],
                         13:[0.6,5],    11:[-0.6,5],  9:[-1.2,3],   3:[-1.8,0.9],
                         8:[-1.8,5.1], 14:[0,6.8],   7:[1.8,0.9], 12:[1.8,5.1],
                         0:[-1.0,-1.8],   6:[-0.75,-1.8], 5:[-0.5,-1.8], 15:[-0.25,-1.8], }
    node_pos['42_16'] = {1:[0,-0.8],    2:[-0.6,1],  4:[0.6,1],   10:[1.2,3],
                         13:[0.6,5],    11:[-0.6,5],  9:[-1.2,3],   3:[-1.8,0.9],
                         8:[-1.8,5.1], 14:[0,6.8],   7:[1.8,0.9], 12:[1.8,5.1],
                         0:[-1.0,-1.8],   6:[-0.75,-1.8], 5:[-0.5,-1.8], 15:[-0.25,-1.8], }
    node_pos['43_16'] = {1:[0,-0.8],    2:[-0.6,1],  4:[0.6,1],   10:[1.2,3],
                         13:[0.6,5],    11:[-0.6,5],  9:[-1.2,3],   3:[-1.8,0.9],
                         8:[-1.8,5.1], 14:[0,6.8],   7:[1.8,0.9], 12:[1.8,5.1],
                         0:[-1.0,-1.8],   6:[-0.75,-1.8], 5:[-0.5,-1.8], 15:[-0.25,-1.8], }
    #==================   Double Triangle Node  ============================
    node_pos['5_5'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7],
                       4:base[4], }
    node_pos['5_7'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7], 5:base[4],
                       4:extras[0], 6:extras[1], }
    node_pos['5_7m'] = {0:base[13], 2:base[10], 1:base[12], 3:base[7], 4:base[4],
                        3:extras[0], 6:extras[1], }
    node_pos['5_12'] = {0:base[13], 1:base[10], 2:base[12], 3:base[7], 4:base[4],
                        5:extras[1],  6:extras[2], 8:extras[3],
                        7:extras[4], 10:extras[5], 9:extras[6], 11:extras[7], }
    node_pos['5_16'] = {1:[0,-0.8],    2:[-0.6,1],  4:[0.6,1],   10:[1.2,3],
                         13:[0.6,5],    11:[-0.6,5],  5:[-1.2,3],   3:[-1.8,0.9],
                         8:[-1.8,5.1], 14:[0,6.8],   7:[1.8,0.9], 12:[1.8,5.1],
                         0:[-1.0,-1.8],   6:[-0.75,-1.8], 9:[-0.5,-1.8], 15:[-0.25,-1.8], }

    return node_pos

def init_cells():
    cells = {}

    cells['3_3'] = kagome.create_lattice(3,edges_3)
    cells['3_5'] = kagome.create_lattice(5,edges_3)
    cells['3_7'] = kagome.create_lattice(7,edges_3)

    cells['sq_4'] = kagome.create_lattice(4,edges_sq)
    cells['sq_5'] = kagome.create_lattice(5,edges_sq)
    cells['sq_m'] = kagome.create_lattice(5,edges_sq) # manila has unique 5 qubit architecture
    cells['sq_7'] = kagome.create_lattice(7,edges_sq)

    cells['4_4']  = kagome.create_lattice(4,edges_4)
    cells['4_5']  = kagome.create_lattice(5,edges_4)
    cells['4_7']  = kagome.create_lattice(7,edges_4)
    cells['4_12'] = kagome.create_lattice(12,edges_4)
    
    cells['4_16']  = kagome.create_lattice(16,edges_4_16)
    cells['42_16']  = kagome.create_lattice(16,edges_42_16)
    cells['43_16']  = kagome.create_lattice(16,edges_43_16)

    cells['5_5']  = kagome.create_lattice(5, edges_5)
    cells['5_7']  = kagome.create_lattice(7, edges_5_7)
    cells['5_7m'] = kagome.create_lattice(7, edges_5_7m)
    # cells['5_12']  = kagome.create_lattice(12,edges_5)
    # cells['5_12a'] = kagome.create_lattice(12,edges_5)
    cells['5_16']  = kagome.create_lattice(16,edges_5_16)

    # cells['7_7']   = kagome.create_lattice(7, edges_7)

    cells['12_16'] = kagome.create_lattice(16,edges_12_16)

    return cells

def edge_hamiltonians(edges, num_qubits):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.opflow import X, Y, Z, I
    hams = []
    for (i,j,t) in edges:
        hams.append(SparsePauliOp.from_sparse_list([("XX", [i,j], t), ], num_qubits=num_qubits),)
        hams.append(SparsePauliOp.from_sparse_list([("YY", [i,j], t), ], num_qubits=num_qubits),)
        hams.append(SparsePauliOp.from_sparse_list([("ZZ", [i,j], t), ], num_qubits=num_qubits),)
    return hams

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

def init_hamiltonians(cells=None, force=False, k=64, display=None, fname='data/eigenvalues.dump'):
    hams = {}
    # hams['12_12']  = get_hamiltonian(cells['12_12'])
    hams['12_16']  = get_hamiltonian(cells['12_16'])

    ##### Triangles
    hams['3_3']    = get_hamiltonian(cells['3_3'])
    hams['3_5']    = get_hamiltonian(cells['3_5'])
    hams['3_7']    = get_hamiltonian(cells['3_7'])

    ##### Squares
    hams['sq_4']   = get_hamiltonian(cells['sq_4'])
    hams['sq_5']   = get_hamiltonian(cells['sq_5'])
    hams['sq_m']   = get_hamiltonian(cells['sq_m'])
    hams['sq_7']   = get_hamiltonian(cells['sq_7'])

    ##### Partial Lattices
    hams['4_4']    = get_hamiltonian(cells['4_4'])
    hams['4_5']    = get_hamiltonian(cells['4_5'])
    hams['4_7']    = get_hamiltonian(cells['4_7'])
    hams['4_12']   = get_hamiltonian(cells['4_12'])
    ##   4_16
    hams['4_16']   = get_hamiltonian(cells['4_16'])
    H4_16_BCS_C1 = BoundaryCondition([2,4],I,16) + BoundaryCondition([3,4],I,16)
    hams['4_16_BC1'] = hams['4_16'] + H4_16_BCS_C1
    cells['4_16_BC1'] = cells['4_16']    
    ## 42_16
    hams['42_16']   = get_hamiltonian(cells['42_16'])
    H41_16_BCS_C1 = BoundaryCondition([11,9],I,16) + BoundaryCondition([14,9],I,16)
    hams['42_16_BC1'] = hams['42_16'] + H41_16_BCS_C1
    cells['42_16_BC1'] = cells['42_16']    
    ## 43_16
    hams['43_16']   = get_hamiltonian(cells['43_16'])    
    H42_16_BCS_C1 = BoundaryCondition([7,13],I,16) + BoundaryCondition([10,13],I,16)
    hams['43_16_BC1'] = hams['43_16'] + H42_16_BCS_C1
    cells['43_16_BC1'] = cells['43_16']    

    hams['5_5']    = get_hamiltonian(cells['5_5'])
    hams['5_7']    = get_hamiltonian(cells['5_7'])
    hams['5_7m']    = get_hamiltonian(cells['5_7m'])
    hams['5_16']   = get_hamiltonian(cells['5_16'])
    # hams['5_12']   = get_hamiltonian(cells['5_12'])

    # hams['7_7']    = get_hamiltonian(cells['7_7'])
    hams['5_12a'] = BoundaryCondition([1,4],I, 12) + BoundaryCondition([3,4],I, 12)
    for curEdge in edges_4:
        hams['5_12a'] += BoundaryCondition([curEdge[0], curEdge[1]], I, 12)

    # Case 1  K = I
    H4_4_BCS_C1 = BoundaryCondition([0,1],I,4) + BoundaryCondition([0,3],I,4)
    hams['4_4_BC1']  =  hams['4_4'] + H4_4_BCS_C1
    cells['4_4_BC1'] = cells['4_4']

    H4_5_BCS_C1 = BoundaryCondition([0,1],I,5) + BoundaryCondition([0,3],I,5)
    hams['4_5_BC1']  =  hams['4_5'] + H4_5_BCS_C1
    cells['4_5_BC1'] = cells['4_5']

    H4_7_BCS_C1 = BoundaryCondition([0,1],I,7) + BoundaryCondition([3,0],I,7)
    hams['4_7_BC1'] = hams['4_7'] + H4_7_BCS_C1
    cells['4_7_BC1'] = cells['4_4']

    H4_12_BCS_C1 = BoundaryCondition([0,1],I,12) + BoundaryCondition([3,0],I,12)
    hams['4_12_BC1'] = hams['4_12'] + H4_12_BCS_C1
    cells['4_12_BC1'] = cells['4_12']


    eigenvalue_results = {} if force else kagome.load_object(fname)
    eigenvalue_results=compute_eigenvalues(hams,k=k,force=force,prev_results=eigenvalue_results)
    targets = list_eigenvalues(eigenvalue_results,cells,display=display)
    kagome.save_object(eigenvalue_results,fname)

    return hams, eigenvalue_results, targets

def get_hamiltonian(lattice,spin_interaction=1.0,global_potential=0.0):
    from heisenberg_model import HeisenbergModel
    from qiskit_nature.mappers.second_quantization import LogarithmicMapper
    heis = HeisenbergModel.uniform_parameters(lattice=lattice,
                                                 uniform_interaction=spin_interaction,
                                                 uniform_onsite_potential=global_potential,
                                                )
    H = 4 * LogarithmicMapper().map(heis.second_q_ops().simplify())
    return H

def compute_eigenvalues(hams,k=64,force=False, prev_results=None):
    from qiskit.algorithms import NumPyEigensolver
    if prev_results is None:
        prev_results = {}
    for key,H in hams.items():
        degen_list = ''
        curEigenvalues = prev_results.get(key,None)
        if force or (curEigenvalues is None):
            print(f"Computing eigenvalues for {key}")
            prev_results[key]  = NumPyEigensolver(k=k).compute_eigenvalues(H)
    return(prev_results)

def list_eigenvalues(eigData,cells,display=0):
    if not isinstance(eigData,dict):
        eigData = {'input':eigData}
    targets={}
    for key, result in eigData.items():
        targets[key] = result.eigenvalues[0]
        curCount = 0 if not isinstance(display,int) else display
        if curCount != 0:
            degen_list = ''
            unq_value, unq_counts = degeneracy(result.eigenvalues)
            for eig in unq_value:
                degen = unq_counts[eig]
                if curCount != 0:
                    curCount-=1
                    degen_list += f"\n\t{np.around(eig,4):8.4f}:[{degen}]"
            if cells.get(key,None) is not None:
                print(f"\nH{key}: Edges {len(cells[key].weighted_edge_list)} "
                      f"Eigenvalues {len(result.eigenvalues)} {degen_list}")
    return targets

def degeneracy(list1,precision=6):
    unique_list  = []
    unique_count = {}
    for x1 in list1:
        x = np.around(x1,precision)
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            unique_count[x] = 1
        else:
            unique_count[x] += 1
    return unique_list, unique_count

def SparsePauliPrint(pauli,label='Sparse Pauli'):
    cnt=0
    print(f"{label}{pauli.to_matrix().shape} as list:")
    for curItem in pauli.to_list():
        cnt+=1
        print(f"{cnt}:\t{curItem[0]} * {curItem[1]}")
    print("\n")


