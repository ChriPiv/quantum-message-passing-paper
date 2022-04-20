import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import *
from qiskit.providers.aer import *
from qiskit.providers.aer.extensions.snapshot_probabilities import *
import cvxpy as cp
import time

from bpqm import *
from cloner import *
from linearcode import *
from cvxpy_partial_trace import *

"""
Helper function to compute tensor products
exprs: list of matrices
"""
def TP(exprs):
    out = exprs[0]
    for i in range(1, len(exprs)):
        out = np.kron(out, exprs[i])
    return out

"""
Decodes a single
code: `LinearCode` object of desired code
cloner: `Cloner` object describing desired cloning strategy
cloner: doesn't matter which one you take if you have a tree graph
mode: 'bit' or 'codeword'
if mode=='bit' must specify integer bit, e.g. 0 
if mode=='codeword' must specify list of integers order, e.g. [2, 0, 3]
only_zero_codeword: if this is false, every input codeword is computed and the average probability is returned
debug: if this is true, debug messages will be printed
"""
def decode_bpqm(code, theta, cloner, height, mode, bit=None, order=None, only_zero_codeword=True, debug=False):
    # validate input
    assert mode in ['bit', 'codeword']
    if mode == 'bit':
        assert bit is not None
        assert order is None
        order = [bit]
    if mode == 'codeword':
        assert bit is None
        assert order is not None
        assert len(order) == code.k

    # determine the size of the quantum circuit
    # (this depends on how many qubits are to be cloned)
    # For this we have to generate the computational graph for each bit to be decoded
    cgraphs = [code.get_computation_graph("x"+str(b), height) for b in order]
    # cgraph is a list of tuples (graph, occurances, root)
    if debug:
        for i in range(len(order)):
            print("bit", order[i], ": Computational graph occurances:", cgraphs[i][1])
            print("plotting computational graph around bit ", order[i], "...")
            nx.draw(cgraphs[i][0], with_labels=True)
            plt.show()
    # also account for ancilla registers required to store the result
    n_data_qubits = max(sum(cg[1].values()) for cg in cgraphs)
    if n_data_qubits < code.n: n_data_qubits = code.n
    n_qubits = n_data_qubits + (len(order)-1)

    # generate quantum circuit
    qc = QuantumCircuit(n_qubits)
    qubit_mappings = list()
    for i in range(len(order)):
        b = order[i]
        graph, occurances, root = cgraphs[i]
        if debug: print("Generating circuit for the bit", b, "...")

        # generate qubit mapping (i.e. a dict output node label -> qubit index in circuit)
        # we choose the order like this: x0_0, x1_0, x2_0, ..., xn_0, x0_1, x0_2, ...
        qubit_mapping = dict()
        out_leaves = [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]
        out_leaves = sorted(out_leaves, key = lambda s : int(s.split("_")[0][1:]) + 1000*int(s.split("_")[1]) ) # ugly hack
        for j in range(code.n): qubit_mapping["y{}_0".format(j)] = j
        count = code.n
        for l in out_leaves:
            if int(l.split("_")[1])==0: continue
            qubit_mapping[l] = count
            count += 1
        if debug:
            print("Occurances: ", occurances)
            print("Qubit mapping: ", qubit_mapping)

        # We must mark some metadata into the graph which bpqm needs,
        # namely about the angles of the individual qubits and the qubit indices
        # in the cirucit
        cloner.mark_angles(graph, occurances)
        for l in out_leaves:
            graph.nodes[l]["qubit_idx"] = qubit_mapping[l]
    
        # generate BPQM circuit
        qc_bpqm = QuantumCircuit(n_qubits)
        idx, angles = tree_bpqm(graph, qc_bpqm, root=root)
        # generate cloner circuit
        qc_cloner = cloner.generate_cloner_circuit(graph, occurances, qubit_mapping, n_qubits)
        # append the circuit
        qc += qc_cloner
        qc.barrier()
        qc += qc_bpqm
        qc.barrier()
        if i != len(order)-1:
            # if this is not the last bit to decode: store the result in an ancilla
            qc.h(idx)
            qc.cx(idx, n_data_qubits+i)
            qc.h(idx)
            qc.barrier()
            qc += qc_bpqm.inverse()
            qc.barrier()
            qc += qc_cloner.inverse()
            qc.barrier()
        else:
            # if it is the last qubit: leave the data in place
            qc.h(idx)
    # qubits containing codeword
    codeword_qubits = list(range(n_data_qubits, n_data_qubits+len(order)-1)) + [idx]
    qc.snapshot_probabilities('prob', qubits=codeword_qubits)
    if debug: print(qc)

    # simulate circuit on desired channel outputs
    if only_zero_codeword: codewords = [[0]*code.n]
    else: codewords = code.get_codewords()
    backend = QasmSimulator(method='statevector')
    prob = 0. # probability of success
    for c in codewords:
        c = [int(x) for x in c]
        # generate initialization circuit
        qc_init = QuantumCircuit(n_qubits)
        state_plus  = np.array([np.cos(0.5*theta),  np.sin(0.5*theta)])
        state_minus = np.array([np.cos(0.5*theta), -np.sin(0.5*theta)])
        for j in range(min(code.n, n_data_qubits)):
            if c[j] == 0: qc_init.initialize(state_plus, [j])
            if c[j] == 1: qc_init.initialize(state_minus, [j])
        qc_tot = qc_init + qc

        # run simulation
        job = execute(qc_tot, backend=backend)
        vals = job.result().data()['snapshots']['probabilities']['prob'][0]['value']

        # get hexadecimal representation of desired output
        key = hex(int("".join(list(reversed([str(int(c[i])) for i in order]))), 2))
        if debug: print("Success probability", vals[key], "for codeword", c)

        prob += vals[key] / len(codewords)
    return prob


"""
Optimally decodes a single bit
index: index of bit to be decoded
"""
def decode_bit_optimal_quantum(code, theta, index):
    rho_0 = np.zeros((2**code.n,2**code.n), dtype=np.complex128)
    rho_1 = np.zeros((2**code.n,2**code.n), dtype=np.complex128)
    codewords = code.get_codewords()
    vecs = [ np.array([[np.cos(0.5*theta)], [ np.sin(0.5*theta)]]),
             np.array([[np.cos(0.5*theta)], [-np.sin(0.5*theta)]]) ]

    for c in [cw for cw in codewords if int(cw[index])==0]:
        psi = TP([ vecs[int(k)] for k in c ])
        rho_0 += psi@psi.T / (0.5*len(codewords))
    for c in [cw for cw in codewords if int(cw[index])==1]:
        psi = TP([ vecs[int(k)] for k in c ])
        rho_1 += psi@psi.T / (0.5*len(codewords))
    eig = np.linalg.eigvals(rho_0-rho_1)
    return 0.5 + 0.25*np.sum(np.abs(eig))

def decode_codeword_PGM(code, theta):
    codewords = code.get_codewords()
    vecs = [ np.array([[np.cos(0.5*theta)], [ np.sin(0.5*theta)]]),
             np.array([[np.cos(0.5*theta)], [-np.sin(0.5*theta)]]) ]
    
    # build density matrix of state at hand
    rho = np.zeros((2**code.n, 2**code.n), dtype=np.complex128)
    for c in codewords:
        psi = TP([ vecs[int(i)] for i in c])
        rho += (psi@psi.T) / len(codewords)
    eigvals, eigvecs = np.linalg.eig(rho)
    # do the square root inverse on the support of the matrix
    idx = np.abs(eigvals)>1e-8
    eigvals[idx] = eigvals[idx]**(-0.5)
    rho_sqrt_inv = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)

    probab = 0.
    for c in codewords:
        vec = TP([ vecs[int(i)] for i in c])
        # build PVM basis element of the PGM
        basis_vec = rho_sqrt_inv @ vec / np.sqrt(len(codewords))
        probab += np.abs( np.sum(vec * basis_vec) )**2 / len(codewords)
    return probab


"""
solves distinguishability SDP
"""
def decode_codeword_optimal_quantum(code, theta):
    vecs = [ np.array([[np.cos(0.5*theta)], [ np.sin(0.5*theta)]]),
             np.array([[np.cos(0.5*theta)], [-np.sin(0.5*theta)]]) ]
    vecs_c = [ np.array([[1.],[0.]]), np.array([[0.],[1.]]) ]

    sigma = cp.Variable((2**(code.n),2**(code.n)),PSD=True)
    objective = cp.trace(sigma)
    codewords = code.get_codewords()
    constraints = []
    for c in codewords:
        psi = TP([ vecs[int(i)] for i in c])
        constraints.append(sigma >> (psi@psi.T) / len(codewords))
    problem = cp.Problem(cp.Minimize(objective), constraints)

    problem.solve(solver="SCS")
    assert problem.status == 'optimal' 
    return objective.value


"""
success probability to decode a single bit with strategy which consists of
* performing the individual Helstrom measurement for each qubit
* performing ideal classical MAP decoder
index: index of bit to be decoded
"""
def decode_bit_optimal_classical(code, theta, index):
    if theta < 1e-8: return 0.5
    codewords = code.get_codewords()
    # single-bit helstrom probabilities
    p_right = 0.5*(1. + np.sin(theta))
    p_wrong = 0.5*(1. - np.sin(theta))

    prob = 0. # probability of success
    for m in range(2**code.n): # iterate over all possible channel outputs
        ch_out = (bin(m)[2:].zfill(code.n))
        ch_out = [float(c) for c in ch_out]

        # obtain bitwise MAP decoder output
        def probab(c): # Prob[Y=ch_out|X=c]
            num_diff = np.sum(np.abs( np.array(c) - np.array(ch_out)))
            return p_wrong**num_diff * p_right**(code.n-num_diff)
        P0 = np.sum([probab(c) for c in [cw for cw in codewords if int(cw[index])==0]])
        P1 = np.sum([probab(c) for c in [cw for cw in codewords if int(cw[index])==1]])
        if np.abs(P0 - P1) < 1e-8: continue
        if P0 > P1: map_out = 0.
        else: map_out = 1.

        # loop over all possible codewords and see how likely it was they
        # caused that channel output
        for ch_in in codewords:
            weight = probab(ch_in) / len(codewords)
            if np.abs(ch_in[index] - map_out) < 1e-8:
                prob += weight
    return prob

def decode_codeword_optimal_classical(code, theta):
    if theta < 1e-8: return 1. / (2**code.k)
    codewords = code.get_codewords()
    # single-bit helstrom probabilities
    p_right = 0.5*(1. + np.sin(theta))
    p_wrong = 0.5*(1. - np.sin(theta))

    prob = 0. # probability of success
    for m in range(2**code.n): # iterate over all possible channel outputs
        ch_out = (bin(m)[2:].zfill(code.n))
        ch_out = [float(c) for c in ch_out]

        # obtain bitwise MAP decoder output
        def probab(c): # Prob[Y=ch_out|X=c]
            num_diff = np.sum(np.abs( np.array(c) - np.array(ch_out)))
            return p_wrong**num_diff * p_right**(code.n-num_diff)
        
        map_out = sorted([np.array(c) for c in codewords], key=probab, reverse=True)[0]

        prob += probab(map_out) / len(codewords)
    return prob

