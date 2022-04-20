import numpy as np
from qiskit.extensions.unitary import *

"""
General remark: During the construction of the BPQM circuit, we have to keep
track of the angle of each qubit. That angle is classically conditioned on
certain other qubits, which happen to be in a classical state.

The angles are stored as list of tuples [(t1,c1), (t2,c2), ...] where ti
is the float value of the angle anc ci is the 'classical conditioning' for
that angle. For example, let's consider an angle that is 0.1 if qubit 0 has
value 0 and 0.2 if qubit 0 has value 1. That would be stored as
[(0.1, {0:0}), (0.2, {0:1})]
In general ci is a dictionary that contains key-value pairs where the key
is a qubit-index and the value is the value on which this qubit is conditioned.
"""

"""
Extends the qiskit.QuantumCircuit object qc by performing variable node
operation on the two qubits with indices idx1 and idx2. The angles of these
two qubits are given by angles1 and agles2.

The output qubit index and the output qubit angle are returned.
"""
def combine_variable(qc, idx1, angles1, idx2, angles2):
    idx_out = idx1
    angles_out = list()

    control_qubits = list()
    if len(angles1)>0: control_qubits += list(angles1[0][1].keys())
    if len(angles2)>0: control_qubits += list(angles2[0][1].keys())
    
    angles_alpha = [None] * (2**len(control_qubits))
    angles_beta = [None] * (2**len(control_qubits))
    for t1,c1 in angles1:
        for t2,c2 in angles2:
            controls = {**c1, **c2} # merge dictionaries
            angles_out.append((np.arccos(np.cos(t1)*np.cos(t2)), controls))

            # get index of location of angles list in which we have to store the angle
            if len(control_qubits)==0: idx = 0
            else:
                controls_sorted = [controls[i] for i in control_qubits]
                idx = int("".join([str(x) for x in controls_sorted]), 2)

            a_min = (np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2)) ) / (np.sqrt(2.)*np.sqrt(1. + np.cos(t1)*np.cos(t2)))
            b_min = (np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2)) ) / (np.sqrt(2.)*np.sqrt(1. - np.cos(t1)*np.cos(t2)))
            alpha = np.arccos(-a_min) + np.arccos(-b_min)
            beta  = np.arccos(-a_min) - np.arccos(-b_min)
            angles_alpha[idx] = alpha
            angles_beta[idx] = beta

    # apply variable node unitary
    qc.cx(idx2, idx1)
    qc.x(idx1)
    qc.cx(idx1, idx2)
    qc.x(idx1)
    control_qubits = list(reversed(control_qubits)) # qiskit uses the inverse qubit ordering that we do
    qc.ucry(angles_alpha, control_qubits, idx2)
    qc.cx(idx1, idx2)
    qc.ucry(angles_beta, control_qubits, idx2)
    qc.cx(idx1, idx2)

    return idx_out, angles_out

"""
Same as combine_variable, but for check node operation.
"""
def combine_check(qc, idx1, angles1, idx2, angles2):
    idx_out = idx1
    angles_out = list()

    qc.cx(idx1, idx2)
    for t1,c1 in angles1:
        for t2,c2 in angles2:
            controls = {**c1, **c2} # merge dictionaries

            # branch 1
            tout_0 = np.arccos( (np.cos(t1)+np.cos(t2)) / (1. + np.cos(t1)*np.cos(t2)) )
            cout_0 = controls.copy()
            cout_0[idx2] = 0
            # branch 2
            tout_1 = np.arccos( (np.cos(t1)-np.cos(t2)) / (1. - np.cos(t1)*np.cos(t2)) )
            cout_1 = controls.copy()
            cout_1[idx2] = 1

            angles_out.append((tout_0, cout_0))
            angles_out.append((tout_1, cout_1))

    return idx_out, angles_out

"""
Input:
* a networkx graph 'tree' that represents a factor graph. Each node of the graph
  must have the attribute 'type' that must be iether 'variable', 'check' or 'output'.
  Furthermore the output nodes require an attribute 'angle' and 'qubit_idx'. The latter
  specifies which qubit index is used in the quantum circuit.
* a qsikit.QuantumCircuit object 'qc' onto which the BPQM circuit operations are applied
* The root node 'root' of 'tree' which determines which bit of the code is to be determined by BPQM

Returns:
* Index of final qubit in circuit (onto which you want to perform the Helstrom measurement)
* Angles of output qubit (under the assumption that input of circuit respects the factor graph 'tree')
"""
def tree_bpqm(tree, qc, root):
    succs = list(tree.successors(root))
    num_succ = len(succs)

    if num_succ == 0:
        # root is a leaf
        assert tree.nodes[root]["type"] == "output"
        return tree.nodes[root]["qubit_idx"], tree.nodes[root]["angle"]

    if num_succ == 1:
        # do nothing
        return tree_bpqm(tree, qc, succs[0])

    # >= 2 descendents: combine them 2 at a time
    idx, angles = tree_bpqm(tree, qc, succs[0])
    for i in range(1, num_succ):
        idx2, angles2 = tree_bpqm(tree, qc, succs[i])
        type = tree.nodes[root]["type"] 
        if type == "variable":
            idx, angles = combine_variable(qc, idx, angles, idx2, angles2)
        elif type == "check":
            idx, angles = combine_check(qc, idx, angles, idx2, angles2)
        else: raise
    return idx, angles
