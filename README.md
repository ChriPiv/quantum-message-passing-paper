# Quantum message-passing algorithm for optimal and efficient decoding

This repository contains all the code to reproduce the numerical experiments and figures for the paper _Quantum message-passing algorithm for optimal and efficient decoding_ by Christophe Piveteau and Joe Renes ([https://arxiv.org/abs/2109.08170](https://arxiv.org/abs/2109.08170)). 


Setup
=====
To reproduce the experiments, you require following packages:

* numpy
* matplotlib
* networkx
* qiskit
* cvxpy

In order to ensure there are no compatibility problems, I suggest to use the exact same version of these libraries as used during the deveoplment of the code.
This can be easily achieved with a virtual environment:

```bash
python3.8 -m venv <path to environment>
source <path to environment>/bin/activate
pip install numpy==1.19.5 matplotlib==3.3.3 networkx==2.5 qiskit==0.23.2
pip install cvxpy
```

Note that the used version of qiskit did not support python versions >= 3.9 yet.

Overview of code
================
Each subdirectory contains the files required to reproduce one of the figures in the paper. Each subdirectory contains a file `gen_data.py`, which generates the data to be plotted, as well as a file `plot_data.py` which generates the actual plot with matplotlib.

The python files in the main directory contain code to generate a BPQM decoding quantum circuit given a specified binary linear code.
The qiskit library is used to handle the generation and simulation of quantum circuits.
Following sample code shows how these files can be used to generate and simulate BPQM for a given code.

```python
from decoders import *
from matplotlib import pyplot as plt 

# define the linear code by providing a generator matrix G and parity check matrix H
# In this example we use the 8-bit code which is also investigated in section 6 in the paper.
G = np.array([[1, 0, 0, 0, 1, 0, 0, 1], 
              [0, 1, 0, 0, 1, 1, 0, 0], 
              [0, 0, 1, 0, 0, 1, 1, 0], 
              [0, 0, 0, 1, 0, 0, 1, 1]])
H = np.array([[1, 1, 0, 0, 1, 0, 0, 0], 
              [0, 1, 1, 0, 0, 1, 0, 0], 
              [0, 0, 1, 1, 0, 0, 1, 0], 
              [1, 0, 0, 1, 0, 0, 0, 1]])
code = LinearCode(G, H)


# We can draw the factor graph of our code for debugging purposes
nx.draw(code.get_factor_graph(), with_labels=True)
plt.show()

# specify channel parameter
theta = 0.2 * np.pi
# specify cloner to be used
cloner = VarNodeCloner(theta) # ENU cloner

# run BPQM to decode the fifth bit with an unrolling depth of 2
# (codeword bits are zero-indexed)
p_bit = decode_bpqm(code, theta, cloner=VarNodeCloner(theta), height=2, mode='bit', bit=4, only_zero_codeword=True, debug=False)
print("The probability of successfully decoding the first codeword bit is", p_bit)

# run BPQM to decode complete codeword with an unrolling depth of 2
p_codeword = decode_bpqm(code, theta, cloner=VarNodeCloner(theta), height=2, mode='codeword', order=[0,1,2,3], only_zero_codeword=True, debug=False)
print("The probability of successfully decoding the complete codeword is", p_codeword)
```

More detailed documentation about the `decode_bpqm` function can be found in the file `decoders.py`.
If you are interested in taking a look at the computational graphs and quantum circuits involved in the decoding, you can set the `debug` flag to `True` or have a look at the source of of the `decode_bpqm` function itself.

The file `decoders.py` also contains additional helper functions to evaluate the best achievable decoding performance for classical and quantum decoders for bitwise and codeword-wise decoding.

Specifying an approximate cloner is necessary in the case that the graph contains loops.
To see available cloners, see the file `cloner.py` (`VarNodeCloner` corresponds to the ENU cloner in the paper).
