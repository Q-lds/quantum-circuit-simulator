# Quantum circuit simulator

This is very rudimental classical simulator for an universal quantum computer.

There are two ways to execute the quantum circuit.

This project was (partially) inspired by https://github.com/adamisntdead/QuSimPy 

### Sequential model

In the sequential gates gates update the input state, sequentially.

As a simple getting-start example, consider creating the Bell state $|Phi^+> = (H1 \otimes Cnot(1,2))|00>$

```python
from qcircuit.quantumcircuit import QuantumCircuit
from qcircuit.quantumregister import QuantumRegister
from qcircuit.classicalregister import ClassicalRegister

qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(2)) # Create the register
qc.setName('|Phi^+>')
qc.quantumRegister.applySingleGate('H', 1) # Apply H on qubit 1
qc.quantumRegister.applyCnotGate(1,2) # Apply a Cnot between qubit 1 and qubit 2
qc.readOut('classical') # Perform the measurements, and store the values in the classical register
```

we can also print the amplitudes as follows

```
print('Amplitudeds of |Phi^+> are ', qc.quantumRegister.formatAmplitudes()) # Read out the amplitudes

```

yielding;

`Amplitudeds of |Phi^+> are  [array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])]`



## The all-gates-at-once model

In the sequential model we update the amplitudes for each gate. That is, we perform the input-gate multiplication as many times as the total number of gates provided. When many gates are provided, and for large quantum register the approach is both cumbersome (to write the circuit) and inefficient (we have to perform many matrix multiplications).



In the all-gates-at-once model, we start by  specifying the list of gates (and the system upon which they are applied). Each gate is generated as a sparse matrix. The sparse representation of the gates are the multiplied into a new sparse matrix, which represent the whole circuit. This gate is then applied to then densed and applied to the input state.

Let's look at the $|\Phi^+>$ example from above

```python
bell_gates = [('H', 1), ('Cnot', (1,2))]
```

and then

```python
    qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(2))
    qc.quantumRegister.applyMultipleGatesAtOnce(qc.classicalRegister, gate_instructions=gateSet)
```

The resulting amplitudes are the same as before, that is

``Amplitudeds of |Phi^+> are [array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])]`



## Classical control

We can also use the classical register to act as a conditional (IF) structure

```python
gateSet = [('classicalControl', (1,1), 'X')]

qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(2))
qc.classicalRegister.flip(1) # Flip the second bit
qc.quantumRegister.applyMultipleGatesAtOnce(qc.classicalRegister, gateSet) # Give directly the list of tuples

#### Assertion ####
assert qc.quantumRegister.amplitudes.all() == np.complex128([(0+0j), (1+0j), (0+0j), (0+0j)]).all()
```





## To do

* Classical control does not (currently) control the cnot gate
* Sparse matrices (in scipy.sparse ) allow for a sparse implementation of the kron function. Now, in the current implementation there are two obvious overheads: 
  * the gates are generated, and then turned into sparse. One could start working with sparse matrices from the start
  * having a sparse representation of the input