import numpy as np
import pandas as pd


class QuantumCircuit(object):
    def __init__(self, quantumRegister, classicalRegister):
        self.quantumRegister = quantumRegister
        self.classicalRegister = classicalRegister
        self.circuit_name = 'genericCircuit'

    def setName(self, circuit_name):
        self.circuit_name = circuit_name

    def readOut(self, outcome):
        """
        Performing a classical or a quantum measurement.

        Classical: the quantum state is measured and the outcomes are stored in the classical register
        Quantum: returns the amplitude of the quantum register

        Args:
            outcome: a string indicating the readout tipe (i.e. classical or quantum)

        """
        if outcome == 'quantum':
            return self.quantumRegister.amplitudes
        elif outcome == 'classical':
            for i in range(self.quantumRegister.n_qubits):
                # Measure the state
                outcomes = self.quantumRegister.measure()
                # Set the classical register to match with the outcome
                self.classicalRegister.setValues(outcomes)
        else:
            raise Exception("Outcome can only be either \"quantum\" or \"classical\",  ('%s' was provided)" % (outcome))

    def generateStatistics(self, n_measurements):
        """
        This returns a np.array containing the repteder measurements outcomes of the quantum register.

        The expensive part of simulating the quantum circuit is to actually obtain the final state. Once it is obtained, this function allows to obtain a statistics over the given final state.

        Args:
            n_measurements: an int specifying the number of measurements to implement

        Returns:
            a np.array containing the outcomes of the mesurements
        """
        stats = []
        for i in range(n_measurements):
            self.readOut('classical')
            stats.append(''.join([str(x) for x in self.classicalRegister.formatValues()]))

        return np.array(stats)

    def plotResults(self, stats):
        """
        Simple helper to plot the the measured statistics

        Args: 
            stats: an np.array() obtained from QuantumCircuit.generateStatistics containing the measurement statistics
        """
        pd.DataFrame(stats).rename(columns={0: 'outcome'}).outcome.value_counts().plot(kind='bar',
                                                                                       title='Ourcomes for circuit {} '.format(
                                                                                           self.circuit_name))
