import numpy as np

class ClassicalRegister(object):
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.values = np.zeros(n_bits, dtype=float)
        
        
    def flip(self, bit_index):
        """ 
        Flips the bit_index from zero to one
        """
        if self.values[bit_index] == 0:
            self.values[bit_index] = 1
        elif self.values[bit_index] == 1:
            self.values[bit_index] = 0
            
    
    def setValues(self, values):
        """
        Set the ClassicalRegister's values to those provided by the array
        """
        self.values = np.array(values)
            
                
    def formatValues(self):
        """
        An (almost pretty) priting utility
        """
        return [i.round(4) for i in self.values]