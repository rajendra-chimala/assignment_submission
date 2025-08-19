import numpy as np


def Matrix_Multiplication (a,b):
    return a@b

def E_wise_multiplication(a,b):
    return a*b

def determinant(a):
    det = np.linalg.det(a)
    return det
    
def transpose(a):
    transposed = a.T
    return transposed



def main():
    
    a = np.array([[2,3],[1,4]])
    b = np.array([[5,2],[3,1]])
    
    
    print(f"Matrix Multiplication : ",Matrix_Multiplication(a,b))
    print(f"Element Wise Matrix Multiplication : ",E_wise_multiplication(a,b))
    print(f"Determinant : ",determinant(a))
    print(f"Transposed : ",transpose(a))
    
    
main()
