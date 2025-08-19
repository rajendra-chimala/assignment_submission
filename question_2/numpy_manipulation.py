
import numpy as np

array = np.random.randint(10,101,(6,6))

print(f"Random array : {array}")

diagonal_elemtnts = np.diag(array)

print(f"Diagonal elements of an array : {diagonal_elemtnts}")


new_matrix = array.astype(float)

even_pos = (new_matrix %2 == 0)

new_matrix[even_pos] = np.sqrt(new_matrix[even_pos])

print(f"The Replaced Matrix with even number with squar root : {new_matrix}")

print(f"Mean value : {np.mean(new_matrix)}")
print(f"Median value : {np.median(new_matrix)}")
print(f"Standard Deviation value : {np.std(new_matrix)}")


reshaped_array = new_matrix.reshape(4,9)


print(f"REshaped Matris is : {reshaped_array}")

