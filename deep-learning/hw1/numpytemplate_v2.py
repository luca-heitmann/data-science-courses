import numpy as np

#define a random generator with a seed here using numpy, using the newer convention for numpy https://numpy.org/doc/stable/reference/random/generator.html
rng = np.random.default_rng(seed=10)


##################################################
# part 1 the easy stuff, indexing, searching
##################################################

a = np.arange(36).reshape((6, 6))
print("a=\n", a)

def index1(mat: np.ndarray):
    #Return row indexed with 5 (and columns 0,1)
    return mat[5, 0:2]

res = index1(a)
print("\n\nindex1 (row indexed with 5 (and columns 0,1))=\n", res)

def index2(mat: np.ndarray):
    #Return column indexed with 2 (and rows 0,1)
    return mat[0:2, 2]

res = index2(a)
print("\n\nindex2 (column indexed with 2 (and rows 0,1))=\n", res)

def index3(mat: np.ndarray):
    #Return the columns indexed with 1 and with 3 (and rows 0,1)
    return mat[0:2, 1:4:2]

res = index3(a)
print("\n\nindex3 (columns indexed with 1 and with 3 (and rows 0,1))=\n", res)

def index4(mat: np.ndarray):
    #Return the rows indexed with 0 to 2 (and columns 0,1). It should be 3 rows here
    return mat[0:3, 0:2]

res = index4(a)
print("\n\nindex4 (rows indexed with 0 to 2 (and columns 0,1). It should be 3 rows here)=\n", res)

def index5(mat: np.ndarray):
    # return the last row using a negative index (and columns 0,1)
    return mat[-1, 0:2]

res = index5(a)
print("\n\nindex5 (last row using a negative index (and columns 0,1))=\n", res)

def index6(mat: np.ndarray): 
    # return the third last row using a negative index (and columns 0,1)
    return mat[-3, 0:2]

res = index6(a)
print("\n\nindex6 (third last row using a negative index (and columns 0,1))=\n", res)

def index7(mat: np.ndarray):
    # return the the last two rows using a negative start index (and columns 0,1)
    return mat[-2:, 0:2]

res = index7(a)
print("\n\nindex7 (last two rows using a negative start index (and columns 0,1))=\n", res)

def index8(mat: np.ndarray):
    # return the the last three columns using a negative start index (and rows 0,1)
    return mat[0:2, -3:]

res = index8(a)
print("\n\nindex8 (last three columns using a negative start index (and rows 0,1))=\n", res)

def index9(mat: np.ndarray):
    # return the columns indexed with 3,4 using a negative start index and a negative stop index (and rows 0,1)
    return mat[0:2, -3:-1]

res = index9(a)
print("\nindex9 (columns indexed with 3,4 using a negative start index and a negative stop index (and rows 0,1))=\n", res)

def index10(mat: np.ndarray):
    # return the columns indexed with 0,1,2 using a negative start index and a negative stop index (and rows 0,1)
    return mat[0:2, -6:-3]

res = index10(a)
print("\nindex10 (columns indexed with 0,1,2 using a negative start index and a negative stop index (and rows 0,1))=\n", res)

def index11(mat: np.ndarray):
    # return every 2nd column, starting at index 0  (and rows 0,1)
    return mat[0:2, 0:7:2]

res = index11(a)
print("\nindex11 (every 2nd column, starting at index 0  (and rows 0,1))=\n", res)

def index12(mat: np.ndarray):
    # return every 3rd column, starting at index 1 (and rows 0,1)
    return mat[0:2, 1:7:3]

res = index12(a)
print("\nindex12 (every 3rd column, starting at index 1 (and rows 0,1))=\n", res)

def index13(mat: np.ndarray):
    # return every 2nd column, starting at the last index, in reversed order  (and rows 0,1)
    return mat[0:2, ::-2]

res = index13(a)
print("\nindex13 (every 2nd column, starting at the last index, in reversed order  (and rows 0,1))=\n", res)

def index14(mat: np.ndarray):
    # return every 2nd column, starting at the second last index, in reversed order (and rows 0,1) 
    return mat[0:2, -2::-2]
 
res = index14(a)
print("\nindex14 (every 2nd column, starting at the second last index, in reversed order (and rows 0,1))=\n", res)

b = np.arange(0,22)
print("\n\nb=\n", b)

def index15(mat: np.ndarray):
    # return every 3rd element, between index 3 and 12 (12 not included)
    return mat[3:13:3]

res = index15(b)
print("\nindex15 (every 3rd element, between index 3 and 12 (12 not included))=\n", res)
      
a = rng.integers(low=2,high=15,size=(5,5))
print("\n\na=\n", a)

def indexr1(mat: np.ndarray):
    #Return a matrix which is true where mat-values are higher than 5  (and false otherwise) 
    return mat > 5 #np.where(mat > 5, True, False)

res = indexr1(a)
print("\nindexr1 (matrix which is true where mat-values are higher than 5  (and false otherwise))=\n", res)

def indexr2(mat: np.ndarray):
    #Return the indices of the matrix where mat-values are higher than 5   
    return np.where(mat > 5)

res = indexr2(a)
print("\nindexr2 (indices of the matrix where mat-values are higher than 5)=\n", res)

def indexr3(mat: np.ndarray):
    #Return the values where mat-values are higher than 5 
    #will the result be a matrix?  
    return mat[mat > 5]

res = indexr3(a)
print("\nindexr3 (values where mat-values are higher than 5)=\n", res)

def indexr4(mat: np.ndarray):
    #Return the values where mat-values are higher than 5 and lower than 10
    #will the result be a matrix?  
    return mat[(mat > 5) & (mat < 10)]

res = indexr4(a)
print("\nindexr4 (values where mat-values are higher than 5 and lower than 10)=\n", res)

##################################################
# part 2 the easy stuff,  element-wise operations
##################################################

a = np.arange(9).reshape((3, 3))+1
print(a)

def op1(mat: np.ndarray):
    # return the sum over all elements    
    return np.sum(a)

res = op1(a)
print(res)

a = np.arange(36).reshape((3, 3, 4))      
print(a)

def op2(mat: np.ndarray):
    # return the sum over the axis #1 , indexing starts at 0
    return np.sum(mat, axis=0)

res = op2(a)
print(res)
     
a = np.arange(9).reshape((3, 3))

def op3(mat: np.ndarray):
    # return a scaled version so that it sums up to 1  
    return mat / np.sum(mat)

print(a)
res = op3(a)
print(res)
print(np.sum(res))

def op4(mat: np.ndarray):
    # square each entry element wise  
    return np.power(mat, 2)

res = op4(a)
print(res)    

a = np.arange(6).reshape((3, 2))
b = np.arange(6).reshape((3, 2)) -5  
def op5(mat1,mat2):
    # multiply both element-wise
    return mat1 * mat2

print(a,b)
res = op5(a,b)
print(res)

##################################################
# part 5 the intermediate stuff, inner products and the like
##################################################    

a = np.arange(6).reshape((3, 2))
b = np.arange(3) -5 

# 0 1
# 2 3
# 4 5

# -5 -4 -3

def op6(mat1, v):
    # compute the inner product between v and each vector in mat1 which is defined by fixing index in axis #1 and cycling through elements in axis #0
    result = np.zeros(mat1.shape[1])

    for i in range(mat1.shape[1]):
        result[i] = np.inner(v, mat1[:,i])

    return result

res = op6(a, b)
print(res)

a = np.arange(12).reshape((4, 3))
b = np.arange(3) -5 

def op7(mat1,v):
    # compute the inner product between v and each vector in mat1 which is defined by fixing index in axis #0 and cycling through elements in axis #1
    result = np.zeros(mat1.shape[0])

    for i in range(mat1.shape[0]):
        result[i] = np.inner(v, mat1[i])

    return result

res = op7(a,b)
print(res)        
  
a = np.arange(6).reshape((3, 2))
b = np.arange(3) -5
def op8(mat1,v):
    # matrix multiply v from the left to mat1
    # answer: is this equal to op6 or op7 ? => op6
    return np.matmul(v, mat1)

res = op8(a,b)
print(res)

a = np.arange(72).reshape((4, 3, 3, 2))
b = np.arange(3) -5 

def op9(mat1,v):
    # compute the inner product between v and each vector in mat1 which is defined by fixing index in axis #0 and #1 and #3 and cycling through elements in axis #2
    result = np.zeros((mat1.shape[0], mat1.shape[1], mat1.shape[3]))

    for i0 in range(mat1.shape[0]):
        for i1 in range(mat1.shape[1]):
            for i3 in range(mat1.shape[3]):
                result[i0, i1, i3] = np.inner(v, mat1[i0, i1, :, i3])

    return result

res = op9(a,b)
print(res)  
     
