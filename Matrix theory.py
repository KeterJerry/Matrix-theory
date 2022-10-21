import numpy as np
import sympy as sp
import fractions
import scipy


np.set_printoptions(suppress=True)#Prohibition of Scientific notation
np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})#Limit the output result to a fraction

def Unit_V(V):
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        Q = np.zeros_like(V)
        c = 0
        for a in V.T:
            u = np.copy(a)
            u = u/np.linalg.norm(u)
            Q[:,c] = u
            c += 1
        return Q


def Gram_Schmidt_Orthogonalization(V):
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        Q = np.zeros_like(V)
        
        c = 0
        for a in V.T:
            u = np.copy(a)
            #print (u)
            for i in range(0,c):
                u -= (np.dot(Q[:,i].T,a)/np.dot(Q[:,i],Q[:,i]))*Q[:,i]
            Q[:,c] = u
            c += 1
    
    return Q


def QR_decomposition(V):
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        if(np.linalg.matrix_rank(V) ==  V.shape[1]): #V.shape[0]and[1]means row and column
            #True or false of L indicates whether QR decomposition is unique
            L = True
        else:
            L = False  
                       
        U = Gram_Schmidt_Orthogonalization(V)
        Q = Unit_V(U)
        R = np.zeros_like(V)
        c = 0
        for a in U.T:
            R[c,c] = np.linalg.norm(a)
            #print(R[c,c])
            c += 1
        c = 1
        
        for a in V.T:
            
            for i in range (0,c-1):
                #print(i,c-1)
                #print(np.dot(a,Q[:,i]))
                R[i,c-1] = np.dot(a,Q[:,i])
            c += 1
  
    return Q,R,L


def np_RREF(V):
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        C = sp.Matrix(np.copy(V))
        R = np.array(C.rref()[0].tolist())
        
    
    return R


def Full_rank_decomposition(V):
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        V_rref = np_RREF(np.copy(V))
       
        
        V_index = np.array(np.where(V_rref == 1)).T
        
        
        x_axis = V_index[:,0]
        
        element,repeat_index=[],[]
        for i in range(0,len(x_axis)):
            if x_axis[i] in element:
                repeat_index.append(i)
            else:
                element.append(x_axis[i])
        V_index = np.delete(V_index,repeat_index,axis=0)

        
        B = V.T[V_index[:,1]].T
        C = V_rref[V_index[:,0]]

    return B,C


def LU_decomposition(V):#This function is currently unavailable
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        if V.shape[0] != V.shape[1]:
            raise ValueError
        else:
            n = V.shape[0]
            lu ,piv = scipy.linalg.lu_factor(np.copy(V), overwrite_a=False, check_finite=True)
            #print(lu)
            L = np.tril(lu,k=-1) + np.eye(n)
            U = np.triu(lu)
            



    return L,U,piv





if __name__ == "__main__":
    A = np.array([[2,5,8,7],[5,2,2,8],[7,5,6,6],[5,4,4,8]]) #Input by column vector
    #np.allclose()
    
    B,C,piv=LU_decomposition(A)

    print(np.matmul(B,C)[piv])
    print(piv@B@C)
    
    

