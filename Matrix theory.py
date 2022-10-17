import numpy as np

np.set_printoptions(suppress=True)#Prohibition of Scientific notation

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



if __name__ == "__main__":
    A = np.array([[1,2,3],[2,1,2],[1,2,1]],dtype=float) #Input by column vector
    Q,R ,L= QR_decomposition(A)

    

