import numpy as np
class A(object):
    def __init__(self):
        pass

    @staticmethod
    def rank(array):
        '''return an array of rank of fitness\nhave tested'''
        # assert array.shape[0]==1,'Shape setting error'
        rank=np.ones(array.shape,int)# rank as [1,1,1,1,1]

        rank[array.argsort()]=np.arange(len(array))
        rank=(rank.astype(float)/(len(array)-1))-0.5    
        return rank

if __name__ == "__main__":
    b=[1,5,3,7,7,8,3]
    array=np.array(b)
    a=A
    print(A.rank(array))
