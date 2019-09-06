import numpy as np
def Rank(array):
    '''
    Input: array:array of fitness
    Output: ranks: a array of rank of fitness
    '''
    rank=np.zeros(array.shape,int)#random rank
    # sorted_array=np.sort(array)
    rank[array.argsort()]=np.arange(len(array))
    rank=(rank.astype(float)/(len(array)-1))-0.5

    return rank

if __name__ == "__main__":
    array=[2,3,14.7,98,8]
    print(type(np.array(array)))
    print(Rank(np.array(array)))
