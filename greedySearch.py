import math
from typing import List
import numpy as np
import random

def findMinErr(A: List[int], target: int):
    val = 0
    temp = []

    for i in range(len(A) - 1, -1, -1):
        if A[i] > target:
            val = A[i]
        else:
            break

    return val

def findMaxElements(target: int, B: List[int]):
    '''The greedy rule is that in every a in A, we find aprroxiate b in B(a > b), and b is from
    big to small'''
    print('The greedy rule is to find max b in B')
    temp = []
    for i in range(len(B)-1, -1, -1):
        if target >= B[i]:
            temp.append(1)
            target -= B[i]
        else:
            temp.append(0)
    temp.reverse()
    return temp

def findMinElements(target: int, B: List[int]):
    '''The greedy rule is that in every a in A, we find aprroxiate b in B(a > b), and b is from
    small to big'''
    print('The greedy rule is to find in b in B')
    temp = []
    for i in range(len(B)):
        if target >= B[i]:
            temp.append(1)
            target -= B[i]
        else:
            temp.append(0)
    # temp.reverse()
    return temp

def findKnaosack(target: int, B: List[int]):
    '''The greedy rule is that in every a in A, we find the sum of b is max, i.d. sum(b) is max,
     so we apply Knapsack problem'''
    print('The greedy rule is Knapsack')
    weight = B
    value = B
    bag_weight = target

    # 初始化: 全为0
    dp = [0] * (bag_weight + 1)
    # 先遍历物品, 再遍历背包容量
    for i in range(len(weight)):
        for j in range(bag_weight, int(math.ceil(weight[i] - 1)), -1):
            # 递归公式
            dp[j] = max(dp[j], dp[int(math.floor(j - weight[i]))] + value[i])

    print(dp)

def tMatrix(A: List[int], B: List[int]):
    t_M = []
    for i in range(len(A)):
        # t_M.append(findMaxElements(A[i], B))
        t_M.append(findMinElements(A[i], B))
    t_M = np.array(t_M)
    print('*****The transit matix is : *****')
    print(t_M)
    print('**********')
    return t_M

def calCost(t_results):
    '''calculate the final cost of every plan'''
    m, n = t_results.shape
    cost = 0
    count = 0
    for  i in range(m):
        if t_results[i].any():
            cost += A[i]
            count += 1
        if count > n:
            break       ## find enough, early quit
    return cost

def greedySearch(A: List[int], B: List[int]):
    '''greedy search from left to right in B'''
    t_results = tMatrix(A, B)
    temp = np.array([0 for i in range(len(B))])

    ## greedy search, for every b in B, find min{a}, a>b & a in A
    ## and b is from small to big
    for j in range(len(B)):
        ## check complement
        if ~t_results[:, j].any():
            print(j, 'th elements in B don\'t reflect in A')
            return
        for i in range(len(A)):
            if t_results[i,j] == 1:
                t_results[i+1:,j] = 0
                temp |= t_results[i]
                break
        ## have already found all plans, quit
        if temp.all() == 1:
            break
    return t_results

def greedySearch2(A: List[int], B: List[int]):
    '''greedy search,  randomly select b in B'''
    t_results = tMatrix(A, B)
    temp = np.array([0 for i in range(len(B))])

    randIdx_i = [i for i in range(len(A))]
    random.shuffle(randIdx_i)
    randIdx_j = [j for j in range(len(B))]
    random.shuffle(randIdx_j)

    ## greedy search, for every b in B, find min{a}, a>b & a in A
    ## and randomly select b in B
    for j_ in range(len(B)):
        j = randIdx_j[j_]
        print('current j is : ', j)
        ## check complement
        if ~t_results[:, j].any():
            print(j, 'th elements in B don\'t reflect in A')
            return
        for i_ in range(len(A)):
            i = randIdx_i[i_]
            if t_results[i,j] == 1:
                t_results[:,j] = 0
                t_results[i,j] = 1
                # t_results[i+1:,j] = 0
                # temp |= t_results[i]
                # print('current temp is : ', temp)
                break
        ## have already found all plans, quit
        # if temp.all() == 1:
        #     break
    return t_results

if __name__ == "__main__":
    ##read files
    lines = []
    with open('input.txt', 'r') as f:
        lines = f.readlines()

    A_size = lines[0]
    A_size = int(A_size.replace('\n', ''))
    A = lines[1].replace('\n', '').split(',')
    A = [float(i) for i in A]
    assert A_size == len(A)

    B_size = lines[2]
    B_size = int(B_size.replace('\n', ''))
    B = lines[3].replace('\n', '').split(',')
    B = [float(i) for i in B]
    assert B_size == len(B)
    print('***** The input is *****')
    print('A: ', A)
    print('B: ', B)

    t_results = greedySearch2(A, B)
    cost = calCost(t_results)
    print(t_results)
    print('cost: ', cost)