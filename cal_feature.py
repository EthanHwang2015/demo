#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import heapq
import numpy as np
import scipy.spatial.distance
import time

class TimeCost():
    def __init__(self):
        self.start = time.time()
    def cost(self, reset=True):
        cost = time.time() - self.start
        if reset:
            self.start = time.time()
        return cost

    def reset(self):
        self.start = time.time()

class TopKHeap(object):
    def __init__(self, k, heap_type='min'):
        self.k = k
        self.data = []
        self.heap_type = heap_type

    #elem in tuple format => (value, id)
    def push(self, elem):
        if self.heap_type == 'max':
            elem = (-elem[0],elem[1])

        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem[0] > topk_small[0]:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        tops = []
        for i in xrange(len(self.data)):
            tops.append(heapq.heappop(self.data))
        reverse_list = reversed(tops)

        if self.heap_type == 'max':
            return [(-x[0],x[1]) for x in reverse_list]
        else:
            return [x for x in reverse_list]

def cos_distance1(v1,v2):
    v1 = v1.reshape(1,-1)
    v2 = v2.reshape(1,-1)
    return scipy.spatial.distance.cdist(v1,v2,'cosine')

def cos_distance2(v1,v2):
    dot = v1.dot(v2)
    v1_norm = np.linalg.norm(v1,2)
    v2_norm = np.linalg.norm(v2,2)
    multi = np.multiply(v1_norm, v2_norm)
    return 1 - np.divide(dot, multi)

def cos_distance3(matrix,v):
    dot = matrix.dot(v)    
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(v)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    return 1  - np.divide(dot, matrix_vector_norms)

def cos_distance4(matrix,v, matrix_norms):
    dot = matrix.dot(v)    
    #matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(v)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    return 1  - np.divide(dot, matrix_vector_norms)

def testTopKheap():
    #大顶堆求最小topk
    th = TopKHeap(3, 'max')

    th.push((1,'g'))
    th.push((2,'h'))
    th.push((3,'a'))
    th.push((4,'b'))
    th.push((5,'d'))
    th.push((6,'f'))
    tops = th.topk()
    for top in tops:
        print top

def main():
    tc = TimeCost()
    raw_user_vec = np.loadtxt("online_req.tail_986_head100", delimiter=",")
    #remove first column =>userid
    user_vec = raw_user_vec[:,1:]
    user_id = np.array(raw_user_vec[:,0], dtype=np.int64)
    print 'load user cost', tc.cost()

    #remove first column =>userid
    #raw_item_vec= np.loadtxt("post_vec.top10000", delimiter=",")
    raw_item_vec= np.loadtxt("60W", delimiter=",")
    item_vec = raw_item_vec[:,1:]
    item_id = np.array(raw_item_vec[:,0], dtype=np.int64)
    print 'load item cost', tc.cost()

    matrix_norms = np.linalg.norm(item_vec, axis=1)
    for i in xrange(len(user_vec)):
        #distance_list = cos_distance3(item_vec, user_vec[i])
        distance_list = cos_distance4(item_vec, user_vec[i], matrix_norms)
        #大顶堆求最小topk
        myheap = TopKHeap(30, 'max')
        for j in xrange(len(item_vec)):
            myheap.push((distance_list[j],item_id[j]))
        topks = myheap.topk()
        #print user_id[i],topks

    print 'calculate cost', tc.cost()

if __name__ == "__main__":
    #testTopKheap()
    main()
