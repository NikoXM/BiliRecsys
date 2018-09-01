import heapq

class TopkHeap:
    def __init__(self, k_int):
        self.k_int = k_int
        self.data_list = []

    def push(self, element_tuple):
        if len(self.data_list) < self.k_int:
            heapq.heappush(self.data_list, element_tuple)
        else:
            if element_tuple[0] > self.data_list[0][0]:
                heapq.heapreplace(self.data_list, element_tuple)

    def topK(self):
        return sorted(self.data_list, key=lambda x: x[0])
