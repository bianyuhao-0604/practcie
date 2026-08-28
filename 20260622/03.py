class  Node:
    def __init__(self,value):
        self.value = value
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,value):
        node = Node(value)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node
    def to_list(self):
        result = []
        cur = self.head
        while cur:
            result.append(cur.value)
            cur = cur.next
        return result
if __name__ == "__main__":
        ll = LinkedList()
    
        ll.append(1)
        ll.append(2)
        ll.append(3)
    
        print(ll.to_list())