# Java API

## Basic

```java

i << 1; // 左移  *2
i >> 1; // 右移   /2
i >>> 1; // unsigned 右移 /2

(int) x == x // check if x is a integer.

Character.getNumericValue(char ch)
//Returns the int value that the specified Unicode character represents.
Character.isAlphabetic(int codePoint)
//Determines if the specified character (Unicode code point) is an alphabet.

Character.isDigit(char ch)
Determines if the specified character is a digit.

```
## Character


## StringBuilder
```java
setLength(int newLength)
//Sets the length of the character sequence. This is used to remove the last appendant

StringBuilder	insert(int offset, String str)
//Inserts the string into this character sequence.
//The offset argument must be greater than or equal to 0, and less than or equal to the length of this sequence.
```

## String
```java
indexOf(char)
split(String reg);
substring(s,e);
charAt();
setCharAt();

StringBuilder sb =  new StringBuilder();
while(!stack.isEmpty()) {
    sb.insert(0, stack.pop());  // use insert and append
}
```

## Math


## Array
```java
// dp array creation, usually it should be nums.length+1
boolean[] dp = new boolean[nums.length];

// backtrack visited array creation
if (board == null || board.length == 0) return false;
boolean[][] visited = new boolean[board.length][board[0].length];

// sort by reverse order
Arrays.sort(arr, Collections.reverseOrder()); 

Integer[] arr2 = new Integer[] {54,432,53,21,43};
Arrays.sort(arr2, Comparator.reverseOrder());
```

### Convert Aarry to List
```java

1. 
LinkedList<Integer> l = new LinkedList<>();
for (int i = 0; i < digits.length; i++) {
    l.offer(digits[i]);
}

2. 
List l = Arrays.asList(digits); // digits must be Integer[], not int[];

List l = Arrays.asList(1,2,3); // this is OK, it will be boxed automatically.

3.
List<Integer> list = Arrays.stream(digits).boxed().collect(Collectors.toList());
// convert list to linkedlist
LinkedList<Integer> l = new LinkedList<>(list); 
```

### List to Array

```java
list as List<Integer>
list.stream().mapToInt(x -> x).toArray();

String[] arr = list.toArray(new String[list.size()]); 
```

## Tuple class

```java

class Tuple implements Comparable<Tuple> {
    int x, y, val;
    public Tuple (int x, int y, int val) {
        this.x = x;
        this.y = y;
        this.val = val;
    }
    
    @Override
    public int compareTo (Tuple that) {
        return this.val - that.val;
    }
}
```

## LinkedList
Doubly-linked list implementation of the List and Deque interfaces. Implements all optional list operations, and permits all elements **(including null)**.

This interface defines methods to access the elements at both ends of the deque. Methods are provided to insert, remove, and examine the element. Each of these methods exists in two forms: one throws an exception if the operation fails, the other returns a special value (either null or false, depending on the operation). The latter form of the insert operation is designed specifically for use with capacity-restricted Deque implementations; in most implementations, insert operations cannot fail.

```java

boolean list.offer(obj); // could offer null, but you can't use add, it don't allow null
//Adds the specified element as the tail (last element) of this list.

list.poll();
//Retrieves and removes the head of this list, or null if this list is empty

boolean list.offerFirst(obj);
//Inserts the specified element at the front of this list.

list.pollLast();
//Retrieves and removes the last element of this list, or returns null if this list is empty.

list.get(int index);
//Returns the element at the specified position in this list.

list.getFirst()
//Returns the first element in this list.Throw ex when list is empty.

list.peek()
//Retrieves, but does not remove, the head (first element) of this list.

list.peekLast()
//Retrieves, but does not remove, the last element of this list, or returns null if this list is empty.

list.getLast()
//Returns the last element in this list. Throw ex when list is empty.
```

## ArrayList

```java
Collections.emptyList();
// empty list;

list.sort(Comparator.naturalOrder());
Collections.sort(list);
Collections.sort(list, Compartor.reverseOrder());
```

## PriortyQueue

```java
PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());
        pq.addAll(map.entrySet());
```

## Stream

```java
//生成字符串
String s = list.stream().collect(Collectors.joining("/"));
//过滤， 计数。
long count1 = Arrays.stream(nums).filter(x -> x==n1).count();
```

## Map

java 9 initialize

```java
Map<String, String> emptyMap = Map.of();
Map<String, String> singletonMap = Map.of("key1", "value");
Map<String, String> map = Map.of("key1","value1", "key2", "value2");


getOrDefault(Object key, V defaultValue)
//Returns the value to which the specified key is mapped, or defaultValue if this map contains no mapping for the key.

putIfAbsent(K key, V value)
//If the specified key is not already associated with a value (or is mapped to null) associates it with the given value and returns null, else returns the current value.
```

* TreeMap.

```java

TreeMap(Comparator<? super K> comparator)
//Constructs a new, empty tree map, ordered according to the given comparator.

floorKey(K key)
//Returns the greatest key less than or equal to the given key, or null if there is no such key.

forEach(BiConsumer<? super K,? super V> action)
//Performs the given action for each entry in this map until all entries have been processed or the action throws an exception.

headMap(K toKey, boolean inclusive)
//Returns a view of the portion of this map whose keys are less than (or equal to, if inclusive is true) toKey.

higherKey(K key)
//Returns the least key strictly greater than the given key, or null if there is no such key.

lowerKey(K key)
//Returns the greatest key strictly less than the given key, or null if there is no such key.

lastKey()
//Returns the last (highest) key currently in this map.

firstKey()
//Returns the first (lowest) key currently in this map.

tailMap(K fromKey, boolean inclusive)
//Returns a view of the portion of this map whose keys are greater than (or equal to, if inclusive is true) fromKey.

ceilingKey(K key)
//Returns the least key greater than or equal to the given key, or null if there is no such key.
```

## Set


## List algorithm

* in the loop, always use the temp at the beginning of the loop. not the end of the loop;

https://leetcode.com/submissions/detail/279046317/ 
```java
ListNode odd1 = head;
ListNode even1 = head.next;
ListNode odd2 = even1.next;
ListNode even2 = odd2.next;
while (even1 != null && even1.next != null) {
    odd2 = even1.next; // alwasy set the temp node at the beginning of the loop.
    even2 = odd2.next;
    odd1.next = odd2;
    even1.next = even2;
    odd1 = odd2;
    even1 = even2;  
}
```

## Sort and select

stand partition by qsort.

```java
// Standard partition process of QuickSort.  
    // It considers the last element as pivot  
    // and moves all smaller element to left of 
    // it and greater elements to right 
public static int partition(Integer [] arr, int l,  
                                               int r) 
    { 
        int x = arr[r], i = l; // always keep arr[i] > x or i == j
        for (int j = l; j <= r - 1; j++) 
        { 
            if (arr[j] <= x) 
            { 
                //Swapping arr[i] and arr[j] 
                int temp = arr[i]; 
                arr[i] = arr[j]; 
                arr[j] = temp; 
  
                i++; 
            } 
        } 
          
        //Swapping arr[i] and arr[r] 
        int temp = arr[i]; 
        arr[i] = arr[r]; 
        arr[r] = temp; 
  
        return i; 
    } 
```
 
### merge sort related problem.
https://leetcode.com/problems/reverse-pairs/discuss/97268/General-principles-behind-problems-similar-to-%22Reverse-Pairs%22 

## Binary Search
如果使用lo<=hi的方式, find the exact element in the array. 
```java
while (lo <= hi) {
    int mid = (lo+hi) >> 1;
    if (nums[mid] == target) return mid;
    if () {
        lo = mid + 1;
    } else {
        hi = mid - 1;
    }
}

at the end of the loop, lo = hi + 1;
```

if you want to find element > or < closest to the target.

```java
lo = 0;
hi = nums.length-1;

while (lo < hi) {
    int mid = (lo+hi) >> 1;
    if (num[mid] < target) { // find the first item >= target
        lo = mid + 1; 
    } else {
        hi = mid;
    }
}

while (lo < hi) {
    int mid = (lo+hi + 1) >> 1;
    if (num[mid] > target>) { // find the last item <= target
        hi = mid - 1; 
    } else {
        lo = mid;
    }
}

at the end of the loop, lo = hi;
```

https://leetcode.com/explore/learn/card/binary-search/

http://www.codebelief.com/article/2018/04/completely-understand-binary-search-and-its-boundary-cases/

https://leetcode.com/problems/find-k-th-smallest-pair-distance/discuss/109082/Approach-the-problem-using-the-%22trial-and-error%22-algorithm

## Union find
中间可能有很多的中间节点，总要用find才能找到根节点。
https://www.hackerearth.com/zh/practice/notes/disjoint-set-union-union-find/

```java
public int find(int[] f, int id){
    while (id!=f[id]){
        f[id]=f[f[id]];
        id=f[id];
    }
    return id;
}

public void union(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX != rootY) {
        id[rootX] = rootY;
    }
}
```

# Tree

### Preorder
https://www.geeksforgeeks.org/iterative-preorder-traversal/
```java
 void iterativePreorder(Node node) { 
          
        // Base Case 
        if (node == null) { 
            return; 
        } 
  
        // Create an empty stack and push root to it 
        Stack<Node> nodeStack = new Stack<Node>(); 
        nodeStack.push(root); 
  
        /* Pop all items one by one. Do following for every popped item 
         a) print it 
         b) push its right child 
         c) push its left child 
         Note that right child is pushed first so that left is processed first */
        while (nodeStack.empty() == false) { 
              
            // Pop the top item from stack and print it 
            Node mynode = nodeStack.peek(); 
            System.out.print(mynode.data + " "); 
            nodeStack.pop(); 
  
            // Push right and left children of the popped node to stack 
            if (mynode.right != null) { 
                nodeStack.push(mynode.right); 
            } 
            if (mynode.left != null) { 
                nodeStack.push(mynode.left); 
            } 
        } 
    } 
```

### Inorder
1) Create an empty stack S.
2) Initialize current node as root
3) Push the current node to S and set current = current->left until current is NULL
4) If current is NULL and stack is not empty then 
     * a) Pop the top item from stack.
     * b) Print the popped item, set current = popped_item->right 
     * c) Go to step 3.
5) If current is NULL and stack is empty then we are done.
```java
void inorder() 
    { 
        if (root == null) 
            return; 
  
  
        Stack<Node> s = new Stack<Node>(); 
        Node curr = root; 
  
        // traverse the tree 
        while (curr != null || s.size() > 0) 
        { 
  
            /* Reach the left most Node of the 
            curr Node */
            while (curr !=  null) 
            { 
                /* place pointer to a tree node on 
                   the stack before traversing 
                  the node's left subtree */
                s.push(curr); 
                curr = curr.left; 
            } 
  
            /* Current must be NULL at this point */
            curr = s.pop(); 
  
            System.out.print(curr.data + " "); 
  
            /* we have visited the node and its 
               left subtree.  Now, it's right 
               subtree's turn */
            curr = curr.right; 
        } 
    } 




public void morrisTraversal(TreeNode root){
		TreeNode temp = null;
		while(root!=null){
			if(root.left!=null){
				// connect threading for root
				temp = root.left;
				while(temp.right!=null && temp.right != root)
					temp = temp.right;
				// the threading already exists
				if(temp.right!=null){
					temp.right = null;
					System.out.println(root.val); // visit the node
					root = root.right;
				}else{
					// construct the threading
					temp.right = root;
					root = root.left;
				}
			}else{
				System.out.println(root.val); // visit the node
				root = root.right;
			}
		}
	}

```

### post order
1. Push root to first stack.
2. Loop while first stack is not empty
   * 2.1 Pop a node from first stack and push it to second stack
   * 2.2 Push left and right children of the popped node to first stack
3. Print contents of second stack
```java
static void postOrderIterative(node root) 
    { 
        // Create two stacks 
        s1 = new Stack<>(); 
        s2 = new Stack<>(); 
  
        if (root == null) 
            return; 
  
        // push root to first stack 
        s1.push(root); 
  
        // Run while first stack is not empty 
        while (!s1.isEmpty()) { 
            // Pop an item from s1 and push it to s2 
            node temp = s1.pop(); 
            s2.push(temp); 
  
            // Push left and right children of 
            // removed item to s1 
            if (temp.left != null) 
                s1.push(temp.left); 
            if (temp.right != null) 
                s1.push(temp.right); 
        } 
  
        // Print all elements of second stack 
        while (!s2.isEmpty()) { 
            node temp = s2.pop(); 
            System.out.print(temp.data + " "); 
        } 
    } 
```

## Graph

### Cycle
https://www.geeksforgeeks.org/detect-cycle-in-a-graph/

https://www.geeksforgeeks.org/detect-cycle-undirected-graph/

## Indegree


## Backtrack
### Subsets, Permutations, Combination Sum, Palindrome Partitioning
https://leetcode.com/problems/subsets/discuss/27281/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partitioning)

**backtrack 如果可以有返回值（数值）求和，就相当于 top down dp，可以cache返回值剪枝。**

## Sliding windonws

https://leetcode.com/problems/find-all-anagrams-in-a-string/discuss/92007/Sliding-Window-algorithm-template-to-solve-all-the-Leetcode-substring-search-problem. 

2D sliding window

https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/discuss/83599/Accepted-C%2B%2B-codes-with-explanation-and-references

https://leetcode.com/problems/minimum-window-substring/discuss/26808/Here-is-a-10-line-template-that-can-solve-most-'substring'-problems

1. Use two pointers: start and end to represent a window.
2. Move end to find a valid window.
3. When a valid window is found, move start to find a smaller window.

## Single Number Bit manupulation
http://liadbiz.github.io/leetcode-single-number-problems-summary/

## DFS

2d Matrix offset
```java 
public static final int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
```

## BFS

## Bits
https://leetcode.com/problems/sum-of-two-integers/discuss/84278/A-summary%3A-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently

## DP

* 对于01背包问题，需要dp数组对应Weight的值来建立。

https://www.jianshu.com/p/03340080165e

https://zxi.mytechroad.com/blog/sp/knapsack-problem/
