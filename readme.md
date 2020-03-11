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

// create a generic List Array
List<Integer>[] bucket = new List[nums.length + 1];
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

// Map to object, and sort with lamda
Arrays.stream(nums).mapToObj(x1 -> String.valueOf(x1)).sorted((String s1,String s2) -> {
            int i = (s1+s2).compareTo(s2+s1);
            return i * -1;
        }).collect(Collectors.joining());

list.stream().mapToInt(x -> x).toArray();
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


// iterate the map
// using for-each loop for iteration over Map.entrySet() 
        for (Map.Entry<String,String> entry : map.entrySet())  
            System.out.println("Key = " + entry.getKey() + 
                             ", Value = " + entry.getValue());
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
    if (num[mid] < target) { // find the first item >= target 这个就是维护lo的一个过程，找到第一个不符合num[mid] < target这个条件的元素。
        lo = mid + 1; 
    } else {
        hi = mid;
    }
}

while (lo < hi) {
    int mid = (lo+hi + 1) >> 1;
    if (num[mid] > target>) { // find the last item <= target，这个和上门的刚好相反。
        hi = mid - 1; 
    } else {
        lo = mid;
    }
}
```
at the end of the loop, lo = hi; **如果最后数组中没有符合条件的结果，这个算法最后会停在索引0或者nums.length-1，所以要判断一下最后停的位置是不是要的结果。如果数组中肯定有结果，那就无所谓了。**


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


    void MorrisTraversal(tNode root) 
    { 
        tNode current, pre; 
  
        if (root == null) 
            return; 
  
        current = root; 
        while (current != null) { 
            if (current.left == null) { 
                System.out.print(current.data + " "); 
                current = current.right; 
            } 
            else { 
                /* Find the inorder predecessor of current */
                pre = current.left; 
                while (pre.right != null && pre.right != current) 
                    pre = pre.right; 
  
                /* Make current as right child of its inorder predecessor */
                if (pre.right == null) { 
                    pre.right = current; 
                    current = current.left; 
                } 
  
                /* Revert the changes made in the 'if' part to restore the  
                    original tree i.e., fix the right child of predecessor*/
                else { 
                    pre.right = null; 
                    System.out.print(current.data + " "); 
                    current = current.right; 
                } /* End of if condition pre->right == NULL */
  
            } /* End of if condition current->left == NULL*/
  
        } /* End of while */
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

* 对于01背包问题，需要dp数组对应Weight的值来建立。第一层循环是循环每一个物品。第二层是逆序的，循环背包的weight。

我们看到的求最优解的背包问题题目中，事实上有两种不太相同的问法。有的题目要求“恰好装满背包”时的最优解，有的题目则并没有要求必须把背包装满。一种区别这两种问法的实现方法是在初始化的时候有所不同。

如果是第一种问法，要求恰好装满背包，那么在初始化时除了f[0]为0其它f[1..V]均设为-∞，这样就可以保证最终得到的f[N]是一种恰好装满背包的最优解。

如果并没有要求必须把背包装满，而是只希望价格尽量大，初始化时应该将f[0..V]全部设为0。

为什么呢？可以这样理解：初始化的f数组事实上就是在没有任何物品可以放入背包时的合法状态。如果要求背包恰好装满，那么此时只有容量为0的背包可能被价值为0的nothing“恰好装满”，其它容量的背包均没有合法的解，属于未定义的状态，它们的值就都应该是-∞了。如果背包并非必须被装满，那么任何容量的背包都有一个合法解“什么都不装”，这个解的价值为0，所以初始时状态的值也就全部为0了。

这个小技巧完全可以推广到其它类型的背包问题，后面也就不再对进行状态转移之前的初始化进行讲解。

https://www.jianshu.com/p/03340080165e

https://zxi.mytechroad.com/blog/sp/knapsack-problem/

https://github.com/tianyicui/pack/blob/master/V2.pdf

## segment tree

https://zxi.mytechroad.com/blog/sp/segment-tree-sp14/

## Time complexity.

https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms) 

## Tomcat webappclassloader

1. 由于classloader 加载类用的是全盘负责委托机制。所谓全盘负责，即是当一个classloader加载一个Class的时候，这个Class所依赖的和引用的所有 Class也由这个classloader负责载入，除非是显式的使用另外一个classloader载入，HelloServlet是由WebappClassLoaderBase加载的，那么Hello也由WebappClassLoaderBase，可以自行打断点验证

2. findLoadedClass0("test.Hello")查看当前类加载器resourceEntries是否缓存

    protected Class<?> findLoadedClass0(String name) { String path = binaryNameToPath(name, true); ResourceEntry entry = resourceEntries.get(path); if (entry != null) { return entry.loadedClass; } return null; } 

3. findLoadedClass("test.Hello") 从native方法中查看类缓存

protected final Class<?> findLoadedClass(String name) { if (!checkName(name)) return null; return findLoadedClass0(name); } private native final Class<?> findLoadedClass0(String name); 

4. ClassLoader javaseLoader = getJavaseClassLoader() 得到的sun.misc.Launcher￼AppClassLoader）clazz = Class.forName(name, false, parent);
比如以javax开头的类
6. clazz = findClass(name); 由当前类加载WebappClassLoaderBase加载，从/WEB-INF/classes/test/Hello.class进行查找文件将文件放入byte[],transformer.transform()进行插桩改造byte[],最终defineClass生成class
7. WebappClassLoaderBase加载不到的类由父类加载器AppClassLoader 记载clazz = Class.forName(name, false, parent);
8.throw new ClassNotFoundException(name); 第7步中加载不到就抛异常啦

### Spring classloader
Spring class loader
https://www.cnblogs.com/binarylei/p/10312531.html
