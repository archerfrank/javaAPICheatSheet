# Java API

## Basic

```java

i << 1; // 左移  *2
i >> 1; // 右移   /2
i >>> 1; // unsigned 右移 /2

```
## Character


## StringBuilder


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

3.
List<Integer> list = Arrays.stream(digits).boxed().collect(Collectors.toList());
// convert list to linkedlist
LinkedList<Integer> l = new LinkedList<>(list); 
```

### List to Array

```java
list as List<Integer>
list.stream().mapToInt(x -> x).toArray();
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
//Returns the first element in this list.

list.peek()
//Retrieves, but does not remove, the head (first element) of this list.

list.peekLast()
//Retrieves, but does not remove, the last element of this list, or returns null if this list is empty.
```

## ArrayList

```java
Collections.emptyList();
// empty list;

```

## PriortyQueue

## Stream

```java
//生成字符串
String s = list.stream().collect(Collectors.joining("/"));
//过滤， 计数。
long count1 = Arrays.stream(nums).filter(x -> x==n1).count();
```

## Map

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

