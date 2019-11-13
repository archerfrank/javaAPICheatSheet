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

indexOf(char)
split(String reg);
substring(s,e);
charAt();
setCharAt();

## Array
```java
boolean[] dp = new boolean[nums.length];
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
Adds the specified element as the tail (last element) of this list.

list.poll();
Retrieves and removes the head of this list, or null if this list is empty

boolean list.offerFirst(obj);
Inserts the specified element at the front of this list.

list.pollLast();
Retrieves and removes the last element of this list, or returns null if this list is empty.

list.get(int index);
Returns the element at the specified position in this list.

list.getFirst()
Returns the first element in this list.

list.peek()
Retrieves, but does not remove, the head (first element) of this list.

list.peekLast()
Retrieves, but does not remove, the last element of this list, or returns null if this list is empty.
```

## ArrayList

## PriortyQueue

## Stream

## Map

## Set