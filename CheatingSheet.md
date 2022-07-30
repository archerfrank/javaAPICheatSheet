

# Java Cheating Sheet 

## 易忘记思路

### 二分查找
### 逆向思维


## 滑动窗口

```python
def findSubArray(nums):
    N = len(nums) # 数组/字符串长度
    left, right = 0, 0 # 双指针，表示当前遍历的区间[left, right]，闭区间
    sums = 0 # 用于统计 子数组/子区间 是否有效，根据题目可能会改成求和/计数
    res = 0 # 保存最大的满足题目要求的 子数组/子串 长度
    while right < N: # 当右边的指针没有搜索到 数组/字符串 的结尾
        sums += nums[right] # 增加当前右边指针的数字/字符的求和/计数
        while 区间[left, right]不符合题意：# 此时需要一直移动左指针，直至找到一个符合题意的区间
            sums -= nums[left] # 移动左指针前需要从counter中减少left位置字符的求和/计数
            left += 1 # 真正的移动左指针，注意不能跟上面一行代码写反
        # 到 while 结束时，我们找到了一个符合题意要求的 子数组/子串
        res = max(res, right - left + 1) # 需要更新结果
        right += 1 # 移动右指针，去探索新的区间
    return res

作者：fuxuemingzhu
链接：https://leetcode-cn.com/problems/max-consecutive-ones-iii/solution/fen-xiang-hua-dong-chuang-kou-mo-ban-mia-f76z/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



## Union Find
```python
p = [i for i in range(n)]
def find(x):
    if p[x] != x:
        p[x] = find(p[x])
    return p[x]
def union(x, y):
    p[find(x)] = find(y)
    
def defaultdict_int():
    return defaultdict(int)


# 并查集模板
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.n = n
        # 当前连通分量数目
        self.setCount = n
    
    def findset(self, x: int) -> int:
        if self.parent[x] == x:
            return x
        self.parent[x] = self.findset(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        x, y = self.findset(x), self.findset(y)
        if x == y:
            return False
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]
        self.setCount -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        x, y = self.findset(x), self.findset(y)
        return x == y
    
 
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n   # 层高

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x: int, y: int) -> None:
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        elif self.rank[x] < self.rank[y]:
            self.parent[x] = y
        else:
            self.parent[y] = x
            self.rank[x] += 1

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/largest-component-size-by-common-factor/solution/an-gong-yin-shu-ji-suan-zui-da-zu-jian-d-amdx/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

```java
// 带权重union find
private class UnionFind {

    private int[] parent;

    /**
        * 指向的父结点的权值
        */
    private double[] weight;
    public UnionFind(int n) {
        this.parent = new int[n];
        this.weight = new double[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            weight[i] = 1.0d;
        }
    }

    public void union(int x, int y, double value) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX == rootY) {
            return;
        }

        parent[rootX] = rootY;
      	// 关系式的推导请见「参考代码」下方的示意图
        weight[rootX] = weight[y] * value / weight[x];
    }

    /**
     * 路径压缩
     *
     * @param x
     * @return 根结点的 id
     */
    public int find(int x) {
        if (x != parent[x]) {
            int origin = parent[x];
            parent[x] = find(parent[x]);
            weight[x] *= weight[origin];
        }
        return parent[x];
    }

    public double isConnected(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX == rootY) {
            return weight[x] / weight[y];
        } else {
            return -1.0d;
        }
    }
}


//按rank union
private class UnionFind {

        private int[] parent;
        /**
         * 以 i 为根结点的子树的高度（引入了路径压缩以后该定义并不准确）
         */
        private int[] rank;

        public UnionFind(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
            for (int i = 0; i < n; i++) {
                this.parent[i] = i;
                this.rank[i] = 1;
            }
        }

        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return;
            }

            if (rank[rootX] == rank[rootY]) {
                parent[rootX] = rootY;
                // 此时以 rootY 为根结点的树的高度仅加了 1
                rank[rootY]++;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
                // 此时以 rootY 为根结点的树的高度不变
            } else {
                // 同理，此时以 rootX 为根结点的树的高度不变
                parent[rootY] = rootX;
            }
        }

        public int find(int x) {
            if (x != parent[x]) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
    }
```

## SortedList

```python
import sortedcontainers
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        sl = sortedcontainers.SortedList()
```



## Segment Tree

In [here](https://github.com/archerfrank/javaAPICheatSheet/blob/master/segment_tree.md) 

## Binary indexed tree

```java
class BIT {
    int[] tree;
    int n;

    public BIT(int n) {
        this.n = n;
        this.tree = new int[n + 1];
    }

    public static int lowbit(int x) {
        return x & (-x);
    }

    public void update(int x, int newValue, int oldValue) {
        while (x <= n) {
            tree[x] = tree[x]-oldValue+newValue;
            x += lowbit(x);
        }
    }

    public int query(int x) {
        int ans = 0;
        while (x != 0) {
            ans += tree[x];
            x -= lowbit(x);
        }
        return ans;
    }
}

```
注意查询的时候是从1开始的，也就是和presum类似。0代表没有，1代表第一个数。
```python
class BIT:
    def __init__(self, nums: List[int]):
        self.sum = [0] * (len(nums) + 1)
        self.n = len(nums)
        for i in range(len(nums)):
            self.update(i, 0, nums[i])
    #[i,j]全包含查询
    def sumRange(self, i: int, j: int) -> int:
        return self.query(j + 1) - self.query(i)

    def lowbit(self, i: int) -> int:
        return i & (-i)

    def update(self, pos:int, oldValue: int, newValue:int):
        pos = pos + 1 # 更新的位置要加1
        while pos <= self.n:
            self.sum[pos] += newValue - oldValue
            pos += self.lowbit(pos)

    def query(self, pos:int) -> int:
        ans = 0
        while pos > 0:
            ans += self.sum[pos]
            pos -= self.lowbit(pos)
        return ans
```

## LRU Cache
```python
        n, m = len(nums), len(multipliers)
        @lru_cache(None)
        def helper(i, j, idx):
            if idx == m:
                return 0
            r1 = multipliers[idx] * nums[i] + helper(i+1, j, idx+1)
            r2 = multipliers[idx] * nums[j] + helper(i, j-1, idx + 1)
            return max(r1, r2)
        res = helper(0, n-1, 0)
        helper.cache_clear()
```

## 最短路径 shortest path  dijkstra

```python
		adj = defaultdict(list)
        for x, y,w in edges:
            adj[x].append((y, w))
            adj[y].append((x, w))
        q = [(0, n)]
        cost = [inf] * (n+1)
        cost[n] = 0
        while q:
            v = heapq.heappop(q)
            if v[0] > cost[v[1]]: #dijkstra 每个节点只会出队列一次，后面出队列都可以抛弃
                continue
            for child in adj[v[1]]:
                if v[0] + child[1] < cost[child[0]]:
                    cost[child[0]] = v[0] + child[1]
                    heapq.heappush(q, (cost[child[0]], child[0]))
```


```python
        adj = defaultdict(list)
        for x, y,w in edges:
            adj[x].append((y, w))
            adj[y].append((x, w))
        q = [(0, n)]
        cost = [inf] * (n+1)
        cost[n] = 0
        visit = [0] * n
        while q:
            v = heapq.heappop(q)
            if visit[v[1]] == 0:   #dijkstra 每个节点只会出队列一次，后面出队列都可以抛弃
                visit[v[1]] = 1
                for child in adj[v[1]]:
                    if v[0] + child[1] < cost[child[0]]:
                        cost[child[0]] = v[0] + child[1]
                        heapq.heappush(q, (cost[child[0]], child[0]))
```
### GCD

```java
    private static int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }
```

### DP 状态压缩法
```python
        n = len(nums1)
        f = [float("inf")] * (1 << n)
        f[0] = 0
        for mask in range(1, 1 << n):
            c = bin(mask).count("1")
            for i in range(n):
                if mask & (1 << i):
                    f[mask] = min(f[mask], f[mask ^ (1 << i)] + (nums1[c - 1] ^ nums2[i]))
        
        return f[(1 << n) - 1]
```

作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/minimum-xor-sum-of-two-arrays/solution/liang-ge-shu-zu-zui-xiao-de-yi-huo-zhi-z-2uye/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
### Fast Power



```python
        def power(base,exponent):
            res = 1
            while exponent:
                if exponent & 1:  # 判断当前的最后一位是否为1，如果为1的话，就需要把之前的幂乘到结果中。
                    res *= base
                    res %= mod
                base *= base % mod  # 一直累乘，如果最后一位不是1的话，就不用了把这个值乘到结果中，但是还是要乘。
                base %= mod
                exponent = exponent >> 1
            return res % mod
```

### Mask Iteration

```python
	mask = 15
    temp = mask
    while temp:
        temp = temp - 1
        temp = mask & temp
```


### Tries XOR
```python
class Node:
    def __init__(self, val):
        self.left=None
        self.right=None
        self.val = val
        self.N = 20
    # def __str__(self):
    #     return str(self.val) + ", left : " + self.left if self.left else "None" +", right :" +self.right if self.right else "None"

    def add(self, num):
        cur = self
        for i in range(self.N):
            x = (1 << (self.N - 1 - i))
            if x & num > 0:
                if cur.right is None:
                    cur.right = Node(1)
                else:
                    cur.right.val += 1
                cur = cur.right
            else:
                if cur.left is None:
                    cur.left = Node(1)
                else:
                    cur.left.val += 1
                cur = cur.left
    def remove(self, num):
        cur = self
        for i in range(self.N):
            x = (1 << (self.N - 1 - i))
            if x & num > 0:
                cur.right.val -= 1
                if cur.right.val == 0:
                    cur.right = None
                    break
                cur = cur.right
            else:
                cur.left.val -= 1
                if cur.left.val == 0:
                    cur.left = None
                    break
                cur = cur.left
    def query(self, num):
        cur = self
        ans = 0
        for i in range(self.N):
            x = (1 << (self.N - 1 - i))
            if x & num > 0:
                if cur.left:
                    ans |= (1 << (self.N - 1 - i))
                    cur = cur.left
                else:
                    cur = cur.right
            else:
                if cur.right:
                    ans |= (1 << (self.N - 1 - i))
                    cur = cur.right
                else:
                    cur = cur.left
            # print(ans, x & num)
        return ans

```
下面这个版本更加通用。
```python
class TreeNode:
    def __init__ (self):
        self.children = defaultdict(TreeNode)
        self.end = False
    # def __str__(self):
    #     s = ""
    #     for k in self.children:
    #         s += k +":" + str(self.children[k]) + ";"
        return s + str(self.end)
class Tries:
    def __init__ (self):
        self.root = TreeNode()
    def insert(self, word):
        cur = self.root
        for x in word:
            if x not in cur.children:
                cur.children[x] = TreeNode()
            cur = cur.children[x]
        cur.end = True

#build
self.root = Tries()
    for w in dictionary:
        self.root.insert(w)

#query
cur = self.root
for w in word:
    if w in cur.children:
        cur = cur.children[w]
    else:
        return False
return cur.end








#带父节点的前缀树
class Trie:
    def __init__ (self):
        self.list = defaultdict(Trie)
        self.end = False
        self.cur = self
        self.parent = None
    #插入词
    def insert(self, word):
        for x in word:
            if x not in self.cur.list:
                self.cur.list[x] = Trie()
                self.cur.list[x].parent = self.cur
            self.cur = self.cur.list[x]
        self.cur.end = True
        self.cur = self
    
    #获取当前位置，用于回溯
    def getcur(self):
        return self.cur
    #设置当前位置
    def setcur(self, c):
        self.cur = c
    #查询单个字符
   	def query(self,chs):
        if chs in self.cur.list:
            self.cur = self.cur.list[chs]
            return True
		return False
    
    #删除当前节点，配合查询使用，可以删除词语
    def remove(self):
        self.cur = self.cur.parent
   
    def reset(self):
        self.cur = self

    def isEnd(self):
        return self.cur.end
```
## String Hash and KMP and Z 函数

允许K次失配的字符串匹配
最长回文子串
最长公共子字符串
上面三个都可以用string hash解决。
https://oi-wiki.org/string/hash/

https://oi-wiki.org/string/kmp/
前缀函数的应用

```python
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi
```

https://oi-wiki.org/string/z-func/#_3
后缀函数的应用
注意z数据的第一位是0

对于个长度为  的字符串 。定义函数  表示  和 （即以  开头的后缀）的最长公共前缀（LCP）的长度。 被称为  的 Z 函数。特别地，。

```python
# Python Version
def z_function(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r and z[i - l] < r - i + 1:
            z[i] = z[i - l]
        else:
            z[i] = max(0, r - i + 1)
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1
    return z

```


## String Hash and KMP and Z 函数
```python
MOD = 10 ** 9 + 7
base = 171
n = len(s)
val = 0

#用法一：可以滚动计算固定长度hash,比如固定长度k, hsh2就是结果
hsh2=[]
kbase = base ** (k-1)
for i in range(k):
    val = val * base + (ord(s[i]) - ord('a'))
    val %= MOD
    hsh2.append(val)
for i in range(k, n):
    val = (val + MOD) - (kbase * (ord(s[i - k]) - ord('a'))) % MOD
    val *= base
    val += (ord(s[i]) - ord('a'))
    val %= MOD
    hsh2.append(val)

#用法二，计算某一段字符串的hash，用来匹配。
#计算base的n次方，存储好
b = [1]
for i in range(n):
    b.append(b[-1] * base % MOD)

#计算前缀hash数组
hsh = [0]
for i in range(n):
    val = val * base + (ord(s[i]) - ord('a'))
    val %= MOD
    hsh.append(val)

#有了前缀，就可以计算任意一段的hash，比如s[i:j+1]
def check(i, j):
    # return 0
    l = j - i + 1
    val = hsh[i] * b[l] % MOD  #b数组在这里使用
    return (hsh[j + 1] + MOD - val) % MOD



```

```java
int strHash(String ss, String b) {//将要查找的字符串b放到被查找的字符串ss的后面，进行hash
        int P = 131;
        int n = ss.length(), m = b.length();
        String str = ss + b;
        int len = str.length();
        int[] h = new int[len + 10], p = new int[len + 10];
        h[0] = 0; p[0] = 1;
        for (int i = 0; i < len; i++) {
            p[i + 1] = p[i] * P;
            h[i + 1] = h[i] * P + str.charAt(i);
        }
        int r = len, l = r - m + 1;
        int target = h[r] - h[l - 1] * p[r - l + 1]; // b 的哈希值
        for (int i = 1; i <= n; i++) {
            int j = i + m - 1;
            int cur = h[j] - h[i - 1] * p[j - i + 1]; // 子串哈希值
            if (cur == target) return i - 1;
        }
        return -1;
    }
```

```python
P = 8597
X = 3613
def get_occurrences(pattern, text):
    res = []
    lenP = len(pattern)
    lenT = len(text)
    hp = getHash(pattern)
    ht = getHash(text[0 : lenP])

    # most significant coefficient
    msc = 1
    for i in range(lenP-1):
        msc = (msc * X) % P


    for i in range(1, lenT-lenP+2):
        if (hp == ht) and (pattern == text[i-1 : i-1 + lenP]):
            res.append(i-1)
        if i-1 + lenP < lenT:
            ht = (X*(ht - ord(text[i-1]) * msc) + ord(text[i-1 + lenP])) % P
            if ht < 0:
                ht += P

    return res

def getHash(s):
    res = 0
    l = len(s)
    for i in range(l):
        res += (ord(s[i]) * (X**(l-i-1))) % P
    return res % P
```

https://leetcode-cn.com/submissions/detail/251381185/

```python
https://leetcode-cn.com/submissions/detail/251381185/
a1, a2 = random.randint(26, 100), random.randint(26, 100)
        # 生成两个模
mod1, mod2 = random.randint(10**9+7, 2**31-1), random.randint(10**9+7, 2**31-1)
def check(self, arr, m, a1, a2, mod1, mod2):
        n = len(arr)
        aL1, aL2 = pow(a1, m, mod1), pow(a2, m, mod2)
        h1, h2 = 0, 0
        for i in range(m):
            h1 = (h1 * a1 + arr[i]) % mod1
            h2 = (h2 * a2 + arr[i]) % mod2
        # 存储一个编码组合是否出现过
        seen = {(h1, h2)}
        for start in range(1, n - m + 1):
            h1 = (h1 * a1 - arr[start - 1] * aL1 + arr[start + m - 1]) % mod1
            h2 = (h2 * a2 - arr[start - 1] * aL2 + arr[start + m - 1]) % mod2
            # 如果重复，则返回重复串的起点
            if (h1, h2) in seen:
                return start
            seen.add((h1, h2))
        # 没有重复，则返回-1
        return -1

```

## 二维前缀和使用
每个方向要多一排的0，和一维类似，都要多一个0.
```python
presum = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        presum[i][j] = presum[i- 1][j] + presum[i][j - 1] - presum[i-1][j - 1]+ grid[i - 1][j -1]

for x1, y1, x2, y2 in calcs:
    val = presum[x2+1][y2+1] - presum[x1][y2 + 1] - presum[x2 + 1][y1] + presum[x1][y1]

```



## 双向链表



```python
class Node:
    def __init__(self, key="", count=0):
        self.prev = None
        self.next = None
        self.keys = {key}
        self.count = count

    def insert(self, node: 'Node') -> 'Node':  # 在 self 后插入 node
        node.prev = self
        node.next = self.next
        node.prev.next = node
        node.next.prev = node
        return node

    def remove(self):  # 从链表中移除 self
        self.prev.next = self.next
        self.next.prev = self.prev

class AllOne:
    def __init__(self):
        self.root = Node()
        self.root.prev = self.root
        self.root.next = self.root  # 初始化链表哨兵，下面判断节点的 next 若为 self.root，则表示 next 为空（prev 同理）
        

链接：https://leetcode-cn.com/problems/all-oone-data-structure/solution/quan-o1-de-shu-ju-jie-gou-by-leetcode-so-7gdv/

```

## Random

```python
import random

print( random.randint(1,10) )        # 产生 1 到 10 的一个整数型随机数  
print( random.random() )             # 产生 0 到 1 之间的随机浮点数
# 随机整数：
print random.randint(1,50)

# 随机选取0到100间的偶数：
print random.randrange(0, 101, 2)
```

## Cross 坐标系

https://leetcode-cn.com/problems/erect-the-fence/solution/an-zhuang-zha-lan-by-leetcode-solution-75s3/

从上图中，我们可以观察到点 pp，qq 和 rr 形成的向量相应地都是逆时针方向，向量 \vec{pq} 
pq
​	
  和 \vec{qr} 
qr
​	
  旋转方向为逆时针，函数 \texttt{cross}(p,q,r)cross(p,q,r) 返回值大于 00。

```python
#r 在pq的右边，那么返回小于0，大于0，那么r就在pq的上方（pr在pq的逆时针方向）
def cross(p: List[int], q: List[int], r: List[int]) -> int:
            return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])
```