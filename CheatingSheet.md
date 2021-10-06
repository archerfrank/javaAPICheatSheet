# Java Cheating Sheet 

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

## Segment Tree



```java
class SegmentTree {

    private static class TreeNode {
        int start, end, sum;
        TreeNode left, right;

        TreeNode(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    private TreeNode buildTree(int[] nums, int start, int end) {
        if (start > end) return null;
        TreeNode cur = new TreeNode(start, end);
        if (start == end) cur.sum = nums[start];
        else {
            int mid = start + (end - start) / 2;
            cur.left = buildTree(nums, start, mid);
            cur.right = buildTree(nums, mid + 1, end);
            cur.sum = cur.left.sum + cur.right.sum;
        }
        return cur;
    }

    private void updateTree(TreeNode node, int i, int val) {
        if (node.start == node.end) {
            node.sum = val;
        } else {
            int mid = node.start + (node.end - node.start) / 2;
            if (i <= mid) updateTree(node.left, i, val);
            else updateTree(node.right, i, val);
            node.sum = node.left.sum + node.right.sum;
        }
    }

    private int queryTree(TreeNode node, int i, int j) {
        if (node.start == i && node.end == j) return node.sum;
        else {
            int mid = node.start + (node.end - node.start) / 2;
            if (j <= mid) {
                return queryTree(node.left, i, j);
            } else if (i >= (mid + 1)) {
                return queryTree(node.right, i, j);
            } else {
                return queryTree(node.left, i, mid) + queryTree(node.right, mid + 1, j);
            }
        }
    }

    private TreeNode root;

    SegmentTree(int[] nums) {
        root = buildTree(nums, 0, nums.length - 1);
    }

    public void update(int i, int val) {
        updateTree(root, i, val);
    }

    public int sumRange(int i, int j) {
        return queryTree(root, i, j);
    }
}
```

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

```python
class BIT:
    def __init__(self, nums: List[int]):
        self.sum = [0] * (len(nums) + 1)
        self.n = len(nums)
        for i in range(len(nums)):
            self.update(i + 1, 0, nums[i])

    def sumRange(self, i: int, j: int) -> int:
        return self.query(j + 1) - self.query(i)

    def lowbit(self, i: int) -> int:
        return i & (-i)
    def update(self, pos:int, oldValue: int, newValue:int):
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

## 最短路径 shortest path

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

```python
class Trie:
    def __init__ (self):
        self.list = defaultdict(Trie)
        self.end = False
        self.cur = self
        self.parent = None
    def insert(self, word):
        for x in word:
            if x not in self.cur.list:
                self.cur.list[x] = Trie()
                self.cur.list[x].parent = self.cur
            self.cur = self.cur.list[x]
        self.cur.end = True
        self.cur = self
    
    def update(self, w):
        if w in self.cur.list:
            self.cur = self.cur.list[w]
            return True
        return False
        
    def remove(self):
        self.cur = self.cur.parent
    
    def reset(self):
        self.cur = self

    def isEnd(self):
        return self.cur.end
```


