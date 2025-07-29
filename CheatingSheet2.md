## 前缀和

```python
pre_sum = list(accumulate(pos, initial=0))
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

## 二维差分和使用

一维差分使用
https://leetcode.cn/problems/car-pooling/solutions/2550264/suan-fa-xiao-ke-tang-chai-fen-shu-zu-fu-9d4ra/



```python
        m, n = len(grid), len(grid[0])
        s = [[0] * (n + 1) for _ in range(m + 1)]
        # 2. 计算二维差分
        # 为方便第 3 步的计算，在 d 数组的最上面和最左边各加了一行（列），所以下标要 +1
        d = [[0] * (n + 2) for _ in range(m + 2)]
        for i2 in range(stampHeight, m + 1):
            for j2 in range(stampWidth, n + 1):
                i1 = i2 - stampHeight + 1
                j1 = j2 - stampWidth + 1
                if s[i2][j2] - s[i2][j1 - 1] - s[i1 - 1][j2] + s[i1 - 1][j1 - 1] == 0:
                    d[i1][j1] += 1
                    d[i1][j2 + 1] -= 1
                    d[i2 + 1][j1] -= 1
                    d[i2 + 1][j2 + 1] += 1

        # 3. 还原二维差分矩阵对应的计数矩阵（原地计算）
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                d[i + 1][j + 1] += d[i + 1][j] + d[i][j + 1] - d[i][j]


作者：灵茶山艾府
链接：https://leetcode.cn/problems/stamping-the-grid/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

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

## 数位dp
将 nn 转换成字符串 ss，定义 f(i,mask,isLimit,isNum) 表示从构造 nn 从高到低第 i 位及其之后位的方案数，其余参数的含义为：

* mask 表示前面选过的数字集合，换句话说，第 i 位要选的数字不能在 mask 中。

* isLimit 表示当前是否受到了 nn 的约束。若为真，则第 i 位填入的数字至多为 s[i]，否则可以是 9。

* isNum 表示 i 前面的位数是否填了数字。若为假，则当前位可以跳过（不填数字），或者要填入的数字至少为 1；若为真，则要填入的数字可以从 0 开始。

  这个写法计算的数量是从**[1, n],如果0也算在内，要最后特殊处理一下**。

```
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)
        @cache
        def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:  # 可以跳过当前数位
                res = f(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if mask >> d & 1 == 0:  # d 不在 mask 中
                    res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
            return res
        return f(0, 0, True, False)

作者：endlesscheng
链接：https://leetcode.cn/problems/count-special-integers/solution/shu-wei-dp-mo-ban-by-endlesscheng-xtgx/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


# Given an integer n, return the number of positive integers in the range [1, n] that have at least one repeated digit.

class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
        A = list(map(int, str(n)))
        N = len(A)
        @cache
        def f(i, tight, mask, hasDup):
            if i >= N:
                if hasDup:
                    return 1
                return 0
            upperLimit = A[i] if tight else 9
            ans = 0
            for d in range(upperLimit + 1):
                tight2 = tight and d == upperLimit
                mask2 = mask if mask == 0 and d == 0 else mask | (1 << d)
                hasDup2 = hasDup or (mask & (1 << d))
                ans += f(i + 1, tight2, mask2, hasDup2)
            return ans
        return f(0, True, 0, False)

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/numbers-with-repeated-digits/solution/zhi-shao-you-1-wei-zhong-fu-de-shu-zi-by-0mvu/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

下面是我的写法

```
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)
        @lru_cache(None)
        def dp(i, mask, limit, num):
            if i == len(s):
                return num
            res = 0
            if num == 0:
                res = dp(i + 1, mask, False, 0)

            up = int(s[i]) if limit else 9
            for c in range(1 - num, up + 1):
                if mask >> c & 1 == 0:
                    res += dp(i + 1, mask | (1 << c), c == up and limit, 1)
            return res
        return dp(0, 0, True, 0)
```



## 双指针

```python
	#总开销 是 max(chargeTimes) + k * sum(runningCosts)	 要小于budget
    
    	n = len(chargeTimes)
    	q = deque() # idx (保存当前区间最大值的坐标)
        l, r = 0, 0 # 区间从0， 0 开始
        q.append(0)  # 初始化当前区间最大长度
        ans = 0  #符合条件的最大长度
        total = runningCosts[0] # 当前区间之和
        def check(l, r):
            return chargeTimes[q[0]] + (r - l + 1) * total
            
        #需要时刻保证区间对应的q和total正确， 区间变化是可能r < l,需要对应的处理。r是在加一后处理，
        #l是在减一前处理。当r = l - 1是表示没有数据。
        while l < n:
            while r < l:
                r += 1
                while q and chargeTimes[r] >= chargeTimes[q[-1]]:
                        q.pop()
                q.append(r)
                total += runningCosts[r]
            while r < n and check(l, r) <= budget:
                ans = max(r - l + 1, ans)
                r += 1
                if r < n:
                    while q and chargeTimes[r] >= chargeTimes[q[-1]]:
                        q.pop()
                    q.append(r)
                    total += runningCosts[r]
            #     print("--",l, r, total, q)
            # print(l, r, total, q, ans)
            total -= runningCosts[l]
            if q and q[0] == l:
                q.popleft()
            l += 1
        return ans
    
#下面的更好写， 可以列举右边节点r， 最好r从-1开始，r<l时，表示区间没有数据。
# 区间变化是可能r < l,需要对应的处理。r是在加一后处理，l是在减一前处理。当r = l - 1是表示没有数据。
        q = deque() # idx
        n = len(chargeTimes)
        l, r = 0, -1
        ans = 0
        total = 0
        def check(l, r):
            return chargeTimes[q[0]] + (r - l + 1) * total     
        while r < n:
            while l <= r and check(l, r) > budget:
                total -= runningCosts[l]
                if q and q[0] == l:
                    q.popleft()
                l += 1
            ans = max(r - l + 1, ans)  # 出来循环，表示符合要求。
            r += 1
            if r < n:
                while q and chargeTimes[r] >= chargeTimes[q[-1]]:
                    q.pop()
                q.append(r)
                total += runningCosts[r]
        return ans
    
#也可以这么写，似乎更合理， 循环开始，先处理右节点，然后看看左节点到哪里可以和当前右节点符合条件。最后再r+=1
        q = deque() # idx
        n = len(chargeTimes)
        l, r = 0, 0
        ans = 0
        total = 0
		while r < n:
            while q and chargeTimes[r] >= chargeTimes[q[-1]]:
                q.pop()
            q.append(r)
            total += runningCosts[r]
            while l <= r and check(l, r) > budget:
                total -= runningCosts[l]
                if q and q[0] == l:
                    q.popleft()
                l += 1
            ans = max(r - l + 1, ans)
            r += 1                  
        return ans
```

## 树hash

https://oi-wiki.org/graph/tree-hash/#%E6%96%B9%E6%B3%95%E4%BA%8C

https://leetcode.cn/submissions/detail/359535728/



声明几个质数作为seed和MOD。 使用异或和的方法。childhash*seed + childtreesize，然后对全部child做异或和。

选择两套指数，防止hash碰撞。

```python
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        LEFT_WEIGHT = 419
        RIGHT_WEIGHT = 491
        MOD = 10 ** 9 + 7
        seen = set()
        ans = []
        added = set()
        def dfs(node):
            if not node:
                return 0, 0, 0
            l1, l2,sizel = dfs(node.left)
            r1, r2,sizer = dfs(node.right)
            hash_val1 = node.val + 217;
            hash_val1 = (hash_val1 + ((l1 * LEFT_WEIGHT + sizel) ^ (r1 * RIGHT_WEIGHT + sizer))) % MOD;
            hash_val2 = node.val + 317;
            hash_val2 = (hash_val2 + ((l2 * RIGHT_WEIGHT + sizel) ^ (r2 * LEFT_WEIGHT + sizer))) % MOD;
            tr = (hash_val1, hash_val2)
            if tr in seen:
                if tr not in added:
                    # print(tr)
                    added.add(tr)
                    ans.append(node)
            else:
                seen.add(tr)
            return hash_val1, hash_val2, sizel + sizer + 1
        dfs(root)
        return ans
```



## 两个字符串的最长公共前缀。

下面的例子是一个字符串的，两个字符串同样适用。

```python
        n = len(s)
        lcp = [[0] * (n + 1) for _ in range(n + 1)]  # lcp[i][j] 表示 s[i:] 和 s[j:] 的最长公共前缀
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                if s[i] == s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
```

## 两个字符串的最长子序列，并求出这个子序列。

```python
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        f = [[""] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if str1[i] == str2[j]:
                    dp[i + 1][j  +1] = dp[i][j] + 1
                else:
                    dp[i + 1][j  +1] = max(dp[i+1][j], dp[i][j+1])
        # print(dp)
        rs = ""
        i, j = m - 1, n - 1
        while i >= 0 and j >= 0:
            # print(i, j)
            if str1[i] == str2[j]:
                rs += str1[i]
                i -= 1
                j -= 1
            else:
                if dp[i + 1][j] > dp[i][j + 1]:
                    j -= 1
                else:
                    i -= 1
        ans = rs[::-1]
        
# https://leetcode.cn/submissions/detail/418646702/
```

## 使用广搜将无向树 变成 有向生成树，这样可以用direction数据来dfs，不在需要visit set。
```python
        n = len(nums)
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        direction = [[] for _ in range(n)]
        visit = {0}
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if j not in visit:
                        visit.add(j)
                        nex.append(j)
                        direction[i].append(j)
            stack = nex

作者：liupengsay
链接：https://leetcode.cn/circle/discuss/7RMPmn/view/oleAax/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


## Cycle Detection 环检测

```python
from collections import defaultdict

edges = [(1,3), (4,6), (3,6), (1,4)]
adj = defaultdict(set)
for x, y in edges:
    adj[x].add(y)
    adj[y].add(x)

col = defaultdict(int)
def dfs(x, parent=None):
    if col[x] == 1: return True
    if col[x] == 2: return False
    col[x] = 1
    res = False
    for y in adj[x]:
        if y == parent: continue
        if dfs(y, x): res = True
    col[x] = 2
    return res

for x in adj:
    if dfs(x):
        print "There's a cycle reachable from %d!" % x
```



## 图的二分染色



```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        def dfs(i, c):
            color[i] = c
            for j in g[i]:
                if color[j] == c:
                    return False
                if color[j] == 0 and not dfs(j, 3 - c):
                    return False
            return True

        g = defaultdict(list)
        color = [0] * n
        for a, b in dislikes:
            a, b = a - 1, b - 1
            g[a].append(b)
            g[b].append(a)
        return all(c or dfs(i, 1) for i, c in enumerate(color))

作者：lcbin
链接：https://leetcode.cn/problems/possible-bipartition/solution/by-lcbin-rgi1/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

## 乘法逆元。
当要mod求解是，如果有除法，就要使用乘法逆元的方式。
如果要求a/b % MOD,这里要把a和b 都求出来，除法之后取模结果才对。如果a和b都比较大，就会有溢出的可能。 如果可以求出a%MOD 和 b%MOD的值，就可以是用下面的函数，得到a/b % MOD。

```python
    MOD = 10 ** 9 + 7
    def div(a, b):
        return (a*pow(b,MOD-2,MOD))%MOD;

    s1 = div(s1, s2)
```

https://leetcode.cn/submissions/detail/391018907/

https://leetcode.cn/problems/count-anagrams/solution/ccheng-fa-ni-yuan-chuli-by-thdlrt-74ei/



## 质数相关

```python
@lru_cache(None)
def get_prime_factor(num):
    # 质因数分解
    res = []
    for i in range(2, int(math.sqrt(num)) + 1):
        cnt = 0
        while num % i == 0:
            num //= i
            cnt += 1
        if cnt:
            res.append([i, cnt])
        if i > num:
            break
    if num != 1 or not res:
        res.append([num, 1])
    # 从小到大返回质因数分解以及对应的幂次，注意 1 返回 []
    return res

作者：liupengsay
链接：https://leetcode.cn/circle/discuss/TeTCFl/view/4MKAA4/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回小于等于n的素数
    primes = [True] * (n + 1)  # 范围0到n的列表
    p = 2  # 这是最小的素数
    while p * p <= n:  # 一直筛到sqrt(n)就行了
        if primes[p]:  # 如果没被筛，一定是素数
            for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                primes[i] = False
        p += 1
    primes = [element for element in range(2, n + 1) if primes[element]]  # 得到所有小于等于n的素数
    return primes

def euler_flag_prime(n):
    # 欧拉线性筛素数，返回小于等于n的所有素数
    flag = [False for _ in range(n + 1)]
    prime_numbers = []
    for num in range(2, n + 1):
        if not flag[num]:
            prime_numbers.append(num)
        for prime in prime_numbers:
            if num * prime > n:
                break
            flag[num * prime] = True
            if num % prime == 0:
                break
    return prime_numbers

作者：liupengsay
链接：https://leetcode.cn/circle/discuss/TeTCFl/view/4MKAA4/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


#结合上面求出来primes数组，可以是用下面的方法求质因数。
p = sieve_of_eratosthenes(1000000)   
primes = set(p)
@lru_cache(None)
def get_prime_factor(x):
    ans = Counter()
    while x != 1:
        if x in primes:
            ans[x] += 1
            return ans
        for pri in primes:
            while x % pri == 0:
                ans[pri] += 1
                x //= pri
    return ans

https://leetcode.cn/problems/split-the-array-to-make-coprime-products/

#分解质因数
@lru_cache(None)
def get_prime_factor(x):
    ans = Counter()
    d = 2
    while d * d <= x:  # 分解质因数
        while x % d == 0:
            ans[d] += 1
            x //= d
        d += 1
    if x > 1: ans[x] += 1
    return ans

```






## 二维差分

```python
class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        # 二维差分模板
        diff = [[0] * (n + 2) for _ in range(n + 2)]
        for r1, c1, r2, c2 in queries:
            diff[r1 + 1][c1 + 1] += 1
            diff[r1 + 1][c2 + 2] -= 1
            diff[r2 + 2][c1 + 1] -= 1
            diff[r2 + 2][c2 + 2] += 1

        # 用二维前缀和复原（原地修改）
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                diff[i][j] += diff[i][j - 1] + diff[i - 1][j] - diff[i - 1][j - 1]
        # 保留中间 n*n 的部分，即为答案
        diff = diff[1:-1]
        for i, row in enumerate(diff):
            diff[i] = row[1:-1]
        return diff

作者：endlesscheng
链接：https://leetcode.cn/problems/increment-submatrices-by-one/solution/er-wei-chai-fen-by-endlesscheng-mh0h/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



## 数组中有负数的寻找最长或者最短子数组和的方法。
如果没有负数，可以使用双指针来做。
这个有负数就要用到前缀和，然后用stack或者queue来处理。

最长的用stack, 并且要**反向遍历**   https://leetcode.cn/problems/longest-well-performing-interval/ 

最短的用queue, 并且**正向遍历**   https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/ 

```python
#最长
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        n = len(hours)
        f = [0]
        for x in hours:
            if x > 8:
                f.append(f[-1] + 1)
            else:
                f.append(f[-1] - 1)
        
        ans = 0
        stack = [0]
        for i in range(n + 1):
            if f[i] < f[stack[-1]]:
                stack.append(i)
        # print(f)
        # print(stack)
        for r in range(n, 0, -1):
            while stack and f[stack[-1]] < f[r]:
                ans = max(ans, r - stack[-1])
                stack.pop()
        return ans

#最短

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        preSumArr = [0]
        res = len(nums) + 1
        for num in nums:
            preSumArr.append(preSumArr[-1] + num)
        q = deque()
        for i, curSum in enumerate(preSumArr):
            while q and curSum - preSumArr[q[0]] >= k:
                res = min(res, i - q.popleft())
            while q and preSumArr[q[-1]] >= curSum:
                q.pop()
            q.append(i)
        return res if res < len(nums) + 1 else -1

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/solution/he-zhi-shao-wei-k-de-zui-duan-zi-shu-zu-57ffq/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```


## 二维单调队列，求二维区间最大值



```python
class Solution:
    def largestLocal(self, grid: List[List[int]], h: int, w: int) -> List[List[int]]:
        n = len(grid)
        res = [[0 for j in range(n - h)] for i in range(n - w)] # 创建结果数组
        for i in range(n):
            queue = deque()     # 单调队列
            for j in range(n):
                # 单调队列添加候选值
                while len(queue) != 0 and grid[i][j] >= grid[i][queue[-1]]:
                    queue.pop()
                queue.append(j)
                # 滑动窗口达到大小w可以处理
                if j >=  w - 1 :
                    value = grid[i][queue[0]]   # 获取单调队列中当前行当前滑动窗口中的最大值，即队首索引对应的值
                    for k in range(i - (h - 1), i + 1):   # 更新当前行及其前h-1行该列的最大值
                        if k >= 0 and k < n - h:    # 行必须在结果数组的范围内
                            res[k][j - (w - 1)] = max(res[k][j - (w - 1)],value)
                    if queue[0] <= j - (w - 1):   # 当前最大值位于滑动窗口最左侧，弹出这个最大值,
                        queue.popleft()
        return res

作者：lxk1203
链接：https://leetcode.cn/problems/largest-local-values-in-a-matrix/solution/javapythonmei-ju-mo-ni-dan-diao-dui-lie-fm0pn/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

## 子数组翻转下标

对于子数组 [L,R] 中的任意下标 i，翻转后的下标是 L+R-i（中心对称翻转，两个下标相加恒等于 L+R）。

如果 i 在数组左边界 0 附近，那么翻转时会受到数组左边界的约束，当子数组在最左边时，L=0,R=k-1，i 翻转后是 0+(k-1)-i=k-i-1，所以小于 k-i-1 的点是无法翻转到的；
如果 i 在数组右边界 n-1 附近，那么翻转时会受到数组右边界的约束，当子数组在最右边时，L=n-k,R=n-1，i 翻转后是 (n-k)+(n-1) - i=2n - k - i - 1，所以大于 2n - k - i - 1 的点是无法翻转到的。

所以实际范围为

**[max(i−k+1,k−i−1),min(i+k−1,2n−k−i−1)]**

作者：endlesscheng
链接：https://leetcode.cn/problems/minimum-reverse-operations/solution/liang-chong-zuo-fa-ping-heng-shu-bing-ch-vr0z/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## 集合论与位运算

https://leetcode.cn/circle/discuss/CaOJ45/



本题中用到的位运算技巧：

将元素 xx 变成集合 {x}，即 1 << x。

判断元素 x 是否在集合 A 中，即 ((A >> x) & 1) == 1。

计算两个集合 A,B 的并集 A∩B，即 A | B。例如 110 | 11 = 111。

**计算 A-B**，表示从集合 A 中去掉在集合 B 中的元素，即 **A & ~B**。例如 110 & ~11 = 100。
也可以是用 **(A^B|A) - B**.

全集 U={0,1,⋯,n−1}，即 (1 << n) - 1。

作者：endlesscheng
链接：https://leetcode.cn/problems/smallest-sufficient-team/solution/zhuang-ya-0-1-bei-bao-cha-biao-fa-vs-shu-qode/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## Tarjan 树公共祖先
下面这道题除了寻找公共祖先，还要用上树上差分算法。
```python
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)  # 建树

        qs = [[] for _ in range(n)] #对于节点s，需要查询和节点e的公共祖先。
        for s, e in trips:
            qs[s].append(e)  # 路径端点分组
            if s != e:
                qs[e].append(s)

        # 并查集模板
        pa = list(range(n))
        def find(x: int) -> int:
            if x != pa[x]:
                pa[x] = find(pa[x])
            return pa[x]

        diff = [0] * n
        father = [0] * n
        color = [0] * n
        def tarjan(x: int, fa: int) -> None:
            father[x] = fa
            color[x] = 1  # 递归中
            for y in g[x]:
                if color[y] == 0:  # 未递归
                    tarjan(y, x)
                    pa[y] = x  # 相当于把 y 的子树节点全部 merge 到 x
            for y in qs[x]:
                # color[y] == 2 意味着 y 所在子树已经遍历完
                # 也就意味着 y 已经 merge 到它和 x 的 lca 上了
                # 这里也就是要更新节点x和y的最近公共祖先。
                # 这里更新diff是这道题的要求，其他题目，更新下面if里面的语句做具体的处理。x, y 为两个节点，lca是公共父节点。
                if y == x or color[y] == 2:  # 从 y 向上到达 lca 然后拐弯向下到达 x
                    diff[x] += 1
                    diff[y] += 1
                    lca = find(y) #找到公共祖先。
                    diff[lca] -= 1
                    if father[lca] >= 0:
                        diff[father[lca]] -= 1
            color[x] = 2  # 递归结束
        tarjan(0, -1)

作者：endlesscheng
链接：https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/solution/lei-si-da-jia-jie-she-iii-pythonjavacgo-4k3wq/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```



下面是模板

```python
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)  # 建树

        # 并查集模板
        pa = list(range(n))
        def find(x: int) -> int:
            if x != pa[x]:
                pa[x] = find(pa[x])
            return pa[x]

        color = [0] * n
        def tarjan(x: int, fa: int) -> None:
            father[x] = fa
            color[x] = 1  # 递归中
            for y in g[x]:
                if color[y] == 0:  # 未递归
                    tarjan(y, x)
                    pa[y] = x  # 相当于把 y 的子树节点全部 merge 到 x
            for y in qs[x]:
                # color[y] == 2 意味着 y 所在子树已经遍历完
                # 也就意味着 y 已经 merge 到它和 x 的 lca 上了
                # 这里也就是要更新节点x和y的最近公共祖先。
                if y == x or color[y] == 2:  # 从 y 向上到达 lca 然后拐弯向下到达 x
                    lca = find(y) #找到公共祖先。lca为x和y的公共祖先。
                    ########这里之后放找到两个节点的公共祖先后的处理逻辑。两个节点只会找到一次祖先。
            color[x] = 2  # 递归结束
        tarjan(0, -1)

作者：endlesscheng
链接：https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/solution/lei-si-da-jia-jie-she-iii-pythonjavacgo-4k3wq/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```





## 结合律操作， 或， gcd

操作符合结合律，并且每次结合之后都是继续保持单调递增或者递减。然后求全部子数组中结果等于某个值k或者不同结果的数量，或者等于k的个数，最大最小长度等等，可以使用这个方法。
```python
class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        ors = []  # 按位或的值 + 对应子数组的右端点的最小值
        for i in range(n - 1, -1, -1):
            num = nums[i]
            ors.append([0, i])
            k = 0
            for p in ors:
                p[0] |= num
                if ors[k][0] == p[0]:
                    ors[k][1] = p[1]  # 合并相同值，下标取最小的
                else:
                    k += 1
                    ors[k] = p
            del ors[k + 1:]
            # 本题只用到了 ors[0]，如果题目改成任意给定数值，可以在 ors 中查找
            ans[i] = ors[0][1] - i + 1
        return ans

作者：endlesscheng
链接：https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/solution/by-endlesscheng-zai1/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



#求不同或值的个数。
class Solution:
    def subarrayBitwiseORs(self, arr: List[int]) -> int:
        n = len(arr)
        ors = []  # 按位或的值 + 对应子数组的右端点的最小值
        ans = set()
        for i in range(n):
            num = arr[i]
            ors.append([num, i])
            k = 0
            for p in ors:
                p[0] |= num
                ans.add(p[0])
                if ors[k][0] == p[0]:
                    ors[k][1] = p[1]  # 合并相同值，下标取最小的
                else:
                    k += 1
                    ors[k] = p
            del ors[k + 1:]
            # print(num, ors, ans)
            # 本题只用到了 ors[0]，如果题目改成任意给定数值，可以在 ors 中查找
            
        return len(ans)

https://leetcode.cn/submissions/detail/427440342/ 


#gcd相关的。
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        if gcd(*nums) > 1:
            return -1
        n = len(nums)
        cnt1 = sum(x == 1 for x in nums)
        if cnt1:
            return n - cnt1

        min_size = n
        a = []  # [GCD，相同 GCD 闭区间的右端点]
        for i, x in enumerate(nums):
            a.append([x, i])

            # 原地去重，因为相同的 GCD 都相邻在一起
            j = 0
            for p in a:
                p[0] = gcd(p[0], x)
                if a[j][0] != p[0]:
                    j += 1
                    a[j] = p
                else:
                    a[j][1] = p[1]
            del a[j + 1:]

            if a[0][0] == 1:
                # 这里本来是 i-a[0][1]+1，把 +1 提出来合并到 return 中
                min_size = min(min_size, i - a[0][1])
        return min_size + n - 1

作者：endlesscheng
链接：https://leetcode.cn/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/solution/liang-chong-fang-fa-bao-li-mei-ju-li-yon-refp/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```




## 树上倍增法

Binary Lifting 的本质其实是 dp。dp[node][j] 存储的是 node 节点距离为 2^j 的祖先是谁。

根据定义，dp[node][0] 就是 parent[node]，即 node 的距离为 1 的祖先是 parent[node]。

状态转移是： dp[node][j] = dp[dp[node][j - 1]][j - 1]。

意思是：要想找到 node 的距离 2^j 的祖先，先找到 node 的距离 2^(j - 1) 的祖先，然后，再找这个祖先的距离 2^(j - 1) 的祖先。两步得到 node 的距离为 2^j 的祖先。

所以，我们要找到每一个 node 的距离为 1, 2, 4, 8, 16, 32, ... 的祖先，直到达到树的最大的高度。树的最大的高度是 logn 级别的。

作者：liuyubobobo
链接：https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/solution/li-kou-zai-zhu-jian-ba-acm-mo-ban-ti-ban-shang-lai/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```python
class TreeAncestor:
    def __init__(self, n: int, parent: List[int]):
        m = n.bit_length() - 1
        pa = [[p] + [-1] * m for p in parent]
        for i in range(m):
            for x in range(n):
                if (p := pa[x][i]) != -1:
                    pa[x][i + 1] = pa[p][i]
        self.pa = pa

    def getKthAncestor(self, node: int, k: int) -> int:
        for i in range(k.bit_length()):
            if (k >> i) & 1:  # k 的二进制从低到高第 i 位是 1
                node = self.pa[node][i]
                if node < 0: break
        return node

    # 另一种写法，不断去掉 k 的最低位的 1
    def getKthAncestor2(self, node: int, k: int) -> int:
        while k and node != -1:  # 也可以写成 ~node
            lb = k & -k
            node = self.pa[node][lb.bit_length() - 1]
            k ^= lb
        return node

作者：endlesscheng
链接：https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/solution/mo-ban-jiang-jie-shu-shang-bei-zeng-suan-v3rw/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



下面是使用树上倍增法求树上任意两点的最近公共祖先。

```python
class TreeAncestor:
    def __init__(self, edges: List[List[int]]):
        n = len(edges) + 1
        m = n.bit_length()
        g = [[] for _ in range(n)]
        for x, y in edges:  # 节点编号从 0 开始，默认传入的是无向图。
            g[x].append(y)
            g[y].append(x)

        depth = [0] * n
        pa = [[-1] * m for _ in range(n)]
        def dfs(x: int, fa: int) -> None:
            pa[x][0] = fa
            for y in g[x]:
                if y != fa:
                    depth[y] = depth[x] + 1
                    dfs(y, x)
        dfs(0, -1)  # 以第0个节点为root，这里可以根据题意修改。

        for i in range(m - 1):
            for x in range(n):
                if (p := pa[x][i]) != -1:
                    pa[x][i + 1] = pa[p][i]
        self.depth = depth
        self.pa = pa

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(k.bit_length()):
            if (k >> i) & 1:  # k 二进制从低到高第 i 位是 1
                node = self.pa[node][i]
        return node

    # 返回 x 和 y 的最近公共祖先（节点编号从 0 开始）
    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        # 使 y 和 x 在同一深度
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        if y == x:
            return x
        for i in range(len(self.pa[x]) - 1, -1, -1):
            px, py = self.pa[x][i], self.pa[y][i]
            if px != py:
                x, y = px, py  # 同时上跳 2**i 步
        return self.pa[x][0]

作者：endlesscheng
链接：https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/solution/mo-ban-jiang-jie-shu-shang-bei-zeng-suan-v3rw/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


## 组合数

```python
def C(m,n):
    return comb(m,n)%MOD
```


## 动态规划
有时可以使用线段树，前缀和来优化。
比如，下面两个都是使用前缀和优化的例子。

https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/solutions/2495836/ji-bai-100cong-ji-yi-hua-sou-suo-dao-di-421ab/?envType=daily-question&envId=2023-10-24

https://leetcode.cn/problems/count-of-sub-multisets-with-bounded-sum/description/


## 合并区间代码

log(n) 复杂度

https://leetcode.cn/problems/count-integers-in-intervals/submissions/314289002/?envType=daily-question&envId=2023-12-16

```python
import sortedcontainers

class CountIntervals:

    def __init__(self):
        self.sl = sortedcontainers.SortedList()
        self.ans = 0

    def add(self, left: int, right: int) -> None:
        cur = [left, right]
        idx = self.sl.bisect(cur)
        idx = idx - 1
        if idx < 0:
            idx = 0
        n = len(self.sl)
        while idx < len(self.sl) and self.sl[idx][0] <= cur[1]:
            if self.sl[idx][1] < cur[0]:
                idx += 1
            else:
                l, r = self.sl[idx]
                L = min(l, cur[0])
                R = max(r, cur[1])
                cur = [L, R]
                self.sl.pop(idx)  #有交集的就去掉
                self.ans -= r - l + 1  #也要把对那个的cnt减掉,这句和下面一句代码是题目2276的要求，可以根据题的意思修改
        self.ans += cur[1] - cur[0] + 1 #加入新的区间和cnt,这句和上面一句代码是题目2276的要求，可以根据题的意思修改
        self.sl.add(cur)
        # print(self.sl, self.ans)



                  
```


## 求解数组前k大的子序列的和

https://leetcode.cn/problems/find-the-k-sum-of-an-array/?envType=daily-question&envId=2024-03-09

```python
        arr.sort()
        q = [(arr[0], 0)]
        val = 0
        for i in range(k - 1):
            val, idx = heapq.heappop(q)
            if idx != len(arr) - 1:
                heapq.heappush(q, (val + arr[idx + 1], idx + 1))
                heapq.heappush(q, (val - arr[idx] + arr[idx + 1], idx + 1))
```

## 0-1背包的位运算优化。

可以把0-1背包的动态规划时间复杂度大幅度降低，但是要使用python不限制长度的这个特点。
```python
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        f = 1
        for v in sorted(set(rewardValues)):
            f |= (f & ((1 << v) - 1)) << v
        return f.bit_length() - 1

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-total-reward-using-operations-ii/solutions/2805413/bitset-you-hua-0-1-bei-bao-by-endlessche-m1xn/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

## 分组背包

https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/solutions/1371872/zhuan-hua-cheng-fen-zu-bei-bao-pythongoc-3xnk/


```python
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        dp = [0] * (k + 1)
        for p in piles:
            m = min(len(p), k)
            for i in range(k, -1, -1):
                sumn = 0
                for j in range(1, m + 1):
                    sumn += p[j - 1]
                    if i - j >= 0:
                        dp[i] = max(dp[i], dp[i - j] + sumn)
                
        return dp[k]
```

## 欧拉回路， 一笔画

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        def dfs(curr: str):
            while vec[curr]:
                tmp = heapq.heappop(vec[curr])
                dfs(tmp)
            stack.append(curr)

        vec = collections.defaultdict(list)
        for depart, arrive in tickets:
            vec[depart].append(arrive)
        for key in vec:
            heapq.heapify(vec[key])
        
        stack = list()
        dfs("JFK")
        return stack[::-1]

作者：力扣官方题解
链接：https://leetcode.cn/problems/reconstruct-itinerary/solutions/389885/zhong-xin-an-pai-xing-cheng-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

## 子数组个数和logTrick and or gcd
https://leetcode.cn/problems/find-subarray-with-bitwise-or-closest-to-k/solutions/2798206/li-yong-and-de-xing-zhi-pythonjavacgo-by-gg4d/

数组里面的值nums[j] 就相当于从nums[j] ~ nums[i]的或的值。
```python
for i, x in enumerate(nums):
    j = i - 1
    while j >= 0 and nums[j] | x != nums[j]:
        nums[j] |= x
        j -= 1

时间复杂度：O(nlogU)，其中 n 是 nums 的长度，U=max(nums)。由于 2^29 −1<10^9 <2^30 −1，二进制数对应集合的大小不会超过 29，因此在 OR 运算下，每个数字至多可以增大 29 次。总体上看，二重循环的总循环次数等于每个数字可以增大的次数之和，即 O(nlogU)。



下面的写法也可以。

for i, x in enumerate(nums):
    for j in range(i - 1, -1, -1):
        if nums[j] & x == nums[j]:
            break
        nums[j] &= x

作者：灵茶山艾府
链接：https://leetcode.cn/problems/number-of-subarrays-with-and-value-of-k/solutions/2833497/jian-ji-xie-fa-o1-kong-jian-pythonjavacg-u7fv/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


# GCD

for i, x in enumerate(arr):
    for j in range(i - 1, -1, -1):
        if gcd(arr[j], x) == arr[j]:
            break
        arr[j] = gcd(arr[j], x)
```

## 旋转矩阵

```
 # 顺时针旋转矩阵 90°
    def rotate(self, a: List[List[int]]) -> List[List[int]]:
        return list(zip(*reversed(a)))

作者：灵茶山艾府
链接：https://leetcode.cn/problems/find-the-minimum-area-to-cover-all-ones-ii/solutions/2819357/mei-ju-pythonjavacgo-by-endlesscheng-uu5p/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

## 换根dp

有时候需要维护换根后root 和father 节点的状态。下面的例子就是维护每个节点的最大和第二大的子结果，来完成换根。
每个节点的最大值，就是max(当前节点的子结果最大值, 到父节点的值)
https://leetcode.cn/problems/time-taken-to-mark-all-nodes/solutions/2868276/di-er-lei-huan-gen-dppythonjavacgo-by-en-411w/


# 最长上升子序列 二维

参考

https://leetcode.cn/problems/russian-doll-envelopes/solutions/633231/e-luo-si-tao-wa-xin-feng-wen-ti-by-leetc-wj68/

```

class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        
        n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0], -x[1]))

        f = [envelopes[0][1]]
        for i in range(1, n):
            num = envelopes[i][1]
            if num > f[-1]:
                f.append(num)
            else:
                index = bisect.bisect_left(f, num)
                f[index] = num
        
        return len(f)

作者：力扣官方题解
链接：https://leetcode.cn/problems/russian-doll-envelopes/solutions/633231/e-luo-si-tao-wa-xin-feng-wen-ti-by-leetc-wj68/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```


## 先单调下降，再单调上升的 二分查找。

具体看题解。

https://leetcode.cn/problems/super-egg-drop/solutions/197163/ji-dan-diao-luo-by-leetcode-solution-2/

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        f = [i for i in range(n + 1)]
        # print(f)
        for i in range(2, k + 1):
            g = [inf] * (n + 1)
            g[0] = 0
            a ,b = 0, 0
            for j in range(1, n + 1):
                lo = 1
                hi = j - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if g[mid - 1] < f[j - mid]:
                        lo = mid + 1
                    else:
                        hi = mid
                g[j] = min(g[j], max(g[lo - 1], f[j - lo]) + 1)
                if lo > 1:
                    g[j] = min(g[j], max(g[lo - 2], f[j - lo + 1]) + 1)
                # for x in range(1, j + 1):
                #     g[j] = min(g[j], max(g[x - 1], f[j - x]) + 1)
            f = g
            # print(f)
        return f[n]

```

## 排列组合 球放盒子的问题。

定理：从 k 个不同元素中，选奇数个数，有 2 **（k−1） 种选法

https://blog.csdn.net/qq_43337119/article/details/137109881


https://leetcode.cn/problems/find-the-number-of-possible-ways-for-an-event/solutions/2948578/zu-he-shu-xue-di-er-lei-si-te-lin-shu-py-e6sv/


组合数的算法，用下面第二个，更快。
```python
N = 1000
combs = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(0, N + 1):
    combs[i][0] = combs[i][i] = 1
for i in range(1, N + 1):
    for j in range(1, i):
        combs[i][j] = combs[i - 1][j] + combs[i - 1][j - 1]




MOD = 1_000_000_007
MX = 100_000

f = [0] * MX  # f[i] = i!
f[0] = 1
for i in range(1, MX):
    f[i] = f[i - 1] * i % MOD

inv_f = [0] * MX  # inv_f[i] = i!^-1
inv_f[-1] = pow(f[-1], -1, MOD)
for i in range(MX - 1, 0, -1):
    inv_f[i - 1] = inv_f[i] * i % MOD

def comb(n: int, m: int) -> int:
    return f[n] * inv_f[m] * inv_f[n - m] % MOD

作者：灵茶山艾府
链接：https://leetcode.cn/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/solutions/3033292/chun-shu-xue-ti-pythonjavacgo-by-endless-mxj7/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```
如果mod不是质数，看下面这个。

https://leetcode.cn/problems/check-if-digits-are-equal-in-string-after-operations-ii/solutions/3086169/mo-shu-wei-he-shu-shi-de-zu-he-shu-by-en-8x7t/

## 树上滑动窗口

https://leetcode.cn/problems/longest-special-path/solutions/3051377/shu-shang-hua-chuang-pythonjavacgo-by-en-rh5m/

