###  固定开点

不带2的和带2的不能混用。

```python
class Seg:
    def __init__(self, nums):
        self.n = len(nums)
        self.arr = [0] * (self.n * 4)
        self.base = nums
        self.build(1, 0, self.n - 1)
        # self.build2()
    #p, l, r总是一起出现， p代表线段数组的位置， l，r代表在原数组的位置
    def build(self, p, l, r):
        if l == r:
            self.arr[p] = self.base[l]
            return
        mid = (l + r) // 2
        self.build(p * 2, l , mid)
        self.build(p * 2 + 1, mid + 1, r)
        self.push_up(p, mid, l, r)
        # print(self.arr)
    # 下面三个带2的函数是一组，不能和不带2的混用。
    def build2(self):
        for i in range(self.n):
            p = i + self.n
            self.arr[p] = self.base[i]
        for i in range(self.n - 1, 0, -1):
            self.arr[i] = self.arr[i * 2] + self.arr[i*2+1]
        # print(self.arr)
        
    def update2(self, i, val):
        p = i + self.n
        self.arr[p] = val
        while p > 1:
            self.arr[p >> 1] = self.arr[p] + self.arr[p ^ 1]
            p >>= 1
        # print(self.arr)

    def query2(self, l, r):
        l += self.n
        r += self.n
        ans = 0
        while l <= r:
            if (l % 2) == 1:
                ans += self.arr[l]
                l += 1
            if (r % 2) == 0:
                ans += self.arr[r]
                r-=1
            l //= 2
            r //= 2
        return ans

    def update(self, p, diff, l, r, i) -> None: #传进来就是差值了。 #p = 1, l, r表示当前节点的位置一般传入0， len(nums) - 1
        if l == r:
            self.arr[p] += diff
            return
        mid = (l + r) // 2
        if i <= mid:
            self.update(p * 2, diff, l, mid, i)
        else:
            self.update(p * 2 + 1, diff, mid + 1, r, i)
        self.push_up(p, mid, l, r)

    def query(self, p, l, r, i, j) -> int: #p = 1, l, r表示当前节点的位置一般传入0， len(nums) - 1，i，j，表示查询的范围
        if l == r:
            return self.arr[p]
        if l == i and r == j:
            return self.arr[p]
        mid = (l + r) // 2
        if j <= mid:
            return self.query(p*2, l, mid, i, j)
        elif i > mid:
            return self.query(p * 2 + 1, mid + 1, r, i, j)
        else:
            return self.query(p *2, l, mid, i, mid) + self.query(p*2+1, mid + 1,r, mid + 1, j)

    def push_up(self, p, mid, l, r): # 有些情况下mid, l, r 可能会用到。
        self.arr[p] = self.arr[p * 2] + self.arr[p * 2 + 1]
```

Python 动态创建节点。区间查询的例子。

https://leetcode.cn/problems/count-integers-in-intervals/

```python
class Node:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.val = 0
        self.left = None
        self.right = None
class Seg:
    def __init__(self, s, e):
        self.root = Node(s, e)
        # self.build2()
    def query(self):
        return self.root.val

    def update(self, p, l, r) -> None: #只需要判断有没有交集
        if p.start > r or p.end < l:
            return
        if p.start >= l and p.end <= r:
            p.val = p.end - p.start + 1
            return
        mid = (p.start + p.end) // 2
        if not p.left:
            p.left = Node(p.start,mid)
        self.update(p.left, l, r)
        if not p.right:
            p.right = Node(mid+1, p.end)
        self.update(p.right, l, r)
        self.push_up(p, mid, l, r)

    def push_up(self, p, mid, l, r): # 有些情况下mid, l, r 可能会用到。
        p.val = max(p.left.val + p.right.val, p.val)
        # print(self.arr[p], l, r)
```
### 动态开点

下面是动态开点，区间赋值，然后求区间最大值的例子。
```python
class Node:
    def __init__(self, l, r):
        self.left = None
        self.right = None
        self.l = l
        self.r = r
        self.mid = (l + r) >> 1
        self.v = 0
        self.add = 0


class SegmentTree:
    def __init__(self):
        self.root = Node(1, int(1e9))

    def modify(self, l, r, v, node=None):
        if l > r:
            return
        if node is None:
            node = self.root
        if node.l >= l and node.r <= r:
            node.v = v   ## todo
            node.add = v  ## todo
            return
        self.pushdown(node)
        if l <= node.mid:
            self.modify(l, r, v, node.left)
        if r > node.mid:
            self.modify(l, r, v, node.right)
        self.pushup(node)

    def query(self, l, r, node=None):
        if l > r:
            return 0
        if node is None:
            node = self.root
        if node.l >= l and node.r <= r:
            return node.v
        self.pushdown(node)
        v = 0
        if l <= node.mid:
            v = max(v, self.query(l, r, node.left))  ## todo
        if r > node.mid:
            v = max(v, self.query(l, r, node.right))  ## todo
        return v

    def pushup(self, node):
        node.v = max(node.left.v, node.right.v)   ## todo

    def pushdown(self, node):
        if node.left is None:
            node.left = Node(node.l, node.mid)
        if node.right is None:
            node.right = Node(node.mid + 1, node.r)
        if node.add:  ## todo
            node.left.v = node.add
            node.right.v = node.add
            node.left.add = node.add
            node.right.add = node.add
            node.add = 0

作者：lcbin
链接：https://leetcode.cn/problems/falling-squares/solution/-by-lcbin-5rop/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

下面是动态开点，区间赋值，然后区间求和的例子

```python
class Node:
    def __init__(self, l, r):
        self.left = None
        self.right = None
        self.l = l
        self.r = r
        self.mid = (l + r) >> 1
        self.v = 0
        self.add = 0


class SegmentTree:
    def __init__(self):
        self.root = Node(1, int(1e9))

    def modify(self, l, r, v, node=None):
        if l > r:
            return
        if node is None:
            node = self.root
        if node.l >= l and node.r <= r:
            node.v = v * (node.r - node.l + 1)
            node.add = v
            # print(node.l, node.r, node.v, node.add)
            return
        self.pushdown(node)
        if l <= node.mid:
            self.modify(l, r, v, node.left)
        if r > node.mid:
            self.modify(l, r, v, node.right)
        self.pushup(node)

    def query(self, l, r, node=None):
        if l > r:
            return 0
        if node is None:
            node = self.root
        if node.l >= l and node.r <= r:
            print(node.l, node.r, node.v)
            return node.v
        self.pushdown(node)
        v = 0
        if l <= node.mid:
            v += self.query(l, r, node.left)
        if r > node.mid:
            v += self.query(l, r, node.right)
        # print(node.l, node.r, v)
        return v

    def pushup(self, node):
        node.v = node.left.v + node.right.v

    def pushdown(self, node):
        if node.left is None:
            node.left = Node(node.l, node.mid)
        if node.right is None:
            node.right = Node(node.mid + 1, node.r)
        if node.add:
            node.left.v = node.add * (node.left.r - node.left.l + 1)
            node.right.v = node.add * (node.right.r - node.right.l + 1)
            node.left.add = node.add
            node.right.add = node.add
            node.add = 0
```

动态开点，区间叠加， 然后区间求和。

```python
class Node:
    def __init__(self, l, r):
        self.left = None
        self.right = None
        self.l = l
        self.r = r
        self.mid = (l + r) >> 1
        self.v = 0
        self.add = 0


class SegmentTree:
    def __init__(self):
        self.root = Node(1, int(1e9))

    def modify(self, l, r, v, node=None):
        if l > r:
            return
        if node is None:
            node = self.root
        if node.l >= l and node.r <= r:
            node.v += v * (node.r - node.l + 1)
            node.add += v
            # print(node.l, node.r, node.v, node.add)
            return
        self.pushdown(node)
        if l <= node.mid:
            self.modify(l, r, v, node.left)
        if r > node.mid:
            self.modify(l, r, v, node.right)
        self.pushup(node)

    def query(self, l, r, node=None):
        if l > r:
            return 0
        if node is None:
            node = self.root
        if node.l >= l and node.r <= r:
            # print(node.l, node.r, node.v)
            return node.v
        self.pushdown(node)
        v = 0
        if l <= node.mid:
            v += self.query(l, r, node.left)
        if r > node.mid:
            v += self.query(l, r, node.right)
        # print(node.l, node.r, v)
        return v

    def pushup(self, node):
        node.v = node.left.v + node.right.v

    def pushdown(self, node):
        if node.left is None:
            node.left = Node(node.l, node.mid)
        if node.right is None:
            node.right = Node(node.mid + 1, node.r)
        if node.add:
            node.left.v += node.add * (node.left.r- node.left.l + 1)
            node.right.v += node.add * (node.right.r- node.right.l + 1)
            node.left.add += node.add
            node.right.add += node.add
            node.add = 0
```

Python 固定开点，模板, 区间赋值，求最大值。
```python
    class Solution:
    def lengthOfLIS(self, nums: List[int], k: int) -> int:
        n = 100001  #x线段树的区间，这里是1 - 100000。如果查询区间有零，那么update和query的时候都要加一，做一个映射。
        node = [0] * (4 * n)
        todo = [0] * (4 * n)

        def pushup(o: int) -> None:
            node[o] = max(node[o * 2], node[o * 2 + 1])

        def do(o: int, l: int, r: int, V:int) -> None:
            node[o] = V
            todo[o] = V

        # 初始化线段树   o,l,r=1,1,n
        def build(o: int, l: int, r: int) -> None:
            if l == r:
                node[o] = 0
                return
            m = (l + r) // 2
            build(o * 2, l, m)
            build(o * 2 + 1, m + 1, r)
            pushup(o)

        def pushdown(o:int, l:int, r:int):
            # print(o, l, r)
            if todo[o]:
                do(o * 2, l, m, todo[o])
                do(o * 2 + 1, m + 1, r, todo[o])
                todo[o] = 0

        # 修改区间 [L,R]   o,l,r=1,1,n
        def update(o: int, l: int, r: int, L: int, R: int, V:int) -> None:
            if L <= l and r <= R:
                do(o, l, r, V)
                return
            pushdown(o, l, r)
            m = (l + r) // 2
            if m >= L: update(o * 2, l, m, L, R, V)
            if m < R: update(o * 2 + 1, m + 1, r, L, R, V)
            pushup(o)

        # 查询区间 [L,R]   o,l,r=1,1,n
        def query(o, l, r, L, R) -> int: #o, l, r表示当前节点的位置，L，R，表示查询的范围
            if L <= l and r <= R:
                return node[o]
            pushdown(o, l, r)
            m = (l + r) // 2
            ans = 0
            if m >= L: ans = max(query(o * 2, l, m, L, R), ans)
            if m < R: ans = max(query(o * 2 + 1, m + 1, r, L, R), ans)
            return ans
        
        ans = 1
        build(1, 1, n)
        for x in nums:
            if x == 1:  #1 的时候需要特别注意。
                update(1,1,n,1,1,1)
                continue
            L, R = max(1, x - k), x - 1  # 注意因为线段树是从1开始，所以L不能为0
            m = query(1, 1, n, L, R)
            ans = max(ans, m + 1)
            # print(x, m, ans, L, R)
            update(1,1,n,x,x,m+1)
        return ans
#https://leetcode.cn/problems/longest-increasing-subsequence-ii/ 
#https://leetcode.cn/problems/handling-sum-queries-after-update/
```

java动态开点。https://leetcode.cn/problems/count-integers-in-intervals/submissions/ 这样不用先build出来整个树，动态的添加节点，防止内存被爆。

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



java 动态开点，更抽象实现

```java
class Solution {
    int N = (int)1e9;
    class Node {
        // ls 和 rs 分别代表当前区间的左右子节点
        Node ls, rs;
        // val 代表当前区间的最大高度，add 为懒标记
        int val, add;
    }
    Node root = new Node();
    void update(Node node, int lc, int rc, int l, int r, int v) {
        if (l <= lc && rc <= r) {
            node.add = v;  // to do  可以参考上面python看不同的实现要怎么改。
            node.val = v;  // to do
            return ;
        }
        pushdown(node);
        int mid = lc + rc >> 1;
        if (l <= mid) update(node.ls, lc, mid, l, r, v);
        if (r > mid) update(node.rs, mid + 1, rc, l, r, v);
        pushup(node);
    }
    int query(Node node, int lc, int rc, int l, int r) {
        if (l <= lc && rc <= r) return node.val;
        pushdown(node);
        int mid = lc + rc >> 1, ans = 0;
        if (l <= mid) ans = query(node.ls, lc, mid, l, r);
        if (r > mid) ans = Math.max(ans, query(node.rs, mid + 1, rc, l, r));
        return ans;
    }
    void pushdown(Node node) {
        if (node.ls == null) node.ls = new Node();
        if (node.rs == null) node.rs = new Node();
        if (node.add == 0) return ;
        node.ls.add = node.add;   // to do
        node.rs.add = node.add;   // to do
        node.ls.val = node.add; // to do
        node.rs.val = node.add;// to do
        node.add = 0;
    }
    void pushup(Node node) {
        node.val = Math.max(node.ls.val, node.rs.val);  // to do
    }
    public List<Integer> fallingSquares(int[][] ps) {
        List<Integer> ans = new ArrayList<>();
        for (int[] info : ps) {
            int x = info[0], h = info[1], cur = query(root, 0, N, x, x + h - 1);
            update(root, 0, N, x, x + h - 1, cur + h);
            ans.add(root.val);
        }
        return ans;
    }
}

// 作者：AC_OIer
// 链接：https://leetcode.cn/problems/falling-squares/solution/by-ac_oier-zpf0/
// 来源：力扣（LeetCode）
// 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

//https://leetcode.cn/submissions/detail/404505941/
            // int L = Math.max(1, x - k);
            // int R = x - 1;
            // int m = query(root, 1, N, L, R);
            // ans = Math.max(ans, m + 1);
            // update(root, 1, N, x, x, m + 1);
```

java 动态开点，区间叠加， 然后区间求和。

```java
/**
 * @Description: 线段树（动态开点）
 * @Author: LFool
 * @Date 2022/6/7 09:15
 **/
public class SegmentTreeDynamic {
    class Node {
        Node left, right;
        int val, add;
    }
    public int N = (int) 1e9;
    public Node root = new Node();
    public void update(Node node, int start, int end, int l, int r, int val) {
        if (l <= start && end <= r) {
            node.val += (end - start + 1) * val;
            node.add += val;
            return ;
        }
        int mid = (start + end) >> 1;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) update(node.left, start, mid, l, r, val);
        if (r > mid) update(node.right, mid + 1, end, l, r, val);
        pushUp(node);
    }
    public int query(Node node, int start, int end, int l, int r) {
        if (l <= start && end <= r) return node.val;
        int mid = (start + end) >> 1, ans = 0;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) ans += query(node.left, start, mid, l, r);
        if (r > mid) ans += query(node.right, mid + 1, end, l, r);
        return ans;
    }
    private void pushUp(Node node) {
        node.val = node.left.val + node.right.val;
    }
    private void pushDown(Node node, int leftNum, int rightNum) {
        if (node.left == null) node.left = new Node();
        if (node.right == null) node.right = new Node();
        if (node.add == 0) return ;
        node.left.val += node.add * leftNum;
        node.right.val += node.add * rightNum;
        // 对区间进行「加减」的更新操作，下推懒惰标记时需要累加起来，不能直接覆盖
        node.left.add += node.add;
        node.right.add += node.add;
        node.add = 0;
    }
}

作者：lfool
链接：https://leetcode.cn/problems/range-module/solution/by-lfool-eo50/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    
this.seg.query(this.seg.root, 0, this.seg.N, start, end);
this.seg.update(this.seg.root, 0, this.seg.N, start, end, 1);
```

区间叠加，求最大值

```java
public class SegmentTreeDynamic {
    class Node {
        Node left, right;
        int val, add;
    }
    public int N = (int) 1e9;
    public Node root = new Node();
    public void update(Node node, int start, int end, int l, int r, int val) {
        if (l <= start && end <= r) {
            node.val += val;
            node.add += val;
            return ;
        }
        int mid = (start + end) >> 1;
        pushDown(node);
        if (l <= mid) update(node.left, start, mid, l, r, val);
        if (r > mid) update(node.right, mid + 1, end, l, r, val);
        pushUp(node);
    }
    public int query(Node node, int start, int end, int l, int r) {
        if (l <= start && end <= r) return node.val;
        int mid = (start + end) >> 1, ans = 0;
        pushDown(node);
        if (l <= mid) ans = Math.max(query(node.left, start, mid, l, r), ans);
        if (r > mid) ans = Math.max(query(node.right, mid + 1, end, l, r), ans);
        return ans;
    }
    private void pushUp(Node node) {
        node.val = Math.max(node.left.val, node.right.val);
    }
    private void pushDown(Node node) {
        if (node.left == null) node.left = new Node();
        if (node.right == null) node.right = new Node();
        if (node.add == 0) return ;
        node.left.val += node.add;
        node.right.val += node.add;
        node.left.add += node.add;
        node.right.add += node.add;
        node.add = 0;
    }
}
```
另外一个区间增，然后求和的的板子。

https://leetcode.cn/problems/subarrays-distinct-element-sum-of-squares-ii/solutions/2504870/javaqu-jian-zeng-ban-zi-by-topcloser-j1qq/



java 动态开点，区间赋值， 然后区间求和。

```java
public class SegmentTreeDynamic {
    class Node {
        Node left, right;
        int val, add;
    }
    public int N = (int) 1e9;
    public Node root = new Node();
    public void update(Node node, int start, int end, int l, int r, int val) {
        if (l <= start && end <= r) {
            node.val = (end - start + 1) * val;
            node.add = val;
            return ;
        }
        int mid = (start + end) >> 1;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) update(node.left, start, mid, l, r, val);
        if (r > mid) update(node.right, mid + 1, end, l, r, val);
        pushUp(node);
    }
    public int query(Node node, int start, int end, int l, int r) {
        if (l <= start && end <= r) return node.val;
        int mid = (start + end) >> 1, ans = 0;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) ans += query(node.left, start, mid, l, r);
        if (r > mid) ans += query(node.right, mid + 1, end, l, r);
        return ans;
    }
    private void pushUp(Node node) {
        node.val = node.left.val + node.right.val;
    }
    private void pushDown(Node node, int leftNum, int rightNum) {
        if (node.left == null) node.left = new Node();
        if (node.right == null) node.right = new Node();
        if (node.add == 0) return ;
        node.left.val = node.add * leftNum;
        node.right.val = node.add * rightNum;
        // 对区间进行「加减」的更新操作，下推懒惰标记时需要累加起来，不能直接覆盖
        node.left.add = node.add;
        node.right.add = node.add;
        node.add = 0;
    }
}
```

java 动态开点，区间赋值， 然后求最大值。

```java
public class SegmentTreeDynamic {
    class Node {
        Node left, right;
        int val, add;
    }
    public int N = (int) 1e9;
    public Node root = new Node();
    public void update(Node node, int start, int end, int l, int r, int val) {
        if (l <= start && end <= r) {
            node.val = val;
            node.add = val;
            return ;
        }
        int mid = (start + end) >> 1;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) update(node.left, start, mid, l, r, val);
        if (r > mid) update(node.right, mid + 1, end, l, r, val);
        pushUp(node);
    }
    public int query(Node node, int start, int end, int l, int r) {
        if (l <= start && end <= r) return node.val;
        int mid = (start + end) >> 1, ans = Integer.MIN_VALUE;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) ans = Math.max(query(node.left, start, mid, l, r), ans);
        if (r > mid) ans = Math.max(query(node.right, mid + 1, end, l, r), ans);
        return ans;
    }
    private void pushUp(Node node) {
        node.val = Math.max(node.left.val , node.right.val);
    }
    private void pushDown(Node node, int leftNum, int rightNum) {
        if (node.left == null) node.left = new Node();
        if (node.right == null) node.right = new Node();
        if (node.add == 0) return ;
        node.left.val = node.add;
        node.right.val = node.add;
        node.left.add = node.add;
        node.right.add = node.add;
        node.add = 0;
    }
}
```



## 线段树二分查找

```python

https://leetcode.cn/problems/fruits-into-baskets-iii/solutions/3603049/xian-duan-shu-er-fen-pythonjavacgo-by-en-ssqf/

class SegmentTree:
    def __init__(self, a: List[int]):
        n = len(a)
        self.max = [0] * (2 << (n - 1).bit_length())  # 注意线段树数组的大小
        self.build(a, 1, 0, n - 1)

    def maintain(self, o: int):
        self.max[o] = max(self.max[o * 2], self.max[o * 2 + 1])

    # 初始化线段树
    def build(self, a: List[int], o: int, l: int, r: int):
        if l == r:
            self.max[o] = a[l]
            return
        m = (l + r) // 2
        self.build(a, o * 2, l, m)
        self.build(a, o * 2 + 1, m + 1, r)
        self.maintain(o)
    
    # 初始化线段树 
    # o 线段树数组的位置，l，r 代表原数组的区间， p 原数组位置p要修改为x
    def update(self, o: int, l: int, r: int , x: int, p: int):
        if l == r:
            self.max[o] = x
            return
        mid = (l + r) // 2
        if p <= mid:
            self.update(o * 2, l, mid, x, p)
        else:
            self.update(o * 2 + 1, mid + 1, r, x, p)
        self.maintain(o)
        

    # 找区间内的第一个 >= x 的数，返回这个数的下标（没有则返回 -1）
    def find(self, o: int, l: int, r: int, x: int) -> int:
        if self.max[o] < x:  # 区间没有 >= x 的数
            return -1
        if l == r:
            return l
        m = (l + r) // 2
        i = self.find(o * 2, l, m, x)  # 先递归左子树
        if i < 0:  # 左子树没找到
            i = self.find(o * 2 + 1, m + 1, r, x)  # 再递归右子树
        return i
    
    # 也可以使用这个方法直接update  找区间内的第一个 >= x 的数，并更新为 -1，返回这个数的下标（没有则返回 -1）
    def find_first_and_update(self, o: int, l: int, r: int, x: int) -> int:
        if self.max[o] < x:  # 区间没有 >= x 的数
            return -1
        if l == r:
            self.max[o] = -1  # 更新为 -1，表示不能放水果
            return l
        m = (l + r) // 2
        i = self.find_first_and_update(o * 2, l, m, x)  # 先递归左子树
        if i < 0:  # 左子树没找到
            i = self.find_first_and_update(o * 2 + 1, m + 1, r, x)  # 再递归右子树
        self.maintain(o)
        return i

作者：灵茶山艾府
链接：https://leetcode.cn/problems/fruits-into-baskets-iii/solutions/3603049/xian-duan-shu-er-fen-pythonjavacgo-by-en-ssqf/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        t = SegmentTree(baskets)
        n = len(baskets)
        ans = 0
        for x in fruits:
            v = t.find(1, 0, n - 1, x)
            if v < 0:
                ans += 1
            else:
                t.update(1,0, n - 1, 0, v)
        return ans

```