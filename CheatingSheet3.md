## DFS 时间戳

https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/solutions/1625899/dfs-shi-jian-chuo-chu-li-shu-shang-wen-t-x1kk/?envType=daily-question&envId=2025-07-24

引入时间戳
怎么知道上述「递」和「归」发生的先后顺序？

在 DFS 一棵树的过程中，维护一个全局的时间戳 clock，每访问一个新的节点，就把 clock 加一。

对于每个节点 x，记录进入这个节点的时间戳 in[x]，和从这个节点往上返回时的时间戳 out[x]。

根据时间戳，分类讨论：

如果 x 是 y 的祖先，那么区间 [in[x],out[x]] 包含区间 [in[y],out[y]]。
如果 y 是 x 的祖先，那么区间 [in[y],out[y]] 包含区间 [in[x],out[x]]。
如果没有区间包含关系，那么 x 和 y 就分别属于两棵不相交的子树。
具体地，当 x

=y 时，如果

in[x]<in[y]≤out[y]≤out[x]
那么 x 是 y 的祖先。其中 out[y]≤out[x] 是因为递归结束的时候，时间戳是不变的，所以可以相等。

由于 in[y]≤out[y] 恒成立，判断方式可以简化为

in[x]<in[y]≤out[x]

```python
clock = 0
        def dfs(x: int, fa: int) -> None:
            nonlocal clock
            clock += 1
            in_[x] = clock  # 递
            xor[x] = nums[x]
            for y in g[x]:
                if y != fa:
                    dfs(y, x)
                    xor[x] ^= xor[y]
            out[x] = clock  # 归
        dfs(0, -1)

        # 判断 x 是否为 y 的祖先
        def is_ancestor(x: int, y: int) -> bool:
            return in_[x] < in_[y] <= out[x]

作者：灵茶山艾府
链接：https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/solutions/1625899/dfs-shi-jian-chuo-chu-li-shu-shang-wen-t-x1kk/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
## 对角线遍历

```python
class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        # 第一排在右上，最后一排在左下
        # 每排从左上到右下
        # 令 k=i-j+n，那么右上角 k=1，左下角 k=m+n-1
        for k in range(1, m + n):
            # 核心：计算 j 的最小值和最大值
            min_j = max(n - k, 0)  # i=0 的时候，j=n-k，但不能是负数
            max_j = min(m + n - 1 - k, n - 1)  # i=m-1 的时候，j=m+n-1-k，但不能超过 n-1
            a = [grid[k + j - n][j] for j in range(min_j, max_j + 1)]  # 根据 k 的定义得 i=k+j-n
            a.sort(reverse = min_j==0)
            for j, val in zip(range(min_j, max_j + 1), a):
                grid[k + j - n][j] = val
        return grid

作者：灵茶山艾府
链接：https://leetcode.cn/problems/sort-matrix-by-diagonals/solutions/3068709/mo-ban-mei-ju-dui-jiao-xian-pythonjavacg-pjxp/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)

        for i in range(n):
            tmp = [grid[i + j][j] for j in range(n - i)]
            tmp.sort(reverse=True)
            for j in range(n - i):
                grid[i + j][j] = tmp[j]

        for j in range(1, n):
            tmp = [grid[i][j + i] for i in range(n - j)]
            tmp.sort()
            for i in range(n - j):
                grid[i][j + i] = tmp[i]

        return grid


作者：力扣官方题解
链接：https://leetcode.cn/problems/sort-matrix-by-diagonals/solutions/3756283/an-dui-jiao-xian-jin-xing-ju-zhen-pai-xu-86ki/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```