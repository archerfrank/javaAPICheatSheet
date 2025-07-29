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