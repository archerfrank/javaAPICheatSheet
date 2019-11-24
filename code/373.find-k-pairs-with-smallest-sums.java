import java.util.PriorityQueue;

/*
 * @lc app=leetcode id=373 lang=java
 *
 * [373] Find K Pairs with Smallest Sums
 */

// @lc code=start
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b)-> (nums1[a[0]] + nums2[a[1]]) - (nums1[b[0]] + nums2[b[1]]));
        List<List<Integer>> res = new ArrayList<>();
        
        if (nums1.length == 0 || nums2.length == 0 || k == 0)
            return res;
        for (int i = 0; i < nums1.length; i++){
            queue.offer(new int[]{i,0});
        }

        for (int j = k; j >0 && !queue.isEmpty(); j--) {
            int[]  cur = queue.poll();
            List<Integer> comb = new ArrayList<>();
            comb.add(nums1[cur[0]]);
            comb.add(nums2[cur[1]]);
            res.add(comb);
            if (cur[1] == nums2.length - 1) continue;
            queue.offer(new int[]{cur[0], cur[1] + 1});
        }
        return res;
    }
}

//https://leetcode.com/problems/find-k-pairs-with-smallest-sums/discuss/84551/simple-Java-O(KlogK)-solution-with-explanation
// @lc code=end

