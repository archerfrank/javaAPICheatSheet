/*
 * @lc app=leetcode id=78 lang=java
 *
 * [78] Subsets
 */

// @lc code=start
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        subsets(nums,0, new ArrayList<>(), res);
        return res;
    }

    private void subsets(int[] nums,int s, List<Integer> comb,ArrayList<List<Integer>> res) {
        res.add(new ArrayList(comb));

        for (int i = s; i < nums.length; i++) {
                comb.add(nums[i]);
                subsets(nums,i + 1, comb, res);
                comb.remove(comb.size() -1);
        }
    }
}
// @lc code=end

