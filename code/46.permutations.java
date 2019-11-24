/*
 * @lc app=leetcode id=46 lang=java
 *
 * [46] Permutations
 */

// @lc code=start
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        permute(nums,used, new ArrayList<>(), res);
        return res;
    }
    private void permute(int[] nums,boolean[] used, List<Integer> comb,ArrayList<List<Integer>> res) {
        if (comb.size() == nums.length) {
            res.add(new ArrayList(comb));
            return;
        }

        for (int i =0; i < nums.length; i++) {
            if (used[i] == false) {
                comb.add(nums[i]);
                used[i] = true;
                permute(nums,used, comb, res);
                comb.remove(comb.size() -1);
                used[i] = false;
            }
        }
    }
}
// @lc code=end

