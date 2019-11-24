/*
 * @lc app=leetcode id=90 lang=java
 *
 * [90] Subsets II
 */

// @lc code=start
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        Arrays.sort(nums);
        subsetsWithDup(nums,0,used, new ArrayList<>(), res);
        return res;
    }

    private void subsetsWithDup(int[] nums,int s, boolean[] used, List<Integer> comb,ArrayList<List<Integer>> res) {
        res.add(new ArrayList(comb));

        for (int i = s; i < nums.length; i++) {
            if (i > 0 &&used[i - 1] == false && nums[i - 1] == nums[i]) continue;

            comb.add(nums[i]);
            used[i] = true;  // could also use comb.contains(nums[i]) to eliminate the usage of used.
            subsetsWithDup(nums,i + 1, used, comb, res);
            comb.remove(comb.size() -1);
            used[i] = false;

        }
    }
}
// @lc code=end

