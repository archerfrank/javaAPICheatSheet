/*
 * @lc app=leetcode id=47 lang=java
 *
 * [47] Permutations II
 */

// @lc code=start
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        permuteUnique(nums, new ArrayList<>(), res,new boolean[nums.length]);
        return res;
    }

    private void permuteUnique(int[] nums,List<Integer> comb,ArrayList<List<Integer>> res, boolean[] used) {
        if (comb.size() == nums.length) {
            res.add(new ArrayList(comb));
            return;
        }
        

        for (int i = 0; i < nums.length; i++) {
            if(used[i] || i > 0 && nums[i - 1] == nums[i] && used[i - 1]==false) continue;
            comb.add(nums[i]);
            used[i] = true;
            permuteUnique(nums, comb, res, used);
            comb.remove(comb.size() -1);
            used[i] = false;  
        }
    }
}
// @lc code=end

