/*
 * @lc app=leetcode id=40 lang=java
 *
 * [40] Combination Sum II
 */

// @lc code=start
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum2(candidates,0,new boolean[candidates.length], new ArrayList<>(), res, target);
        return res;
    }

    private void combinationSum2(int[] nums,int s,boolean[] used, List<Integer> comb,ArrayList<List<Integer>> res, int target) {
        if (target == 0) {
            res.add(new ArrayList(comb));
            return;
        }

        if (target < 0) {
            return;
        }
        

        for (int i = s; i < nums.length; i++) {
             if (i > 0 &&used[i - 1] == false && nums[i - 1] == nums[i]) continue;
            comb.add(nums[i]);
            used[i] = true;  
            combinationSum2(nums,i + 1, used,comb, res, target - nums[i]); //use i, not i + 1
            comb.remove(comb.size() -1);
            used[i] = false;

        }
    }
}
// @lc code=end

