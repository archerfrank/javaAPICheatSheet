/*
 * @lc app=leetcode id=39 lang=java
 *
 * [39] Combination Sum
 */

// @lc code=start
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum(candidates,0, new ArrayList<>(), res, target);
        return res;
    }

    private void combinationSum(int[] nums,int s, List<Integer> comb,ArrayList<List<Integer>> res, int target) {
        if (target == 0) {
            res.add(new ArrayList(comb));
            return;
        }

        if (target < 0) {
            return;
        }
        

        for (int i = s; i < nums.length; i++) {
            // if (i > 0 &&used[i - 1] == false && nums[i - 1] == nums[i]) continue;
            comb.add(nums[i]);
            // used[i] = true;  
            combinationSum(nums,i, comb, res, target - nums[i]); //use i, not i + 1
            comb.remove(comb.size() -1);
            // used[i] = false;

        }
    }
}
// @lc code=end

