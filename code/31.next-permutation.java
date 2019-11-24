/*
 * @lc app=leetcode id=31 lang=java
 *
 * [31] Next Permutation
 */

// @lc code=start
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 1;
        while(i > 0 && nums[i - 1] >= nums[i]) {
            i = i -1;
        }

        if (i == 0) {
            reverse(nums, i, nums.length - 1);
        } else {
            int j = nums.length - 1;
            while(nums[j] <= nums[i - 1] && j > 0) j--;
            System.out.println(j);
            swap(nums,j,i - 1);
            reverse(nums, i, nums.length - 1);
        }
    }
    public void swap(int[] A, int i, int j) {
        int tmp = A[i];
        A[i] = A[j];
        A[j] = tmp;
    }

    private void reverse (int [] a, int i, int j) {
        while(i < j) swap(a, i++, j--);
    }
}
// @lc code=end

