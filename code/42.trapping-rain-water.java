/*
 * @lc app=leetcode id=42 lang=java
 *
 * [42] Trapping Rain Water
 */

// @lc code=start
class Solution {
    public int trap(int[] height) {
        if (height ==null || height.length ==0) return 0;
        int max = 0;
        int maxHindex = 0;
        int total = 0;
        for (int i = 0; i < height.length; i++) { // find the last maxmium
            if (height[i] > max) {
                max = height[i];
                maxHindex = i;
            }
        }
        System.out.println(maxHindex);
        int j =0;
        for(int i = 0; i <= maxHindex; i++) { // front to max
            if (height[i] != 0) {
                if (height[j] != 0 && height[i] >= height[j]) {
                    int unit = Math.min(height[j], height[i]);
                    System.out.println(unit + ",," + j + "," + i);
                    for(int k = j+1; k < i; k++) {
                        total += unit;
                        total -= height[k];
                    }
                    j = i;
                }
                if (height[j] == 0) {
                    j = i;
                }
            }
        }
        j = height.length - 1;
        for(int i = height.length - 1; i >= maxHindex; i--) { // end to max
            if (height[i] != 0) {
                if (height[j] != 0 && height[i] >= height[j]) {
                    int unit = Math.min(height[j], height[i]);
                    System.out.println(unit + "," +i+ "," + j);
                    for(int k = j-1; k > i; k--) {
                        total += unit;
                        total -= height[k];
                    }
                    j = i;
                }
                if (height[j] == 0) {
                    j = i;
                }
                
            } 
        }
        return total;
    }
}
// @lc code=end

