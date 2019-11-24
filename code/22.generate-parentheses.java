/*
 * @lc app=leetcode id=22 lang=java
 *
 * [22] Generate Parentheses
 */

// @lc code=start
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        char []  chars = new char[]{'(',')'};
        int lcount = n - 1;
        int rcount = n;
        int count = 1;
        String s = "(";
        back(s, lcount, rcount, count,chars, res);
        return res;
    }
    private void back(String s, int lcount, int rcount, int count,char[] chars, List<String> res) {
        if (lcount == 0 && rcount == 0) {
            res.add(s);
            return;
        }
        for (char c : chars) {
            if(c == '(' && lcount > 0) {
                lcount--;
                s = s+c;
                back(s, lcount, rcount, ++count,chars, res);
                s = s.substring(0, s.length() - 1);
                count--;
                lcount++;
            }

            if(c == ')' && rcount > 0 && count > 0) {
                rcount--;
                s = s+c;
                back(s, lcount, rcount, --count,chars, res);
                s = s.substring(0, s.length() - 1);
                count--;
                rcount++;
            }
        }
    }
}
// @lc code=end

