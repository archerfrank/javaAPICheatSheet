### 排序

下面是两个元素都按照从大到小排。
```cpp

// 二维
vector<vector<int>> filtered;
sort(filtered.begin(), filtered.end(), [](vector<int> &v1, vector<int> &v2) -> bool {
            return v1[1] > v2[1] || (v1[1] == v2[1] && v1[0] > v2[0]);
        });

作者：力扣官方题解
链接：https://leetcode.cn/problems/filter-restaurants-by-vegan-friendly-price-and-distance/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


// 一维

        vector<int> id(plantTime.size());
        iota(id.begin(), id.end(), 0); // id[i] = i
        sort(id.begin(), id.end(), [&](int i, int j) { return growTime[i] > growTime[j]; });

作者：灵茶山艾府
链接：https://leetcode.cn/problems/earliest-possible-day-of-full-bloom/solutions/1200254/tan-xin-ji-qi-zheng-ming-by-endlesscheng-hfwe/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。




// https://blog.csdn.net/qq_37160943/article/details/72797118

#include <iostream>
#include <vector>
#include<algorithm>

using namespace std;

bool cmp(int x,int y)
{
	return x >y;
}

bool mysort(vector<int> a,vector<int> b){
	if(a[0]==b[0])
		return a[1]>b[1];
	return a[0]>b[0];
} 
//sort默认为非降序排序
int main()
{
	vector<int>a{2,5,1,4,6};
	//正向排序
	sort(a.begin(),a.end());
	for(auto i:a)
	{
		cout<<i<<" ";
	}
	cout<<endl;
	//反向排序
	sort(a.rbegin(),a.rend());
	for(auto i:a)
	{
		cout<<i<<" ";
	}
	cout<<endl;
	
	//带cmp参数的排序
	sort(a.begin(),a.end(),cmp);
	for(auto i:a)
	{
		cout<<i<<" ";
	}
	cout<<endl;
	
	
	vector<vector<int>> nums2{{1,90},{3,80},{2,92},{1,95},{3,91},{2,97}};
	sort(nums2.begin(),nums2.end(),mysort);
	for(auto x:nums2){//打印结果
		for(auto y:x)
			cout<<y<<" ";	
		cout<<endl;
	}
	

	
}

```

删除vector 第s个元素
```cpp
vector<int> g(m, 0);
g.erase(g.begin()+s);
```