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

### 删除vector 第s个元素

```cpp
vector<int> g(m, 0);
g.erase(g.begin()+s);
```



## Set

C++的set是树，所以可以在上面做二分查找，和count。



```c++
#include<bits/stdc++.h>
using namespace std;

int main()
{
	set<int> s; 
	  
	    // Function to insert elements 
	    // in the set container 
	    s.insert(1); 
	    s.insert(4); 
	    s.insert(4); 
	    s.insert(4); 
	    s.insert(2); 
	    s.insert(5); 
	    s.insert(6); 
	  
	    cout << "The set elements are: "; 
	    for (auto it = s.begin(); it != s.end(); it++) 
	        cout << *it << " "; 
	  
	    // when 2 is present 
	    // points to next element after 2 
	    auto it = s.upper_bound(2); //第一个大于2的值 
	    cout << "\nThe upper bound of key 2 is "; 
	    cout << (*it) << endl; 
	    
	    it = s.lower_bound(2); //第一个大于等于2的值 
	    cout << "\nThe lower bound of key 2 is "; 
	    cout << (*it) << endl; 
	  
	    // when 3 is not present 
	    // points to next greater after 3 
	    it = s.upper_bound(3); 
	    cout << "The upper bound of key 3 is "; 
	    cout << (*it) << endl; 
	    
	    int key = 8; 
	    it = s.upper_bound(key); 
	    if (it == s.end()) 
	        cout << "The given key is greater "
	                "than or equal to the largest element \n"; 
	    else
	        cout << "The immediate greater element "
	             << "is " << *it << endl; 
	  
	    key = 3; 
	    it = s.upper_bound(key); 
	    if (it == s.end()) 
	        cout << "The given key is greater "
	                "than or equal to the largest element \n"; 
	    else
	        cout << "The immediate greater element "
	             << "is " << *it << endl; 
	  
	  	cout<<s.count(4)<<endl;
	    return 0; 
}
```







## 字符串 数字 转换

```c++
#include <iostream>
#include <string>
 
int main() {
    int num = 1234;
    std::string str = std::to_string(num);
    std::cout << str << std::endl;
    return 0;
}



#include<bits/stdc++.h>
using namespace std;
int main(){
    string str="123";
    cout<<str<<"    str的类型为"<<typeid(str).name()<<endl;//输出Ss表示string类型
    //atoi的参数类型为（const char *nptr）而string是一个类，所以需要获取str的首地址即str.c_str()
    int num=atoi(str.c_str()); 
    cout<<num<<"    num的类型为"<<typeid(num).name()<<endl;//输出i表示int类型 
}


string b = "adcdad";
string c = b;
sort(c.begin(), c.end());
cout<<b<<" "<<c<<endl; // c 被排序，但是b还是原来的。
```

### Memset

将动态的数组，初始化为0.
```c++
	int b[s + 1];
	memset(b, 0, sizeof(b));
```

**0x3f：** 0x指的是十六进制的意思，说明 3f 是一个十六进制数，十六进制0-9是不变的，用a、b、c、d、e、f来代表10、11、12、13、14、15，那么3f对应的二进制就是0011 1111的意思，我们一般需要用最大值的时候是不会取真正的最大值的，因为最大值进行相加运算的时候就会越界了，这是有风险的，我们应该取最大值的一半或者更小一点，即经常使用的 0x3f3f3f3f ，而0x3f就可以把每 8 位都初始化为这个值，如果是 int 型的，那么就会被初始化为 0x3f3f3f3f ,如果是long long型，那么就会被初始化为 0x3f3f3f3f3f3f3f3f

```c++
int型数组
memset(arr1,0,sizeof arr1)后，数组的值为
0 0 0 0 0 0 0 0 0 0
memset(arr1,0x3f,sizeof arr1)后，数组的值为
1061109567 1061109567 1061109567 1061109567 1061109567 1061109567 1061109567 1061109567 1061109567 1061109567

long long型数组
memset(arr2,0,sizeof arr2)后，数组的值为
0 0 0 0 0 0 0 0 0 0
memset(arr2,0x3f,sizeof arr2)后，数组的值为
4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399 4557430888798830399
————————————————
版权声明：本文为CSDN博主「旧林墨烟」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_52115456/article/details/127645202
```
