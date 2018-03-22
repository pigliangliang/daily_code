# -*- coding:utf-8 -*-
#l1 =['b','c','d','c','a','a']
#l2 = {}.fromkeys(l1).keys()
#print(l2)

#for i in l2:
#   print(l1.count(i))
#计数列表中的key出现的次数
#from collections import  Counter
#dict = Counter(l1)
#print(dict)
#print(len(Counter(l1)))
#print(sorted(dict,key=lambda x :dict[x] ))
#print(sorted(dict.values()))
#print(sorted(dict.items(),key=lambda item:item[1]))

'''
while True:
    goods_num =  0
    price_list = []
    goods_list = []
    input_a = map(lambda x:int(x),raw_input().split())
    input_b = map(lambda x:int(x),raw_input().split())
    for i in input_a:
        goods_num = i
    #print(goods_num)
    for i in input_b:
        price_list.append(int(i))
    while goods_num != 0:
        item = raw_input()
        if item != '':
            goods_list.append(item)
            goods_num -=1
        else:
            continue
    from collections import Counter
    dic = Counter(goods_list)
    #min
    price_list = sorted(price_list)
    goods_list = sorted(dic.values(),reverse=True)
    #print price_list,goods_list
    min = map(lambda x: x[0]*x[1] , zip(price_list,goods_list))
    s_min = 0
    for m in min:
        s_min +=int(m)
    #max
    price_list = sorted(price_list,reverse=True)
    goods_list = sorted(dic.values(),reverse=True)
    max = map(lambda x: x[0]*x[1],zip(price_list,goods_list))
    s_max = 0
    for m in max:
        s_max +=int(m)
    if s_max and s_min !=0:
        print s_min,s_max
    else:
        continue
        

#########################################
def BinSearch(array, key, low, high):
    mid = int((low+high)/2)
    if key == array[mid]:  # 若找到
        return array[mid]
    if low > high:
        return False

    if key < array[mid]:
        return BinSearch(array, key, low, mid-1)  #递归
    if key > array[mid]:
        return BinSearch(array, key, mid+1, high)

####折半查找 时间复杂度o（logn）

def BinSearch(array,key,low,high):
    mid = int((low+high)/2)
    if array[mid] == key:
        return  key
    if low>high:
        return False
    if array[mid] >key:
        return BinSearch(array,key,low,mid-1)
    if array[mid] < key:
        return BinSearch(array,key,mid+1,high)

if __name__ == "__main__":
    array = [4, 13, 27, 38, 49, 49, 55, 65, 76, 97]
    ret = BinSearch(array, 4, 0, len(array)-1)  # 通过折半查找，找到65
    print(ret)



#归并排序
#稳定排序，时间复杂度,空间复杂度均为o(nlogn)，适用于n较大的情况
#算法过程
def MergeSort(lists):
    if len(lists) == 1:
        return lists
    num = int( len(lists) / 2)
    left = MergeSort(lists[:num])
    right = MergeSort(lists[num:])
    #print "left:{}".format(left)
    #print "right{}".format(right)
    return Merge(left, right)
#从大到小排列
def Merge(left,right):
    result=[]
    l ,r =0,0
    while l<len(left) and r<len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l +=1
        else:
            result.append(right[r])
            r+=1
    result.extend(left[l:])
    result.extend(right[r:])
 
    return result  

#从大到小排列
def Merge(left,right):
    list_result = []
    while left and right:
        if left[0] >= right[0]:
            list_result.append(left.pop(0))
        else:
            list_result.append(right.pop(0))
    if left:
        list_result.extend(left)
    else:
        list_result.extend(right)
    return list_result
print MergeSort([1, 20, 3, 4, 5, 6, 7, 90, 21, 23, 45])
'''
'''
输入
输入数据是三个整数：n, m, k(1≤n, m≤5 * 105, 1≤k≤nm)。
样例输入
2
3
4
输出
输出n * m乘法表按照不减顺序排列的第k个数。
'''
#方式一，运行时间超过算法题目要求的时间，放弃
'''
input_a = raw_input().split()
loc = int(input_a.pop())
col = int(input_a.pop())
row = int(input_a.pop())
input_array = []

r = 0
while r  < row :
    for c in range(col):
        input_array.append((r+1)*(c+1))
    r+=1
input = sorted(input_array)
#print input_array
print input[loc-1]


#方式二
#note：二分搜索中的的mid在main函数中是index
#在search_key中mid是一个真实的数值
def search_key(n,m,mid):
    locate = 0
    for r in range(1, n + 1):
        if mid / r >= m:
            locate += m
        else:
            locate += mid / r
    return locate
if __name__ =="__main__":
    input_a = raw_input().split()
    k = int(input_a.pop())
    m = int(input_a.pop())
    n = int(input_a.pop())
    low = 1
    high = n*m
    while True:
        mid = int(low + high) / 2
        loc = search_key(n,m,mid)
        if k==loc:
            break
        elif k >loc:
            low = mid +1

        else:
            high = mid -1
    print mid
    

#方式二由于运行时间不通过，修改如下 
#但是时间依然未能通过系统要求的时间，折腾很久，放弃
n,m,k = map(int,raw_input().split())
low = 1
high = n*m
while True:
    mid = int(low + high) // 2
    locate = 0
    for r in range(1,n+1):
        if mid//r >=m:
            locate+=m
        else:
            locate+=mid//r
    if k==locate:
        print mid
        break
    elif k >locate:
        low = mid + 1
    else:
        high = mid - 1
#print mid
'''
#------------------------------
'''输入只有一行，一个字符串，长度不超过100000，只由小写字母组成
样例输入
aaabbaa
输出
输出一行，符合要求的子串种数

string = raw_input()
ls_sub = []
locate = 0
for i in range(len(string)-1):
    if string[i]==string[i+1]:
        continue
    else:
        ls_sub.append(string[locate:i+1])
        locate = i+1
sum = 0
for i in ls_sub:
    sum +=len(i)
ls_sub.append(string[sum:])
ls =[]
c = 0
for i in ls_sub:
    for c in range(len(i)):
        ls.append(i[:c+1])
print len(set(ls))
'''
#------------------------------
'''
一个数A如果按2到A-1进制表达时，各个位数之和的均值是多少？
使用fraction类进行自动约分

from fractions import Fraction
while True:
    a= int(raw_input())
    s= 0
    def jinzhi(a,i):
        y=1
        ls=[]
        while y:
            x = a % i
            ls.append(x)
            y = a / i
            a = y
        return ls
    for i in range(2,a):
        s+=sum(jinzhi(a,i))
    result = Fraction(s,(a-2))
    if result==int(result):
        print str(result)+"/1"
    else:
        print result
'''
#------------------------------
'''
表格单元所在的行则是按数值从1开始编号的，表格单元名称则是其列编号和行编号的组合，如单元格BB22代表的单元格为54列中第22行的单元格。
小B感兴趣的是，编号系统有时也可以采用RxCy的规则，其中x和y为数值，表示单元格位于第x行的有第y列。上述例子中的单元格采用这种编码体系时的名称为R22C54。
小B希望快速实现两种表示之间的转换，请你帮忙设计程序将一种方式表示的坐标转换为另一种方式。

#
import re
k=1
dict ={}
for s in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    dict[s]=k
    k+=1
input_a = int(raw_input())
for i in range(input_a):
    input_b=raw_input().strip()
    result = re.match(r'R(\d+)C(\d+)',input_b)
    if result:
        c = int(result.groups()[1])
        res = ''
        while c>0:
            if c%26==0:
                res = 'Z' + res
                c =c/26 -1
            else:
                res = chr(c%26+64)+res
                c /= 26

        print res+result.groups()[0]
    else:
        result = re.split(r'\d+', input_b)
        row=[]
        for r in result[0]:
            row.append(r)
        sum = 0
        sum +=int(dict[row.pop()])

        digital=26
        for i in row[::-1]:
            sum += int(dict[i])*digital
            digital*=26
        #print sum
        print "R{}C{}".format(input_b[len(result[0]):],sum)

#print sum(map(lambda x:x[0]*x[1],zip([i for i in range(1,len(row)+1)],[dict[c] for c in row])))
'''
#------------------------------
'''头条的2017校招开始了！为了这次校招，我们组织了一个规模宏大的出题团队。每个出题人都出了一些有趣的题目，而我们现在想把这些题目组合成若干场考试出来。在选题之前，我们对题目进行了盲审，并定出了每道题的难度系数。一场考试包含3道开放性题目，假设他们的难度从小到大分别为a, b, c，我们希望这3道题能满足下列条件：
a＜= b＜= c
b - a＜= 10
c - b＜= 10
所有出题人一共出了n道开放性题目。现在我们想把这n道题分布到若干场考试中（1场或多场，每道题都必须使用且只能用一次），然而由于上述条件的限制，可能有一些考试没法凑够3道题，因此出题人就需要多出一些适当难度的题目来让每场考试都达到要求。然而我们出题已经出得很累了，你能计算出我们最少还需要再出几道题吗？

#个人程序
num = raw_input()
score = map(int,raw_input().split())
score.sort()
count = 0
ls =[]
if len(score)==1:
    count=2
else:
    count=0
    while len(score) !=0:
        ls.append(score[0])
        score.pop(0)
        for s in score:
            if s - ls[len(ls)-1] <=10 and len(ls)<=3:
                ls.append(s)
            elif s - ls[0] >10:
                continue
            else:
                break
        if len(ls)==2:
            score.remove(ls[1])
            count +=1
        elif len(ls)==1:
            count +=2
        else:
            score.remove(ls[1])
            score.remove(ls[2])
            count +=0
        ls = []
print count

#方式二，参考其他同学，没太搞懂思想
inp = raw_input()
score = map(int,raw_input().split())
score.sort()
num,result ,locate= 0,0,score[0]
for s in score:
    if num==3 or  s-locate>10:
        result+=(3-num)
        num=0
    num +=1
    locate = s
result+=(3-num)
print "{}".format(result)
'''
#---------------------------
#给定整数m以及n个数字A1, A2, …, An，将数列A中所有元素两两异或，共能得到n(n-1)/2个结果。请求出这些结果中大于m的有多少个。
import math
#n,m =map(int,raw_input().split())
#inp = map(int,raw_input().split())
#python 自带了进制的转换函数
'''
def binary(a):
    la =[]
    if a/2 == 0:
        la.append(a)
        return la
    else:
        i = a %2
        la.append(i)
        x = a/2
        binary(x)
'''
'''
时间超限，运行结果没问题，但是找不到合理的解决方案
n,m =map(int,raw_input().split())
inp = map(int,raw_input().split())
count = 0
while len(inp)>1:
    a  = inp.pop(0)
    sa = bin(a).lstrip('0b')
    for i in inp:
        si = bin(i).lstrip('0b')
        if len(si)>len(sa):
            sa = sa.zfill(len(si))
        if len(si)<len(sa):
            si = si.zfill(len(sa))
        res = [str(1) if mp !=0 else str(0) for mp in map(lambda x:cmp(x[0],x[1]),zip(sa,si))]
        if int(''.join(res),2)>m:
            count+=1
print count
'''
'''
我们规定对一个字符串的shift操作如下：
shift(“ABCD”, 0) = “ABCD”
shift(“ABCD”, 1) = “BCDA”
shift(“ABCD”, 2) = “CDAB”
换言之, 我们把最左侧的N个字符剪切下来, 按序附加到了右侧。
给定一个长度为n的字符串，我们规定最多可以进行n次向左的循环shift操作。如果shift(string, x) = string (0＜= x ＜n), 我们称其为一次匹配(match)。求在shift过程中出现匹配的次数。
'''
#切片的方式实现，没有通过系统的时间复杂度要求
'''
inp = raw_input()
count = 1
for i in range(1,len(inp)):
    if inp[0] == inp[i]:
        if cmp(inp[i:]+inp[0:i],inp)==0:
         count+=1
    else:
        continue
print count
'''
#给定整数n和m，将1到n的这n个整数按字典序排列之后，求其中的第m个数字。
#对于n = 11，m = 4，按字典序排列依次为1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9，因此第4个数字为2。
#根据图的优先遍历的思想
#方法一：
'''
我们分析题目中给的例子可以知道，数字1的子节点有4个(10,11,12,13)，而后面的数字2到9都没有子节点，
那么这道题实际上就变成了一个先序遍历十叉树的问题，那么难点就变成了如何计算出每个节点的子节点的个数，
我们不停的用k减去子节点的个数，当k减到0的时候，当前位置的数字即为所求。现在我们来看如何求子节点个数，
比如数字1和数字2，我们要求按字典遍历顺序从1到2需要经过多少个数字，首先把1本身这一个数字加到step中，
然后我们把范围扩大十倍，范围变成10到20之前，但是由于我们要考虑n的大小，由于n为13，所以只有4个子节点，
这样我们就知道从数字1遍历到数字2需要经过5个数字，然后我们看step是否小于等于k，如果是，我们cur自增1，
k减去step；如果不是，说明要求的数字在子节点中，我们此时cur乘以10，k自减1，以此类推，直到k为0推出循环，
此时cur即为所求：

n,m = map(int,raw_input().split())
cur = 1
m -= 1
while m>0:
    step = 0
    first = cur
    last = cur + 1
    while first<=n:
        step += min(n+1,last) - first
        first *= 10
        last *= 10
    if step <= m:
        cur += 1
        m -= step
    else:
        cur *= 10
        m -= 1
print cur
'''
#方法二
'''
#根据规律1，100，1000....2,20...3,30,300等 
n,m = map(int,raw_input().split())
c =1
cur = 1
while c != m:
    if cur*10 <=n:
        cur = cur*10
        c +=1
    else:
        if cur>=n:
            cur/=10
        cur +=1
        c += 1
        if cur%10==0:
            cur /=10
print cur
'''
#二叉树遍历先序遍历
'''
class Tree:
    def __init__(self,data,left,right):
        self.data=data
        self.left=left
        self.right=right
    def pre_visit(self,Tree):
        if Tree:
            print Tree.data
            self.pre_visit(Tree.left)
            self.pre_visit(Tree.right)
    
tree1 = Tree(1,0,0)
tree2 = Tree(2,0,0)
tree3 = Tree(3,tree1,tree2)
tree3.pre_visit(tree3)
'''
#买糖果问题
#放弃未完成
'''
while True:
    lz = {}
    a1 = []
    a2 = []
    res = []
    n,v = map(int,raw_input().split())
    for i in range(n):
        ti, pi = map(int, raw_input().split())
        lz[i+1] = (ti,pi)
    for z in sorted(lz.items(),key=lambda x:x[1][1],reverse=True):
        if z[1][0] ==1:
            a1.append(z)
        else:
            a2.append(z)
    if v > len(a1)+2*len(a2):
        print 0
        print 'No'
    else:
        i = 0
        while len(a1)-i!=0 and len(a2)-i!=0 and v>1:
            c = a1[i]
            d = a2[i]
            if c[1][1]*2 >=d[1][1]:
                v -=2
                res.append(c)
            else:
                v-=2
                res.append(d)
            i +=1

        if v == 0:
            for r in res:
                print r[1][1]
                print r[0]

        elif len(a1) ==0:
            count = v/2
            res.append()
'''
#交易清单（京东2016实习生真题）
#输入有若干组，每组的第一行为两个正整数n和s（1<=n<=1000，1<=s<=50），
# 分别表示委托数和最抢手的清单数，接下来的n行为具体的委托信息，
# 每行包含3部分，第一部分为一个字母‘B’或‘S’，表示买入或卖出，
# 后两部分为两个整数p和q，表示报价和数量。任何卖出委托的报价都比买入委托的报价高。
'''
while True:
    import sys
    def operation(l,F):
        ls = sorted(l.items(), key=lambda item: item[0], reverse=True)
        if m >= len(ls):
            for s in ls:
                print "{} {} {}".format(F,s[0], s[1])
        else:
            for s in ls[0:m]:
                print "{} {} {}".format(F,s[0], s[1])
    n,m = map(int,sys.stdin.readline().strip().split())
    lb = {}
    ls = {}
    while n!=0:
        c,p,q = raw_input().split()
        p = int(p)
        q = int(q)
        if c =="B":
            if lb.has_key(p):
                lb[p]+=q
            else:
                lb[p] =q

        else:
            if ls.has_key(p):
                ls[p] +=q
            else:
                ls[p] = q
        n -=1
    operation(ls,"S")
    operation(lb,"B")
'''
#选举游戏（京东2016实习生真题）
'''
while True:
    import sys
    n = int(sys.stdin.readline())
    vote = map(int,sys.stdin.readline().strip().split())
    D = vote[0]
    if D>max(vote[1:]):
        print '0'
    elif D==max(vote[1:]):
        print '1'
    else:
        vote = filter(lambda x:x>=D,vote)
        vote.sort(reverse=True)
        count = 0
        while D<=vote[0]:
            D +=1
            vote[0]-=1
            vote.sort(reverse=True)
            count += 1
        print count
'''
#生日礼物（京东2016实习生真题）
'''
while 1:
    def operation(enve,count,p,q,seq):
        s = filter(lambda x: x[1][0] > p and x[1][1] > q, sorted(enve.items(), key=lambda x: x[1]))
        if len(s) == 0:
            print count
            return seq
        else:
            if len(s)==1:
                seq.append(s[0][0])
                print count+1
                return seq
            else:
                count += 1
                p =s[0][1][0]
                q =s[0][1][1]

                seq.append(s[0][0])

                operation(dict(s),count,p,q,seq)
    n,p,q = map(int,raw_input().split())
    enve = {}
    i = 1
    while n!=0:
        size = map(int, raw_input().split())
        if  size not in enve.values() :
            enve[i] = size
            i +=1
        n -= 1
    ls = []
    operation(enve,0,p,q,ls)
    print ' '.join(map(str,ls))
'''
#备考 京东2016实习题
'''
while 1:
    import sys
    import numpy as np
    d,t = map(int,sys.stdin.readline().strip().split())
    lt = [ ]
    while d!=0:
        lt.append(map(int,raw_input().split()))
        d -= 1
    d = np.array(lt)
    minsum = sum(d[:,:1])[0]
    maxsum = sum(d[:,1:])[0]
    if t >maxsum or t<minsum:
        print "No"
    else:
        print "Yes"
        timesum = t - minsum
        for l in lt:
            if l[1] - l[0] >=timesum:
                print l[0] + timesum
                timesum = 0
            else:
                print l[1]
                timesum -= l[1]
'''
#铺地砖（京东2016实习生真题）
'''
def calculator(n,m,a,count):
    if a >=n and a>=m:
        count +=1
    elif a>=n and a<=m:
        if m%a == 0:
            count +=m/a
        else:
            count += m/a+1
    elif a<=n and a>=m:
        if n%a==0:
            count += n/a
        else:
            count += n/a+1
    else:
        count+=1
        calculator(n-a,m,a,count)
        calculator(a,m-a,a,count)
    print count
'''
#方法一思想是行列分别除以地砖的大小然后将结果乘积
'''
import numpy as np
T = int(raw_input())
while T!=0:
    arr = np.array(map(int,raw_input().strip().split()))
    res = np.where(arr[:2]%arr[2],arr[:2]/arr[2]+1,arr[:2]/arr[2] )
    print np.multiply.reduce(res)
    T-=1
'''
#方法二思想同上
'''
T = int(raw_input())
while T!=0:
    T-=1
    n,m,a = map(int,raw_input().split())
    print [n/a+1 if n%a else n/a][0]*[m/a+1 if m%a else m/a][0]
'''
#查找列表中大于某个值的元素
'''
import numpy as np
def great_elem(list,m):
    l = np.array(list)
    print l[l>m]
ls = [1,2,4,5,6,6,3,53,3,42,42,424,24,24,4326565,6546565,6536578]
great_elem(ls,5)
'''
#python 迭代器
#菲薄那也数列为例
'''
class Fib:
    def __init__(self):
        self.prev = 0
        self.curr = 1
    def __iter__(self):
        return self
    #内部迭代
    def next(self):
        value = self.curr
        self.curr +=self.prev
        self.prev = value
        return value
from itertools import islice
f = Fib()
print list(islice(f,0,2))
'''
#生成器
#菲薄那也数列
'''
from itertools import islice
def fib():
    pre,curr = 0,1
    while True:
        yield curr
        pre,curr = curr,curr+pre
print list(islice(fib(),0,2))

#练习
def iter_someting():
    li = [x for x in range(100)]
    for i in li:
        yield i
it = iter_someting()
from itertools import islice
#问题是无法多次迭代
#看结果：
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
print list(islice(it,0,10))
print list(islice(it,0,10))
'''
#函数闭包问题
'''
flist = []

for l in [lambda x:x*i for i in range(3)]:
    print l
for i in range(3):
    def foo(x): print x + i
    flist.append(foo)
for f in flist:
    f(2)
'''
#路径规划（京东2016实习生真题）
'''
start = raw_input()
after = raw_input()
def equal_row_or_col(start,after):
    res = [True if x[0]==x[1] else False for x in  zip(start,after)]
    if res[0]:
        dis = int(after[1])-int(start[1])
        d = abs(dis)
        print abs(dis)
        while d:
            if dis>0:
                print 'U'
            else:
                print 'D'
            d -=1
    elif res[1]:
        dis = ord(after[0])-ord(start[0])
        d = abs(dis)
        print abs(dis)
        while d:
            if dis>0:
                print 'R'
            else:
                print 'L'
            d -=1
def not_equal(start,after):
    dis = map(abs, (ord(start[0]) - ord(after[0]), int(start[1]) - int(after[1])))
    dis = min(dis)
    s0 = ord(start[0]) + dis
    s1 = int(start[1]) + dis
    result = (s0,s1 ,dis)
    return result
if start[0]<after[0] and start[1]<after[1]:
    val = not_equal(start,after)
    start = (val[0],val[1])
    dis = val[2]
    while dis:
        print "RU"
        dis -=1
    print start
    equal_row_or_col(start,after)
elif start[0]<after[0] and start[1]>after[1]:
    val = not_equal(start,after)
    start = (val[0],val[1])
    dis = val[2]
    while dis!=0:
        print "RD"
        dis -=1
    equal_row_or_col(start,after)
elif start[0]>after[0] and start[1]<after[1]:
    val = not_equal(start, after)
    start = (val[0], val[1])
    dis = val[2]
    while dis:
        print "LD"
        dis -=1
    equal_row_or_col(start,after)
else:
    val = not_equal(start, after)
    start = (val[0], val[1])
    dis = val[2]
    while dis:
        print "LU"
        dis -=1
    equal_row_or_col(start, after)
'''
#协程
'''
def consumer():
    r = ''
    while True:
        n = yield r#语句1
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)#预激活协程，使得协程暂停在语句1处
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...'  n)
        r = c.send(n)#激活协程，协程在语句1处开始向下继续执行
                    #n的值作为语句1的值，本语句的值是yield 中参数r
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()
c = consumer()
produce(c)
'''
#快速排序算法
#quick sort
'''
def quickSort(L, low, high):
    i = low
    j = high

    key = L[i]
    while i<j:
        while i<j and L[j]>=key:
            j -=1
        L[i]=L[j]
        while i<j and L[i]<=key:
            i +=1
        L[j]=L[i]
    L[i]=key

    quickSort(L, low, i-1)
    quickSort(L, j+1, high)
    return L
input = map(int,raw_input().split())
print quickSort(input,0,len(input)-1)
'''

#提利昂的赏赐（百度2017秋招真题)
'''
n = int(raw_input())
info = []
while n!=0:
    info.append(raw_input().split())
    n -=1
d = {}

for i in info:
    pcount=0
    while i[1] >80 and i[5]>0:
        pcount +=8000
        break
    while i[1]>85 and i[2]>80:
        pcount +=4000
        break
    while i[1]>90:
        pcount +=2000
        break
    while i[1]>85 and i[4]=='Y':
        pcount +=1000
        break
    while i[2]>80 and i[3]=='Y':
        pcount +=850
        break
    d[i[0]]=pcount
d = sorted(d.items(),key=lambda d:d[1],reverse=True)

for k in d:
    print k

'''

#士兵队列（百度秋招真题）
'''
group = int(raw_input())
info = []
while group!=0:
    num = raw_input()
    info.append(map(int,raw_input().split()))
    group -=1

for i in info:
    count = 1
    max_key = i[0]
    for r in range(1,len(i)):
        if i[r]>max_key:
            count +=1
            max_key = i[r]
    print count
'''
#异或（京东2017实习生真题）
'''
num = int(raw_input())
f_num = raw_input()
s_num = raw_input()
res = [0 if x[0]==x[1] else 1 for x in zip(f_num,s_num)  ]
res.reverse()
count=0
for i,v in enumerate(res):
    if v==1:
        count+=2**i
print count
'''
#三子棋（京东2016实习生真题）
'''
while True:
    r1=raw_input()
    r2=raw_input()
    r3=raw_input()
    r = r1+r2+r3
    if r1=='XXX' or r2=='XXX' or r3=='XXX':
        print '1 won'
    elif r1=='OOO' or r2=="OOO" or r3=='OOO':
        print '2 won'
    elif r1[0]+r2[1]+r3[2]=='XXX':
        print '1 won'
    elif r1[2]+r2[1]+r3[0]=="OOO":
        print '2 won'

    if '.' in r:
        if r.count('X')>r.count('O'):
            print '2'
        else:
            print '1'
    else:
        print 'draw'
'''
#python 反射机制
'''imp = raw_input("请输入模块：")
dd = __import__(imp)
inp_func = raw_input("请输入要执行的函数")
f = getattr(dd,inp_func,None)
f()

class A:
    def __init__(self):
        #self.name = 'pig'
        self.age = '18'
    def method(self):
        print "method func"
Instance = A()
print getattr(Instance,'name','not find')
r = hasattr(Instance,'sex')#判断实例是否有sex属性，没有返回false
print r
setattr(Instance,'sex','male')

print hasattr(Instance,'sex')
'''
#快排
'''
def quicksort(l,low,high):
    if low < high:
        i = low
        j = high
        key = l[i]
        while i<j:
            while i<j and l[j]>=key:
                j -=1
            l[i]=l[j]
            while i<j and l[i]<key:
                i +=1
            l[j]=l[i]

        l[i]=key
        quicksort(l,low,i-1)
        quicksort(l,i+1,high)
    return l
l = [3,2,41,6,7,1,9,0]
quicksort(l,0,len(l)-1)
print l
'''
#折半查找
'''
def BinSearch(l,key,low,high):
    mid = int((low+high)/2)
    if key==l[mid]:
        return l[mid]
    if low>high:
        return False
    if key<mid:
        return BinSearch(l,key,low,mid-1)
    if key>mid:
        return BinSearch(l,key,mid+1,high)

l = [4, 13, 27, 38, 49, 49, 55, 65, 76, 97]
ret = BinSearch(l, 76, 0, len(l)-1)
print ret
'''
#归并排序O(nlogn)
'''
def merge_sort(l):
    mid = len(l)/2
    if len(l)==1:
        return l
    l_left = merge_sort(l[:mid])
    l_right= merge_sort(l[mid:])
    return merge(l_left,l_right)

def merge(l_left,l_right):
    result = []
    while l_left and l_right:
        if l_left[0]<=l_right[0]:
            result.append(l_left.pop(0))
        else:
            result.append(l_right.pop(0))
    return result+l_left+l_right

array = [49,0,38,65,97,76,3,1]
print merge_sort(array)
'''
#找数组中出现次数为1次的元素
#字典统计法
'''
from collections import Counter
l = [2,3,2,3,5,1,1]
dict = Counter(l)
for k,i in Counter(l).items():
    if i==1:
        print k
        '''
#print sorted(dict.items() ,key=lambda item:item[1])
#列表计数
'''for i in l:
    if l.count(i)==1:
        print i'''
#异或，将所有数字做一遍异或即可
#相同数字异或结果为零，所以元素出现一次
#结果肯定不为零
'''key = 0
for i in l:
    key ^=i
print key
'''
'''
n,m  = map(int,raw_input().split())
price = map(int,raw_input().split())
price.sort(reverse=True)
profit = {}
if n>m:

    for i in range(1,m+1):
        profit[price[i-1]]=i*price[i-1]
    print sorted(profit.items(),key=lambda p:p[1],reverse=True)[0][0]
else:
    for i in range(1,n+1):
        profit[price[i-1]]=i*price[i-1]
    print sorted(profit.items(),key=lambda p:p[1],reverse=True)[0][0]
'''
'''
n = int(raw_input())
ch = raw_input()
countx= 0
for i in range(n):
    subch = []
    if ch[i]>='1' and ch[i]<='9':
        if i-int(ch[i])<0 and i+int(ch[i])>n:
            subch.extend(ch[0:n])
        elif i-int(ch[i])<0:
            subch.extend(ch[0:i+int(ch[i])])
        elif i+int(ch[i])>n:
            subch.extend(ch[i-int(ch[i]):n])

        else:
            subch.extend(ch[i-int(ch[i]):i+int(ch[i])])
    if 'X' in subch:
        countx +=subch.count('X')

    else:
        continue
print countx

n = int(raw_input())
m = raw_input()
c = 0
for i in m:
    if i >='1' and i<='9':
        c += m[max(m.index(i)-int(i),0):min(len(m)-1,m.index(i)+int(i)+1)].count('X')
    else:
        continue
print c

'''

































