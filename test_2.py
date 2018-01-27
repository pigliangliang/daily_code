# -*- coding:utf-8 -*-
'''
a = bin(5).lstrip('0b')
print a.zfill(9)
li = []
#for i in a.zfill(9):

li.extend(map(int,[x for x in a.zfill(len(a))]))
str(li)
print  li
print type(li)
b = int(a.lstrip('0b'),2)


sStr1 = '12345'
sStr2 = 'abcdef'
n = 3
sStr1 += sStr2[0:n]
print sStr1

#strspn(sStr1,sStr2)
sStr1 = '12345678'
sStr2 = '456'
#sStr1 and chars both in sStr1 and sStr2
print len(sStr1 and sStr2)
print sStr1 and sStr2  #公共字符串
print cmp(sStr2,sStr1)


a ='110'
b ='011'
for i in range(len(a)):
    print int(a[i])^int(b[i])
#列表推导式中使用if...else
#[x if x>1 else 1 for x in range(4)]

994 53963
65187 30346 85572 28612 43721 59733 99849 40659 29133 976 28470 22746 81760 25743 90717
from array import ArrayType
a =ArrayType('c')
print len(a)
print type(a)
a.append('c')
a.append('f')
print a
for i in a:
    print i
#字符串比较
print  cmp('0','1')
print cmp('1','1')
print cmp('1','0')
print cmp('0','0')
#字符串转换成列表
b ='agjdfd'
#print list(b)
a = ['1','0','1']
#列表转换成字符串
#print ''.join(a)
c = 'oreuoruer'
d = b+''.join("i")
#print d
print b.split(b[1])

n,m =map(int,raw_input().split())
ln=[]
lm=[]
ln.append(n)
lm.append(m)
lz = zip(ln,lm)
for z in lz:
    print z
if lz[0][1]==2:
    print 'df'

5
5 1 11 2 8
3
999 1000 1000
3
1 2 3
7
10 940 926 990 946 980 985
10
5 3 4 5 5 2 1 8 4 1
15
17 15 17 16 13 17 13 16 14 14 17 17 13 15 17
20
90 5 62 9 50 7 14 43 44 44 56 13 71 22 43 35 52 60 73 54
'''
import sys
while 1:
    r = raw_input().split()
    n,w,h = int(r[0]), int(r[1]), int(r[2])
    l = []
    for i in range(n):
        r = raw_input().split()
        wi, hi = int(r[0]), int(r[1])
        l.append((wi*hi, i, wi, hi))
    sl = []
    l = sorted(l)
    y = (w*h, 0, w, h)
    t = y
    c = 0
    for x in l:
        i = x[1]
        if x[0]>t[0] and x[2]>t[2] and x[3]>t[3]:
            if x[0]>y[0] and x[2]>y[2] and x[3]>y[3]:
                sl.append(x)
                c += 1
                if c>1:
                    t = y
                y = x
            elif x[1]<y[1]:
                sl[-1] = x

    print(len(sl))
    print sl
    print(" ".join([str(x[1] + 1) for x in sl])+" ")