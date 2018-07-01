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
'''
#一道Python面试题目，开始以为是0，2，4，6
#输出为6，6，6，6
#个人理解：列表推导式中生成的是匿名函数，包含一个参数
#所以循环遍历的匿名函数，
# <function <lambda> at 0x10210e5f0>
#<function <lambda> at 0x10210e668>
#<function <lambda> at 0x10210e6e0>
#<function <lambda> at 0x10210e758>
#因为包含一个参数，故需要传参
#for l in [lambda x:x*i for i in range(4)]:
#   print l(2)
'''
def f(n):
    while True:
        yield n
        n+=1
cnt = f(4)
for i in range(5):
    print cnt.next()

def simple_coroutine(a):
    while True:
        print('-> start')
        b = yield a
        print('-> recived', a, b)
        c = yield a + b
        print('-> recived', a, b, c)
sc = simple_coroutine(5)

print next(sc)#5
print sc.send(6) #11
print sc.send(7) # 5

'''
#阿里18校招编程题输入6个数，组成最大和最小时间
#未完成最大时间部分
'''
n = int(raw_input())
li = []
while n!=0:
    li.append(raw_input())
    n -=1
li.sort()
if li[0]+li[1]>'24' or li[2]+li[3]>'59' or li[4]+li[5]>'59':
    print 'N/A'#时间无效
else:
    li.insert(2,':')
    li.insert(5,':')
    print ''.join(li)#最小时间

    #最大时间
    s = []
    if '2' in li:
        for i in range(len(li)):
            if li[i]>=0 and li[i]<=3:
                s.append(li)
                s.sort()
                shi = '2'+s[0]
                li.remove('2')
                li.remove(s[0])
            else:
                break
    elif '1' in li:
        li.sort()
        shi = '1'+li[-1]
        li.remove('1')
        li.pop()
    elif '0' in li:
        li.sort()
        shi = '0'+ li[-1]
        li.remove('0')
        li.pop()
    else:
        print 'N/A'

'''
'''
import threading
import time


def action(arg):
    time.sleep(1)
    print 'sub thread start! the thread name is {}'.format(threading.current_thread().getName())
    print 'the arg is {}'.format(arg)
    time.sleep(1)

thread_list = []
for i in range(4):
    t = threading.Thread(target=action,args=(i,))
    t .setDaemon(True)
    thread_list.append(t)

for t in thread_list:
    t.start()
for t in thread_list:
    t.join()
print '{} is end'.format(threading.current_thread().getName())

'''
#py闭包和装饰器
'''
def count():
    a  = 1
    b = 2
    def sum():
        c = 1
        return a+c
    return sum
import time
def decorator(func):
    def wrapper(*args,**kwargs):
        startime = time.time()
        #func()
        endtime = time.time()
        print endtime - startime
    return wrapper
@decorator
def func():
    time.sleep(1)

func()




'''
'''
s1 = '+'
s2 = 'rrr'
print s1.join(s2)
#输出：r+r+r
print s1+s2


import time
import threading
from random import random
from Queue import Queue

q = Queue()
q.put('2')
print q.get_nowait()

'''

def dict2list(dic):#将字典转化为列表
     keys = dic.keys()
     vals = dic.values()
     lst = [(key, val) for key, val in zip(keys, vals)]
     return lst
















