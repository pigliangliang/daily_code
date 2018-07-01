#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：server.py

import socketserver            # 导入 socket 模块
import time
s = socket.socket()         # 创建 socket 对象
host = '127.0.0.1' # 获取本地主机名
port = 12345                # 设置端口
s.bind((host, port))        # 绑定端口

s.listen(5)                 # 等待客户端连接
while True:
    c, addr = s.accept()  # 建立客户端连接。
    print '连接地址：', addr
    time.sleep(2)
    #print c.recv(1024)
    s.sendall('hi{}'.format(addr))

#c.close()                # 关闭连接
