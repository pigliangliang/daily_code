#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：client.py

import socket               # 导入 socket 模块

s = socket.socket()         # 创建 socket 对象
host = '127.0.0.1'  # 获取本地主机名
port = 6675               # 设置端口好


s.connect((host, port))

input = raw_input()
s.send(input)
print s.recv(1024)
