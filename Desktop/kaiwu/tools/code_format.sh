#!/bin/bash

# 对代码进行格式化, 前提是需要安装下black
black --line-length 120 kaiwudrl/ tests tools
