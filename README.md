# Discovering Unprecedented Heuristics for Hub Identification by Joint Graph Embedding and Reinforcement Learning
MICCAI2020 [![](https://img.shields.io/badge/conference-MICCAI-yellowgreen)](https://www.miccai2020.org/en/)

工作整理

## 运行环境
* Linux
* C++

## 配置gnn静态链接库 [gnn](https://github.com/Hanjun-Dai/graphnn)
- 需要装intel的mkl，tbb数学函数库，默认装在根目录下的 /opt/intel/mkl
- 需改makefile文件，把对应库的路径改成当前自己电脑下这些库的路径
- 参考链接，进行编译，测试（由于我没有使用gpu，就把相应gpu的设置都跳过）

