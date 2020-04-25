# Discovering Unprecedented Heuristics for Hub Identification by Joint Graph Embedding and Reinforcement Learning  
[![](https://img.shields.io/badge/conference-MICCAI-yellowgreen)](https://www.miccai2020.org/en/) ![](https://img.shields.io/badge/version-1.0.0-blue) ![](https://img.shields.io/badge/status-submitted-orange)

工作整理  

## 工作概述
A plethora of neuroscience studies find that connector hub nodes play a key role in regulating multiple modules and supporting brain functions such as consciousness and cognition, due to its critical topological location in the net-work. Current approaches mainly rely on the hand-crafted attributes (aka. graph embedding at each node) from the domain knowledge of network neuroscience such as high connectivity degree to identify connector hub nodes. However, sim-ple ranking heuristic based on the pre-defined attributes has limited power to characterize the complex network topology, which often results in less accurate hub identification results. Although graph theory allows us to find connector hubs with a greater mathematical insight, the large scale of brain network often compromises the well-defined optimization into a local and sub-optimal solution. To overcome these limitations, we propose a joint graph embedding and hub identification solution in a reinforcement learning framework to discover the un-precedented heuristics from the existing knowledge of network neuroscience and graph theory, which allows us to outperform the current state-of-the-art hub iden-tification methods. We have achieved more reliable and replicable hub identifica-tion results on both simulated and real brain network data, suggesting the high applicability to various network analysis studies in neuroscience and neuroimag-ing fields.



## 运行环境
* Linux
* C++

## 配置GNN静态链接库 [GNN](https://github.com/Hanjun-Dai/graphnn)
- 需要装intel的mkl，tbb数学函数库，默认装在根目录下的 /opt/intel/mkl
- 需改makefile文件，把对应库的路径改成当前自己电脑下这些库的路径
- 参考链接，进行编译，测试（由于我没有使用gpu，就把相应gpu的设置都跳过）

## 配置reinforcement + graph embedding框架 [RL+GE](https://github.com/Hanjun-Dai/graph_comb_opt)
- 需要配置一个[eigen3](https://eigen.tuxfamily.org/dox/GettingStarted.html)的矩阵计算函数库用来计算特征值，并且在对应makefile文件中添加路径
- 核心修改的代码文件是maxcut_env模块和maxcut_lib.cpp(把主函数,读写文件,参数的设置都放到这了)
- 修改自己的makefile文件

## 代码主要功能
以s2v_maxcut为例
### 核心部分有几个模块
- Config模块主要是设置一些参数
- Graph模块主要是实现存储图的信息(节点，边，邻居)和相应的采图的一些基本操作
- Maxcut_env模块主要是实现强化学习几个要素(初始化状态，存放行动，状态，终止条件，计算奖励)
- Qnet模块主要是配置神经网络的结构
- Simulator模块主要是实现整个训练过程的整合

## 常用工具和配置环境遇到的一些问题总结
### GDB调试相关操作
set print object on 可以显示派生类的成员

set print pretty on 树形打印

显示STL内容
- (https://www.cnblogs.com/silentNight/p/5466418.html)
- (https://sourceware.org/gdb/wiki/STLSupport?action=AttachFile&do=view&target=stl-views-1.0.3.gdb)

### 配置mkl,tbb，遇到找不到相关库的问题和解决方案
一般添加相应的环境变量到home目录下的.bashrc就能解决(添加如下三句命令)

export LD_LIBRARY_PATH=/home/chenanqi/intel/mkl/lib/intel64/:$LD_LIBRARY_PATH
exportLD_LIBRARY_PATH=/home/chenanqi/intel/tbb/lib/intel64/gcc4.7/:$LD_LIBRARY_PATH
source /home/chenanqi/intel/parallel_studio_xe_2019.5.075/bin/psxevars.sh

如果还不能解决，可以参考(https://github.com/davisking/dlib/issues/587)

### 后台跑多个程序使用命令screen
http://man.linuxde.net/screen  
先screen -S xx(窗口的名字)  
然后运行程序 你要跑的  
然后ctrl+a d 把当前会话切后台运行  
再关闭就没问题了

#### 最后一些问题的解决方案可以在原作者的两个项目中issue部分找到解答  
https://github.com/Hanjun-Dai/graph_comb_opt/issues  
https://github.com/Hanjun-Dai/graphnn/issues  



