# MLOps
本工程为部署于k8s集群的MLOps流水线，流水线主体为Jenkins，使用prometheus监控集群性能，通过grafana生成监控数据看板。
## 目录介绍
### jenkins
存放可部署于k8s的jenkins定义文件及相关脚本

jenkins用于搭建流水线

### seldon
存放可部署于k8s的seldon安装脚本

seldon用于部署模型

### istio
存放可部署于k8s的istio定义文件及相关脚本

istio用于网管控制，结合seldon使用


### prometheus
存放可部署于k8s的prometheus定义文件及安装脚本

prometheus用于数据监控

### grafana
存放可部署于k8s的grafana定义文件及安装脚本

grafana用于数据展示

## 使用方式
### 前置条件
#### 软件需求
一个k8s集群（可使用minikube创建的本地集群），已知可用最新版本为v1.21.4（最新seldon支持版本）

安装kubectl、docker、istoctl、s2i
#### 硬件需求
2核CPU、8G内存

### 部署服务
使用各文件夹下（example除外）安装脚本在集群中部署对应服务

## 示例代码
可见example文件夹下kdd99_dagmm示例，该示例为一条涵盖测试、部署、归档三阶段流水线，详见其中README