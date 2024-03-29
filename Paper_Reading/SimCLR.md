# SimCLR

[arxiv](https://arxiv.org/abs/2002.05709)

> 该论文为CV task

论文发现：

* 扩增数据的构成对于预测任务很重要
* 一个在表征与对比损失之间的可学习非线性变换，提高学习到的表征质量
* 相较于监督学习，对比学习受益于更大的batch size和更深更广的训练网络（比监督学习更依赖数据扩增）
* 使用对比交叉熵损失函数的表征学习受益于正则化的emb



**以前的方法怎么做？**

* 设计专业的架构（specialized architectures）
* 使用记忆库（memory bank）



**以前的方法有什么缺点？**

* 繁琐且效果不好



**现在的方法有什么优点？**

* 架构简单
* 效果SOTA



**现在的方法怎么做？**

* 使用对比学习框架

![image-20240308102346598](./src/SImCLR)