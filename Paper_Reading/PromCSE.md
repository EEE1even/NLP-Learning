# PromCSE

[《Improved Universal Sentence Embeddings with Prompt-based Contrastive Learning and Energy-based Learning》](https://aclanthology.org/2022.findings-emnlp.220/)

## 什么是Energy-based Learning?

[知乎](https://www.zhihu.com/question/59264464)

设计一个能量函数。在一个数据集中，有数据和标签，如果数据和标签正确（比如图片是猫，标签也是猫），那么这个组合的能量值低。如果数据标签不正确，能量值高。我们希望能量最小化来达到训练目的

> 最大似然估计和似然估计？[知乎](https://zhuanlan.zhihu.com/p/26614750)



## Soft Prompt

受这篇论文启发：[《The Power of Scale for Parameter-Efficient Prompt Tuning》](https://aclanthology.org/2021.emnlp-main.243.pdf)

上述论文受这篇启发：[《Prefix-Tuning: Optimizing Continuous Prompts for Generation》](https://aclanthology.org/2021.acl-long.353.pdf)





## 阅读小结

**以前的方法有什么缺点？**

* 在域偏移（domain shift setting）下的表现欠佳，从而阻碍句子表示在实际中的应用

将这种缺点表现归结于百万参数预训练模型的参数过载

* NT-Xent loss在监督学习中不能完全利用hard negatives（与正样本非常相似但属于不同类别的负样本）
* 对预训练模型百万参数的调整可能会造成训练数据分布的过拟合，导致域偏移时的易损性



**以前的方法怎么做？**

* 使用NT-Xent Loss function来处理监督学习中的句子对
* 总是在大数据集上训练模型并套用在各种任务中



**现在的方法有什么优点？**

* 缓解向量空间在域迁移时质量下降的问题
* 提高了监督学习的向量空间质量



**现在的方法怎么做？**

* 在保持预训练模型固定时（冻结），只训练小范围的soft prompt
* 使用Energy-based Hinge loss来支持原来的损失函数，从而加强辨别能力
* 冻结SimCSE的预训练模型并增加多层可学习的Soft Prompt



