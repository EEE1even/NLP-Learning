# Universal Language Model Fine-tuning for Text Classification

**论文主要表达**

提出了关于nlp的泛化模型，以及关键训练步骤（以前的方法都要从零开始训练）

100个标注数据的训练效果可以比之前好100倍



**铺垫知识**

inductive learning（归纳式学习）对应于meta-learning 从诸多给定任务中学习然后迁移到陌生任务中去

transductive learning（直推式学习）对应domain adaptation 给定的数据包含目标域的数据，要求训练一个对目标域数据又最小误差的模型



**以前的方法有什么问题？**

* 从零开始训练，成本大
* 将预训练的embedding当作固定参数对待，限制了参数的有效信息表达

> thinking
>
> 以前的方式都是随机初始化模型参数，现在追寻的是使用特定方式来初始化参数，以求在其他条件不变的情况下追求更好的效果

* 使用fine-tuning的归纳式迁移学习在nlp中很失败
* 以前的方法需要大量in-domain数据来达到很好的表现，限制了LM的应用
* 缺少对LM训练方面的知识，一直阻碍着更广泛的应用
* nlp模型通常较为浅（shallow），需要不同的微调方法
  * shallow与deep形成比较，相对于cv而言，nlp的模型更加浅



**现在方法解决了什么**？

* 解决了nlp模型泛化能力不足的问题，可以更广泛的采用



**现在怎么做？**

* 使用判别微调（*discriminative fine-tuning*）
* 斜三角学习率（*slanted triangular learning rates*）
* 逐渐解冻策略（*gradual unfreezing*）

通过这些方式来保留之前学习的知识，防止在fine-tuning时发生灾难性的遗忘

