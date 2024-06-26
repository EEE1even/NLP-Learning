# Contrastive Representation Learning

> 参考翁莉莲的[Blog](https://lilianweng.github.io/posts/2021-05-31-contrastive/#contrastive-loss)，本文主要针对于NLP中的对比学习

对比表示学习可以用来优化嵌入空间，使相似的数据靠近，不相似的数据拉远。同时在面对无监督数据集时，对比学习是一种极其有效的自监督学习方式

## 对比学习目标

在最早期的对比学习中只有一个正样本和一个负样本进行比较，在当前的训练目标中，一个批次的数据集中可以有多个正样本和负样本对。

### 对比损失函数

#### Contrastive loss 

该[论文](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)是以对比方式进行深度度量学习（deep metric learning）的最早训练目标之一

给定一组输入样本 $$\{x_i\}$$，每个样本都有一个对应的标签 $$y_i \in \{1, \dots, L\}$$，共有 $$L$$ 个类别。我们希望学习一个函数 $$f_{\theta}(\cdot) : \mathcal{X} \rightarrow \mathbb{R}^d$$，该函数能将 $$x_i$$ 编码成一个嵌入向量，使得同一类别的样本具有相似的嵌入，而不同类别的样本具有非常不同的嵌入。因此，对比损失（Contrastive Loss）会取一对输入 $$(x_i, x_j)$$，并最小化同一类别样本间的嵌入距离，同时最大化不同类别样本间的嵌入距离。
$$
\mathcal{L}_{\text{cont}}(x_i, x_j, \theta) = \mathbf{1}[y_i = y_j] \left\| f_\theta(x_i) - f_\theta(x_j) \right\|^2 + \mathbf{1}[y_i \neq y_j] \max(0, \epsilon - \left\| f_\theta(x_i) - f_\theta(x_j) \right\|^2)
$$
其中 $$\epsilon$$​ 是一个超参数，用来定义不同类别样本的最低下界。

#### Triplet loss

参考[论文](https://arxiv.org/abs/1503.03832)，提出的目的是用来学习在不同姿势和角度下对同一个人进行人脸识别。

给定一个锚定输入 $x$，我们选择一个正样本 $x^+$ 和一个负样本 $x^-$，意味着 $x^+$ 和 $x$ 属于同一类，而 $x^-$ 则来自另一个不同的类。三元组损失（Triplet Loss）通过以下公式学习，同时最小化锚定 $x$ 和正样本 $x^+$ 之间的距离，并最大化锚定 $x$ 和负样本 $x^-$​ 之间的距离：
$$
\mathcal{L}_{\text{triplet}}(x, x^+, x^-) = \sum_{x \in \mathcal{X}} \max \left(0, \|f(x) - f(x^+)\|^2 - \|f(x) - f(x^-)\|^2 + \epsilon \right)
$$
其中，边界参数 $\epsilon$ 被配置为相似对与不相似对之间距离的最小偏移量。



#### Lifted Structured Loss



