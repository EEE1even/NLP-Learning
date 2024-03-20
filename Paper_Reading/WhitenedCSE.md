# WhitenedCSE

[WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings](https://aclanthology.org/2023.acl-long.677/)

whitening（白化）：去除数据内的冗余信息

白化后特性:

* 线性解耦每个维度的联系。
* 适当缩放每个维度的差异，差异小的放大，差异大的缩小。

对于高相似度的句向量来说，就是放大差异

> 涉及到协方差



白化学习和对比学习在均匀性方面有很大的冗余度



**论文做了什么？**

更好的归一性

更好的对齐性

在sts数据集上刷了sota



**之前的方法怎么做？**

* mask language modeling
* 通过后处理来提高归一性，正则化、白化；将学习好的表征映射到各项同性空间
* 将所有样本散布到特征空间来提高对齐性



**之前的方法有什么缺点？**

* 不能很好的处理归一性和对齐性，不适合句子表征学习(MLM)



**现在的方法怎么做？**

* 给一个主干特征，SWG(shuffled Group Whitening)随机将特征沿着通道轴划分成多组，每组独立进行白化操作，白化后的特征被喂到对比损失进行优化



**现在的方法有什么优点？**

* 虽然典型的白化只对归一性有效，但是论文的方法在提升归一性的同时也提升了对齐性
* 产生的所谓“重复”特征实际上各不相同，这增加了正样本的多样性



> github上开源的代码有bug，目前作者还没有修，建议参考论文思路不建议复现