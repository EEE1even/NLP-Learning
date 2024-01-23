# Transformer QA

[**原文链接**](https://www.qinglite.cn/doc/701264759473e2afe)，这里做一些知识补充

**1、为什么Transformer要使用多头注意力机制？**

* 为了解决模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身位置的问题
* 一定程度上ℎ越大整个模型的表达能力越强，越能提高模型对于注意力权重的合理分配
* [知乎链接](https://www.zhihu.com/question/341222779)



**2、Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？**

* 不同的矩阵可以保证在不同的空间进行投影，增强表达能力和泛化能力
* [中文链接](https://www.zhihu.com/question/319339652)
* [英文链接](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)



**3、Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？**

* 为了计算更快。矩阵加法在加法这一块的计算量确实简单，但是作为一个整体计算attention的时候相当于一个隐层，整体计算量和点积相似。在效果上来说，从实验分析，两者的效果和dk相关，dk越大，加法的效果越显著。
* 当前矩阵乘法有非常成熟的加速实现
* 可以参考第四问的链接，里面也对这个问提做出了一定的解释



**4、为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解**

* 简单的归纳：attention有加分和乘法两种，在乘法attention中，极大的点积值会会将整个softmax推向梯度平缓区，使得收敛困难，所以要对其进行scaled
* [详细解释](https://www.zhihu.com/question/339723385)



**5、在计算attention score的时候如何对padding做mask操作？**

* 对需要mask的位置设为负无穷，再对attention score进行相加
* 目前没有找到针对该问题的文章，如果想要深入了解可以参考源码



**6、为什么在进行多头注意力的时候需要对每个head进行降维？（可以参考上面一个问题）**

* 将原有的高维空间转化为多个低维空间并再最后进行拼接，形成同样维度的输出，借此丰富特性信息，降低了计算量
* 在**不增加时间复杂度**的情况下，同时，借鉴**CNN多核**的思想，在**更低的维度**，在**多个独立的特征空间**，**更容易**学习到更丰富的特征信息。

​	[知乎海晨威老师的回答](https://www.zhihu.com/question/350369171/answer/3304713324) 



**7、大概讲一下Transformer的Encoder模块？**

* 输入嵌入-加上位置编码-多个编码器层（每个编码器层包含全连接层，多头注意力层和点式前馈网络层（包含激活函数层））

* [链接](https://ifwind.github.io/2021/08/18/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%888%EF%BC%89Transformer%E6%A8%A1%E5%9E%8B/)



**8、为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？**

* embedding matrix的初始化方式是xavier init，这种方式的方差是1/embedding size，因此乘以embedding size的开方使得embedding matrix的方差是1，在这个scale下可能更有利于embedding matrix的收敛。
* [知乎链接](https://zhuanlan.zhihu.com/p/442509602)



**9、简单介绍一下Transformer的位置编码？有什么意义和优缺点？**

* 因为self-attention是位置无关的，无论句子的顺序是什么样的，通过self-attention计算的token的hidden embedding都是一样的，这显然不符合人类的思维。因此要有一个办法能够在模型中表达出一个token的位置信息，transformer使用了固定的positional encoding来表示token在句子中的绝对位置信息。
* [位置编码](https://zhuanlan.zhihu.com/p/106644634)  [意义](https://zhuanlan.zhihu.com/p/630082091)

**10、你还了解哪些关于位置编码的技术，各自的优缺点是什么？**

* 相对位置编码（RPE）1.在计算attention score和weighted value时各加入一个可训练的表示相对位置的参数。2.在生成多头注意力时，把对key来说将绝对位置转换为相对query的位置3.复数域函数，已知一个词在某个位置的词向量表示，可以计算出它在任何位置的词向量表示。前两个方法是词向量+位置编码，属于亡羊补牢，复数域是生成词向量的时候即生成对应的位置信息。

* 关于位置编码我也没有太多的学习，会在之后的更新补充

  

**11、简单讲一下Transformer中的残差结构以及意义。**

* encoder和decoder的self-attention层和ffn层都有残差连接。反向传播的时候不会造成梯度消失。
* [知乎链接](https://zhuanlan.zhihu.com/p/459065530)



**12、为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？**





13、简答讲一下BatchNorm技术，以及它的优缺点。

14、简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？

15、Encoder端和Decoder端是如何进行交互的？（在这里可以问一下关于seq2seq的attention知识）

16、Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？（为什么需要decoder自注意力需要进行 sequence mask)

17、Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？

18、简单描述一下wordpiece model 和 byte pair encoding，有实际应用过吗？

19、Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？

20、引申一个关于bert问题，bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？