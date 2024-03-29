# ConSERT

[A Contrastive Framework for Self-Supervised Sentence Representation Transfer](https://aclanthology.org/2021.acl-long.393.pdf)

1. **论文发布之前的方法**：
   - 之前的方法主要包括基于BERT的预训练语言模型，这些模型虽然在许多下游任务中表现出色，但直接从BERT中衍生的句子表示证明在语义文本相似性（STS）任务上表现不佳，几乎所有句子对的相似性得分都在0.6到1.0之间，即使一些句子对被人类注释者视为完全无关。
2. **以前的方法有什么缺点**：
   - 之前的方法存在的缺点是BERT衍生的句子表示存在崩溃问题，几乎所有句子都被映射到一个小区域内，因此产生高相似性。这主要是因为BERT的词表示空间是各向异性的，高频词聚集且靠近原点，而低频词稀疏分布。
3. **这篇论文的创新点**：
   - ConSERT通过对比学习在无监督的方式中有效地微调BERT，解决了BERT衍生句子表示的崩溃问题，并使它们更适用于下游任务。此外，论文提出了多种数据增强策略，包括对抗攻击、令牌洗牌、剪切和丢弃，这些策略有效地转移了句子表示到下游任务。
4. **这篇论文的方法有什么优点**：
   - ConSERT的优点包括不引入额外结构或特殊实现、与BERT相同的参数大小、高效性以及包含多种有效且方便的数据增强方法。特别地，在只有1000个未标记文本的情况下，ConSERT就能显著提升性能，展现了其在数据稀缺场景下的鲁棒性。
5. **这篇论文是怎么处理的**：
   - ConSERT使用对比学习的方法，通过鼓励来自同一句子的两个增强视图更接近，同时保持与其他句子的视图距离，重塑了BERT衍生的句子表示空间。对于每个输入文本，应用数据增强模块生成两个版本的令牌嵌入，然后通过BERT编码器计算句子表示，并通过平均池化获得。