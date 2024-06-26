# 对比表示学习（二）Setence Embedding

> 参考翁莉莲的[Blog](https://lilianweng.github.io/posts/2021-05-31-contrastive/#language-sentence-embedding)，本章主要阐述句子嵌入（sentence embedding）

个人见解：对比学习最早始于CV，但是由于对比学习在使用大量无标签数据时对向量空间的优化很好（[SimCSE](https://arxiv.org/abs/2104.08821)），所以目前使用对比学习来优化embedding space是一个不错的方法

## 文本扩增

绝大部分视觉应用中的对比方法依赖于创造每个图像的增强版本，但是在句子扩增中会变得非常有挑战性。因为不同于图片，在扩增句子的同时非常容易改变句子本身的语义。

### Lexical Edit（词汇编辑）

**EDA**（*Easy Data Augmentation*；[paper](https://arxiv.org/abs/1901.11196)）定义了一组简单但有效的文本增强操作。给定一个句子，EDA随机选择以下四种操作：

* 同义词替换（Synonym replacement）：用同义词随机替换 $n$ 个单词（不能是stop word）。

* 随机插入（Random insertion）：在句子的随机位置插入一个随机选择的非停用词的同义词。

* 随机交换（Random swap）：随机交换两个词，并重复此操作 $n$ 次。

* 随机删除（Random deletion）：以一定概率 $p$ 随机删除句子中的每个词。

其中 $p = \alpha$ 和 $n = \alpha \times \text{sentence\_length}$，按照直觉来看，长句子在吸收更多噪声同时能保持原始标签（原始句子的意思）。超参数 $\alpha$ 大致指示一次增强可能改变的句子中的单词百分比。

研究表明，与没有使用EDA的基线相比，EDA能够在多个分类基准数据集上提高分类准确性。在较小的训练集上，性能提升更为显著。EDA中的所有四种操作都有助于提高分类准确性，但在不同的 $\alpha$ 值下达到最优(参考下图)。

![image-20240527193057015](.\src\EDA.png)

在上下文增强（[Sosuke Kobayashi, 2018](https://arxiv.org/abs/1805.06201)）中，位于位置 $i$ 的单词 $ w_i $ 的替换可以从给定的概率分布 $p(\cdot | S \setminus \{w_i\})$中平滑采样，该分布由双向语言模型如BERT预测。



### Back-translation（回译）

**CERT** (*Contrastive self-supervised Encoder Representations from Transformers*;[paper](https://arxiv.org/abs/2005.12766)) 通过回译的方式来产生增强后的数据。不同语言的各种翻译模型可用于创建不同方式的数据增强。一但我们有了文本样本的噪声版本，就可以通过对比学习框架来训练sentence embedding



### Dropout and Cutoff

**Cutoff：**[Shen et al.(2020)](https://arxiv.org/abs/2009.13818)受跨视图训练的启发，提出将截止值应用于文本增强。他们提出了三种截断扩增策略：

* 标记符截断（Token cutoff）会删除一些选定标记符的信息。为确保没有数据泄露，输入(input)、位置(positional)和其他相关嵌入矩阵(embedding matrice)中的相应标记应全部清零
* 特征截断删除一些特征列。
* 跨度截断删除连续的文本块。

![image-20240527194919759](.\src\cutoff.png)

一个样本可以创建多个增强版本。在训练时，使用了额外的 KL-发散项来衡量不同增强样本预测之间的一致性。

**SimCSE：**[Gao et al.](https://arxiv.org/abs/2104.08821)；在无监督数据中学习时，只需通过句子本身进行预测，将dropout作为噪声。换句话说，他们将dropout视为文本序列的数据增强。一个样本只需输入编码器两次，这两个版本就是正对样本，而其他批次中的样本则被视为负对样本。这种方法感觉上与cutoff很相似，但dropout相对于cutoff处理更灵活。

![image-20240527195620820](.\src\simcse.png)

> 相关内容可以参考阅读笔记 ：[Notes](https://github.com/EEE1even/SimCSE_paper_reading)



### Supervision from  Natural Language Inference

在语义相似性任务中，未经任何微调的预训练 BERT 句子嵌入性能不佳。因此，我们不能直接使用原始的嵌入，而需要通过进一步微调来完善嵌入。

Natural Language Inference（NLI）任务是为句子嵌入学习提供监督信号的主要数据源，如 [SNLI](https://nlp.stanford.edu/projects/snli/)、[MNLI](https://cims.nyu.edu/~sbowman/multinli/) 和 [QQP](https://www.kaggle.com/c/quora-question-pairs)。

#### Sentence-BERT

**SBERT：** ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)) 依赖于暹罗（Siamese）和三重（triplet）网络架构来学习句子嵌入，这样，句子的相似性就可以通过嵌入对之间的余弦相似性来估算。需要注意的是这个框架依赖于监督数据集。实验效果的好坏取决于数据集，所以没有一个较好的优势。



#### BERT-flow

如果嵌入是均匀分布在每一个维度中，那么嵌入空间就被认为是各向同性，反之则为各项异性。 [Li et al, (2020)](https://arxiv.org/abs/2011.05864) 在论文中表示，预训练的bert模型学习到了一个非平滑的各向异性的语义嵌入空间，所以导致了在没有进行微调的情况下，在语义相似任务中的垃圾表现。 根据经验，他们发现 BERT 句子嵌入存在两个问题： 词频会使嵌入空间产生偏差。高频词靠近原点，而低频词远离原点。低频词分布稀疏。低频词的嵌入往往离其 k-NN 邻近词较远，而高频词的嵌入则较为密集。**BERT-flow**通过归一化流将嵌入转化为平滑和各向同性的高斯分布。 

![image-20240527202321127](.\src\bertflow.png)

让 $\mathcal{U}$ 代表观察到的BERT句子嵌入空间，$\mathcal{Z}$ 为期望的潜在空间，它是一个标准高斯分布。因此，$p_{\mathcal{Z}}$ 是一个高斯密度函数，并且 $f_{\phi} : \mathcal{Z} \rightarrow \mathcal{U}$ 是一个可逆变换：

$$
z \sim p_{\mathcal{Z}}(z), \quad u = f_{\phi}(z), \quad z = f_{\phi}^{-1}(u)
$$

一个基于流的生成模型通过最大化 $\mathcal{U}$ 的边际似然来学习这个可逆映射函数：

$$
\max_{\phi} \mathbb{E}_{u=\text{BERT}(s), s\sim \mathcal{D}} \left[ \log p_{\mathcal{Z}}(f_{\phi}^{-1}(u)) + \log \left| \det \frac{\partial f_{\phi}^{-1}(u)}{\partial u} \right| \right]
$$

其中 $s$ 是从文本语料库 $\mathcal{D}$ 中采样的句子。只有流参数 $\phi$ 在优化过程中被优化，而预训练的BERT中的参数保持不变。

BERT-flow已经被证明可以提高大多数语义文本相似性（STS）任务的性能，无论是否有NLI数据集的监督。因为学习用于校准的归一化流不需要标签，它可以利用整个数据集，包括验证集和测试集。



#### Whitening Operation

[Su et al. (2021)](https://arxiv.org/abs/2103.15316) 应用了白化操作来改善学习表示的各向同性，并减少句子嵌入的维度。

他们将句子向量的均值变换设置为0，协方差矩阵变换为单位矩阵。给定一组样本 $\{x_i\}_{i=1}^N$，让 $\tilde{x}_i$ 和 $\tilde{\Sigma}$ 为变换后的样本和相应的协方差矩阵：

$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i, \quad \Sigma = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T
$$

$$
\tilde{x}_i = (x_i - \mu)W, \quad \tilde{\Sigma} = W^T \Sigma W = I \text{，因此 } \Sigma = (W^{-1})^T W^{-1}
$$

如果我们得到 $\Sigma$ 的奇异值分解（SVD）$U \Lambda U^T$，我们将有 $W^{-1} = \sqrt{\Lambda} U^T$ 和 $W = U \Lambda^{-\frac{1}{2}}$。注意，在SVD中，$U$ 是一个正交矩阵，其列向量为特征向量，$\Lambda$ 是一个对角矩阵，包含按顺序排列的正特征值。

可以通过只取 $W$ 的前 $k$ 列来应用降维策略，这种方法被称为$whitening-k$。

在许多 STS 基准测试中，无论是否有 NLI 监督，白化操作都优于 BERT-flow，并在 256 个句子维度上实现了 SOTA。



### Unsupervised Sentence Embedding Learning 

#### Context Prediction

**Quick-Thought （QT）vectors** ([Logeswaran & Lee, 2018](https://arxiv.org/abs/1803.02893))将句子表征学习表述为一个分类问题：给定一个句子及其上下文，分类器根据其向量表征（"cloze test"）将上下文句子与其他对比句子区分开来。这样的表述去除了会导致训练速度减慢的 softmax 输出层。

![image-20240527203926189](.\src\qt.png)

让 $f(\cdot)$ 和 $g(\cdot)$ 是两个将句子 $s$ 编码成固定长度向量的函数。设 $C(s)$ 为在 $s$ 上下文中的句子集合，$S(s)$ 是候选句子集合，包括只有一个真实上下文句子 $s_c \in S(s)$ 和许多其他非上下文负句子。Quick Thoughts模型学习优化预测唯一真实上下文句子 $s_c$ 的概率。它本质上是在考虑句子对 $(s, s_c)$ 为正样本对时使用NCE损失，而其他对 $(s, s')$ 其中 $s' \in S(s), s' \neq s_c$ 作为负样本。

$$
\mathcal{L}_{QT} = -\sum_{s \in D} \sum_{s_c \in C(s)} \log p(s_c | s, S(s)) = -\sum_{s \in D} \sum_{s_c \in C(s)} \log \frac{\exp(f(s)^T g(s_c))}{\sum_{s' \in S(s)} \exp(f(s)^T g(s'))}
$$

这个损失函数计算每个句子$s$及其对应的上下文句子$s_c$的对数概率，相对于所有候选句子的得分的归一化。这有助于模型学习区分正确的上下文句子与其他不相关的句子。



#### Mutual Information Maximization

**IS-BERT (Info-Sentence BERT)**([Zhang et al. 2020](https://arxiv.org/abs/2009.12061))采用基于相互信息最大化的自监督学习目标，以无监督方式学习良好的句子嵌入。

IS-BERT的工作流程如下：

1. 使用BERT将输入句子 $s$ 编码为长度为 $l$ 的令牌嵌入 $h_{1:l}$。
2. 然后应用不同核大小（例如1, 3, 5）的1-D卷积网络来处理令牌嵌入序列以捕获n-gram局部上下文依赖性：$c_i = \text{ReLU}(w \cdot h_{i:i+k-1} + b)$。输出序列被填充以保持输入的相同尺寸。
3. 第 $i$ 个令牌的最终局部表示 $\mathcal{F}_\theta^{(i)}(x)$ 是不同核大小表示的拼接。
4. 通过在令牌表示 $\mathcal{F}_\theta(x) = \{\mathcal{F}_\theta^{(i)}(x) \in \mathbb{R}^{d \times l}\}_{i=1}^l$ 上应用时间平均池化层计算全局句子表示 $\xi_\theta(x)$。

由于互信息估计通常对于连续和高维随机变量来说是难以处理的，IS-BERT依赖于Jensen-Shannon估计器([Nowozin et al., 2016](https://arxiv.org/abs/1606.00709), [Hjelm et al., 2019](https://arxiv.org/abs/1808.06670))来最大化 $\mathcal{E}_\theta(x)$ 和 $\mathcal{F}_\theta^{(i)}(x)$ 之间的互信息：
$$
I_{JSD}^\omega(\mathcal{F}_\theta^{(i)}(x); \mathcal{E}_\theta(x)) = \mathbb{E}_{x \sim P}[-\text{sp}(-T_\omega(\mathcal{F}_\theta^{(i)}(x); \mathcal{E}_\theta(x)))] - \mathbb{E}_{x \sim P, x' \sim \tilde{P}}[\text{sp}(T_\omega(\mathcal{F}_\theta^{(i)}(x'); \mathcal{E}_\theta(x)))]
$$
其中 $T_\omega : \mathcal{F} \times \mathcal{E} \rightarrow \mathbb{R}$ 是一个带参数 $\omega$ 的可学习网络，用于生成判别器得分。负样本 $x'$ 是从分布 $\tilde{P} = P$ 中采样的。$\text{sp}(x) = \log(1 + e^x)$ 是softmax激活函数。

