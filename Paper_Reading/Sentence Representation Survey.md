# Sentence Representation Survey

[A Comprehensive Survey of Sentence Representations: From the BERT Epoch to the CHATGPT Era and Beyond](https://aclanthology.org/2024.eacl-long.104.pdf)

### 1、Background

里程碑时间轴

![image-20240326095151683](./src/sentenceRepresentationSurvey)

有监督与无监督的一些方法汇总

![image-20240326095505829](./src/SRS2)

组成部分：

![image-20240326095702240](./src/SRS3)

*  **Data:** 使用监督学习的标注数据、或是一些基于正例和负例转换产生的数据
*  **Model：** 在大量文本上训练好的预训练模型
*  **Transform：** 对模型中的表示进行转换来获取句子表征
*  **Loss：** 用来将相似句子拉近，不相似句子拉远

上述的四个部分在不同程度上影响句子的表征

![image-20240326104228442](./src/SRS4)

## 2、Supervised 

NLI是评估NLU的一个代理



## 3、Unsupervised

可以看到上图片的绝大部分是围绕无监督学习来做，因为监督学习所需要的标注数据非常难得、

目的是为了获取正样本的数据表示（可以理解为正样本的标签）

### Better Positives

需要仔细的为对比学习选择正负样本对。因为一些数据扩增的改动可能完全改变句子的意思

### Surface Level

改善句子surface特征会使模型依赖捷径而不是学习语义；需要更有效的数据扩增策略

### Model Level ： 

SimCSE中的dropout方法

专用附件（special component）可以被训练来产生语义相似表示

### Representation Level

通过模型生成句子的潜在表示能够带来重要的好处。在这样的场景中，可以通过探索表示空间来发现积极的示例。这种方法的明显优势在于它消除了数据增强的需要（当使用由模型生成的句子的潜在表示作为句子表示时，因为这些表示已经包含了丰富的信息和特征，所以就没有必要再进行数据增强了）。

### Alternative Methods

在缺少大量标注数据的情况下如何通过各种创新方法提取和改进句子的数学表示，以便在机器学习中更有效地使用这些表示。

### Alternative Loss and Obejectives

对比损失函数的局限性，例如缺乏机制来整合hard negatives(难以从正样本对中区分的负样本)

* 补充损失
* 修改对比损失
* 不同的损失

### Better Negative Sampling

选择负样本与选择正样本同样重要

如果不能选择好的负样本，会使得损耗梯度逐渐减小，阻碍模型学习有用的表征

足量的负样本也很重要

### Post-Precessing

将词向量变成固定的长度

归一化流（normalizing flows）

白化

后处理技术只在bert上测试了很久，在现在更新的模型上还没有答案



## 4、Trends

**Limited advantages of supervision：** 目前的sentence representation在有监督的学习下有事有限。主要还是在无监督学习中进行句子表征的训练学习

**Downplaying downstream task evaluation:**  在下游任务评估中对句子表征忽视；利用句子表征来强化few-shot文本分类

**Data-centric innovations:**  针对数据进行创新

**Keeping up with LLMs:**  在有高质量的数据集时，对比学习相较于提高模型参数来说能产生更好的结果



## 5、Challenges

实际应用与工具的兴起

适应不同领域

跨语言的句子表征

句子表征的普遍性
