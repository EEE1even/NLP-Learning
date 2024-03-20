# Bert  issues 

关于bert的一些个人疑问，学的一知半解始终没有透彻理解PLM中的一些观点

* 他是怎么embedding的？
* 词表是什么？
* cls干什么用的？
* 这个双向编码是怎么编的？
* 怎么训练的？
* MLM到底是什么？

写这篇blog学习记录

___________

huggingface发布的关于Bert的文档可以作为一个参考：[huggingface_blog](https://huggingface.co/blog/bert-101)

### Masked Language Model

随机遮盖文本中的一个单词然后让模型训练去预测这个单词，在这个过程中训练模型的权重使得模型的embedding过程中的向量空间更加接近真实空间

> **个人见解：** 接近真实空间之后，模型就更容易对masked的单词进行预测。简单的举个例子 ***{猫吃鱼；狗吃肉} --> {猫 [Mask] 鱼；狗 [Mask] 肉}*** ：将预测过程想象成一个套圈游戏，最开始的分布可能是猫和鱼靠的很近，但是"吃"离得远，我们的圈无法套到这个"吃"。但是在进行训练后，这些词的分布变得均匀，使得我们可以套到正确（接近真实）的东西。

![bert-picture](./src/bert-picture)



### 怎么embedding的？

BERT使用WordPiece嵌入方法处理输入文本。首先，它将文本分割成更小的片段或“tokens”。然后，每个token被转换成对应的词向量。这些词向量不仅包含了单词的语义信息，还加入了位置编码（Position Encoding），以保留单词在句子中的顺序信息。此外，BERT在每个输入序列的开始添加一个特殊的`[CLS]`标记，并为分隔不同句子或段落的地方添加`[SEP]`标记。

```python
#可以在colab进行实践
from transformers import BertTokenizer, BertModel
import torch

# 加载分词器和模型
#uncased不区分大小写，词表上只有小写
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 定义一个句子
sentence = "Hello, BERT. This is a test sentence for embeddings."

# 分词并添加特殊符号
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']

# 转换成PyTorch tensors
token_ids = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([token_ids])

# 获取嵌入
with torch.no_grad():
    outputs = model(tokens_tensor)
    embeddings = outputs.last_hidden_state

print(f"Tokenized Sentence: {tokens}")
print(f"Token IDs: {token_ids}")
# 展示第一个token（[CLS]）的嵌入向量
print(f"Embedding of '[CLS]': {embeddings[0][0]}")
print(f"Embedding '[CLS]' len: {len(embeddings[0][0])}")
print(f"Embedding size: {embeddings.size()}")
```

print:

```bash
Tokenized Sentence: ['[CLS]', 'hello', ',', 'bert', '.', 'this', 'is', 'a', 'test', 'sentence', 'for', 'em', '##bed', '##ding', '##s', '.', '[SEP]']
Token IDs: [101, 7592, 1010, 14324, 1012, 2023, 2003, 1037, 3231, 6251, 2005, 7861, 8270, 4667, 2015, 1012, 102]
Embedding of '[CLS]': tensor(
    	[-1.6189e-01, -1.4461e-01, -1.2250e-01, -1.4407e-01, -3.5012e-01,
        -3.8707e-01,  2.7428e-01,  4.0796e-01,  6.6938e-02, -6.9667e-02,
        ...
        7.0484e-02,  1.6025e-02, -5.3804e-02,  4.9259e-01, -1.4404e-02,
        -3.6002e-01,  1.2062e-01,  8.7608e-01]
		)
Embedding '[CLS]' len: 768
Embedding size: torch.Size([1, 17, 768])
```



### 词表是什么？

词表是预先定义的所有可能token的集合。在BERT中，这个词表是在预训练阶段通过分析预训练数据集构建的，通常包括成千上万的词汇。WordPiece方法允许BERT有效处理未知词汇，通过将未知单词分解成已知的较小片段。

* #### 词表是提前训练好的吗？

​	词表是在BERT模型的预训练阶段之前就已经确定好的。词表的构建是通过分析预训练数据集来完成的，目的是识别和编码数据集中所有唯一的词汇和词片段（token）。BERT使用的是WordPiece算法来生成这个词表。

* #### 词表会根据模型训练进行自动调整吗？还是说固定不变？

​	一旦词表被创建并在预训练过程中使用，它就不会根据模型在后续训练或微调过程中的表现进行自动调整或更新。这意味着所有的训练过程，包括预训练和针对特定任务的微调，都依赖于相同的固定词表。

**兼容性**：保持词表不变确保了模型的输出和行为在整个使用周期内的一致性。这对于模型部署和实际应用是非常重要的，因为改变词表可能导致模型需要重新训练或评估。

**稳定性**：使用固定词表可以避免训练过程中引入不必要的变化和不确定性。如果词表在训练过程中发生变化，模型的学习目标也会随之变化，可能导致模型性能的波动或下降。

**效率**：确定并优化一个词表是一个计算成本高昂的过程，特别是在处理大规模数据集时。一旦建立了一个高效且有效的词表，固定使用它可以节省大量的计算资源和时间。



### cls干什么用的？

`[CLS]`是一个特殊的标记，添加在每个输入序列的开始。在BERT的上下文中，`[CLS]`标记的输出向量（即经过所有Transformer层处理后的`[CLS]`的最终向量）被用来表示整个输入序列的综合信息。这个向量可以用于各种下游任务，比如分类任务，在这种情况下，`[CLS]`向量被送入一个或多个全连接层以预测类别标签。

* #### cls向量的拼接方式？是[cls]A[sep] [cls]B[sep] ; 还是[cls]A[sep]B[sep]?

​	处理双句子任务（如句子对分类、问答任务中的问题和段落对比等）时，正确的拼接方式是将两个句子合并为一个输入序列，而不是将每个句子单独拼接`[CLS]`和`[SEP]`。

```bash
#双句子任务
[CLS] A [SEP] B [SEP]
```

​	在处理单句子任务（分类，预测）时，会对句子做`[CLS]`标注，但是标签不会参与

```bash
#单句子任务
[CLS] This movie is fantastic! [SEP]
```

* #### 为什么要用cls标记？别的不行吗？

​	想象一下，你有一堆积木（这里的积木就像是文字），你要用这些积木来建造一个房子（这个房子就是句子的意思）。现在，假设每个积木都有自己的特点，比如颜色、形状等（这些特点就像是词语的含义），而你想通过组合这些积木的方式，来让别人一眼就能看出这是一个房子（也就是让计算机理解句子的意思）。在BERT模型中，`[CLS]`标记就像是你要建造的房子的基石。你首先放下这个基石，然后围绕它来组装其他的积木。在你完成房子（句子）之后，这个基石（`[CLS]`标记）就会变得与众不同，因为它吸收了周围所有积木（词语）的特点，变成了一个能代表整个房子（句子含义）的超级积木。

​	当我们需要计算机做些事情，比如判断一句话是开心的还是难过的时候，计算机就会看这个超级积木（`[CLS]`标记的向量），因为它包含了整个句子的信息。这样，计算机就可以通过这个基石来快速理解整个句子的大致意思，而不需要再去一块一块拆解积木地分析了。



### 这个双向编码是怎么编的？

BERT的核心特性之一就是其双向编码能力，这得益于Transformer模型的自注意力（Self-Attention）机制。不同于之前的模型（如RNN或LSTM）只能按照单一方向（从左到右或从右到左）处理序列，BERT通过自注意力机制能够在编码每个token时考虑到整个句子中的所有单词，从而实现真正的双向理解。这意味着BERT在处理任何单词时，都会考虑到它在文本中前后的上下文信息。

```python
from transformers import BertTokenizer, BertModel
import torch

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入
text = "Here is some text to encode"
encoded_input = tokenizer(text, return_tensors='pt')# encoded_input.keys() ==> dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

# 获取模型输出
with torch.no_grad():
    output = model(**encoded_input)
    #output.key() ==> dict_keys(['last_hidden_state', 'pooler_output'])

# 获取最后一层的隐藏状态，它包含了句子中每个词的嵌入表示
last_hidden_states = output.last_hidden_state

# 打印第一个词的嵌入表示（注意，由于添加了特殊标记[CLS]和[SEP]，所以实际词的索引要向后偏移）
print(last_hidden_states[0][1])
======================================print=====================================
# len(tensor) = 768
tensor([-5.7594e-01, -3.6501e-01, -1.3834e-01, -4.8850e-01,  1.6220e-01,
        -3.2118e-01, -1.5293e-01,  8.2259e-01, -2.6262e-01,  6.4759e-01,
        ...
        -7.5416e-03, -3.8632e-01, -4.4265e-01,  7.2205e-01, -3.5554e-02,
        -6.7818e-01,  2.0924e-01, -1.6394e-01]) 

```

* #### 如何观察这个双向编码过程

​	直接通过代码查看BERT的双向编码过程比较抽象，因为这一过程发生在模型内部。我们可以通过可视化注意力权重来间接观察BERT如何在编码时考虑一个词的上下文信息。

```python
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import numpy as np

# 初始化分词器和模型，确保模型被配置为输出注意力权重
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# 准备输入并获取模型输出
text = "Here is some text to encode"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# 获取第一层的所有头的注意力权重，并计算平均
attentions = output.attentions[0]  # 获取第一层的注意力权重
average_attentions = attentions.mean(dim=1)[0]  # 计算所有头的平均，取第一个样本

# 可视化平均注意力权重
plt.figure(figsize=(10, 8))
plt.imshow(average_attentions.detach().numpy(), cmap='viridis')

# 设置图表的标签
tokens = tokenizer.tokenize(text)
tokens = ['[CLS]'] + tokens + ['[SEP]']
plt.xticks(range(len(tokens)), tokens, rotation='vertical')
plt.yticks(range(len(tokens)), tokens)

plt.colorbar()
plt.show()
```

![bert-attention](./src/bert-attention)

上述的热力图可以体现词与词之间的注意力机制，颜色越浅关注度越高

```bash
print(average_attentions)
================================print====================================
tensor([[0.5816, 0.0537, 0.0278, 0.0491, 0.0344, 0.0906, 0.0495, 0.0314, 0.0820],
        [0.1994, 0.1106, 0.1390, 0.1029, 0.1695, 0.0554, 0.0487, 0.1013, 0.0733],
        [0.1360, 0.2854, 0.0973, 0.1502, 0.1128, 0.0354, 0.0376, 0.0834, 0.0620],
        [0.1396, 0.1915, 0.1803, 0.0876, 0.1416, 0.0675, 0.0559, 0.0657, 0.0704],
        [0.1014, 0.1025, 0.0843, 0.1158, 0.1286, 0.0860, 0.0488, 0.2489, 0.0837],
        [0.1427, 0.0974, 0.0871, 0.1238, 0.1642, 0.0631, 0.1272, 0.0884, 0.1061],
        [0.1830, 0.0640, 0.0636, 0.0537, 0.1148, 0.0693, 0.0858, 0.2486, 0.1173],
        [0.1604, 0.0670, 0.0297, 0.0409, 0.1872, 0.0670, 0.2358, 0.0779, 0.1341],
        [0.2865, 0.0593, 0.0590, 0.0522, 0.0517, 0.0779, 0.0930, 0.1029, 0.2175]],
       grad_fn=<SelectBackward0>)
```



### 怎么训练的？

BERT的训练分为两个阶段：预训练和微调。

- **预训练**：在这一阶段，BERT在大规模文本语料库上进行训练，学习语言的通用特征。这一阶段主要包括两种类型的任务：掩码语言模型（MLM）和下一个句子预测（NSP）。

  > 但在后续的一些实验中表示：NSP方法对模型并没有很大的提高甚至没啥用

- **微调**：在预训练完成后，BERT可以通过在特定任务上的额外训练进行微调，如情感分析、问题回答或文本分类等。在这一阶段，整个模型的参数都会稍作调整以适应特定任务。





































