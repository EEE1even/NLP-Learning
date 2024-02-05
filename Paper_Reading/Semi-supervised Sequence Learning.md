# Semi-supervised Sequence Learning

rnn中的反向传播训练非常困难，所以很少用到nlp中的文本分类；

rnn在表示顺序结构方面很强大

发现可以用lstm结合rnn来训练模型，在不增加额外数据的情况下效果能超越原先的lstm模型（baseline）

另一个重要结论：使用更多来自相关任务的无标注数据可以提高后续监督模型的泛化能力

用更多的无标注数据训练的无监督学习可以提高监督学习的效果



两个模型

* 一个是有句子自解码器（SA-LSTM）
  * 这个model有一个sequence autoencoders（用rnn读取输入的长句子到一个单一向量），sequence autoencoders加上外部无标注数据，lstm模型可以比之前的baseline表现的更好
* 一个是加上循环语言模型(LM-LSTM)作为无监督方法
  * 用rnn作为无监督训练方式



优势：简单的fine-tuning

不同于Skip-thought vectors

* 之前的方法是一个更难的目标，因为它用来预测相邻的句子
* 之前的方法是纯粹的无监督学习算法，没有fine-tuning

为什么这个方式有效：梯度有捷径，所以autoencoder可以又好又稳定的初始化循环网络



用无监督学习来优化监督学习，减少句子标注的工作。
