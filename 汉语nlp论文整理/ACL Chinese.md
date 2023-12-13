## 2023 ACL Chinese

**[Fantastic Expressions and Where to Find Them: Chinese Simile Generation with Multiple Constraints](https://aclanthology.org/2023.acl-long.28/)**

* 引入 controllable simile generation  (CSG) 可控比拟语生成    **新任务**
* 提出GraCe数据集，有61.3k的比喻元素注释的中文比喻
* 提出CSG模型Similor

**[CATS: A Pragmatic Chinese Answer-to-Sequence Dataset with Large Scale and High Quality](https://aclanthology.org/2023.acl-long.168/)**

* 高质量中文数据集
* 提出统一图表转换方法

**[Advancing Multi-Criteria Chinese Word Segmentation Through Criterion Classification and Denoising](https://aclanthology.org/2023.acl-long.356/)**

* 一个简单的基于输入提示的MCCWS模型，可以达到SOTA

**[READIN: A Chinese Multi-Task Benchmark with Realistic and Diverse Input Noises](https://aclanthology.org/2023.acl-long.460/)**

* 评测数据集

**[Multitask Pre-training of Modular Prompt for Chinese Few-Shot Learning](https://aclanthology.org/2023.acl-long.625/)**

* 中文少样本学习
* [code](https://github.com/Hzfinfdu/MPMP)

**[CHBias: Bias Evaluation and Mitigation of Chinese Conversational Language Models](https://aclanthology.org/2023.acl-long.757/)**

* 提出中文偏见数据集 CHBias
* [github](https://github.com/hyintell/CHBias)



**[Are Pre-trained Language Models Useful for Model Ensemble in Chinese Grammatical Error Correction?](https://aclanthology.org/2023.acl-short.77/)**

* 发现模型合并的PLM对中文语法错误纠正效果非常差
* 测试数据中正确句子的人类参考资料远远不够，而正确句子与地道句子之间的差距值得关注
* 提供了改进方法[code](https://github.com/JamyDon/PLM-based-CGEC-Model-Ensemble)



**[Enhancing Ancient Chinese Understanding with Derived Noisy Syntax Trees](https://aclanthology.org/2023.acl-srw.15/)**

* 提出confidence-based syntax encoding network (cSEN)来缓解噪声影响
* [文言文-现代文语料库](https://github.com/NiuTrans/Classical-Modern)



**[CWSeg: An Efficient and General Approach to Chinese Word Segmentation](https://aclanthology.org/2023.acl-industry.1/)**

* 队列训练和多功能解码策略来增强PLM



**[Rethinking Dictionaries and Glyphs for Chinese Language Pre-training](https://aclanthology.org/2023.findings-acl.70/)**

* 提出 CDBert，通过字典知识和汉字结构来增强中文大模型的语义理解
* 提出多义词判别任务：PolyMRC
* [GitHub](https://github.com/patrick-tssn/CDBert)

**[WYWEB: A NLP Evaluation Benchmark For Classical Chinese](https://aclanthology.org/2023.findings-acl.204/)**

* 古文测试任务

**[Exploiting Hierarchically Structured Categories in Fine-grained Chinese Named Entity Recognition](https://aclanthology.org/2023.findings-acl.211/)**

* 提出一个数据集FiNE
* 提出一个新方法FG-CNER

**[A Pilot Study on Dialogue-Level Dependency Parsing for Chinese](https://aclanthology.org/2023.findings-acl.607/)**

* 提出一个数据集

**[anko at SemEval-2023 Task 2: Bidirectional LSTM Model Based on Pre-training for Chinese Named Entity Recognition](https://aclanthology.org/2023.semeval-1.132/)**

* bert的output作为BiLST网络的input

**[YNUNLP at SemEval-2023 Task 2: The Pseudo Twin Tower Pre-training Model for Chinese Named Entity Recognition](https://aclanthology.org/2023.semeval-1.224/)**



**[Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script](https://aclanthology.org/2023.trustnlp-1.4/)**

* 提出藏文音节级黑盒（Tibetan syllable-level black-box）文本对抗攻击
* 针对少数民族语言的方法



## 2022 ACL Chinese

**[CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark](https://aclanthology.org/2022.acl-long.544.pdf)**

* 一个生物数据集



**[TopWORDS-Seg: Simultaneous Text Segmentation and Word Discovery for Open-Domain Chinese Texts via Bayesian Inference](https://aclanthology.org/2022.acl-long.13/)**

* 用贝叶斯推理在开放中文文本领域来进行词义分割和字词发现



**[RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining](https://aclanthology.org/2022.acl-long.65/)**

* 多模式对比学习预训练
* 多模式：语义、语音、视觉特征



**[Exploring and Adapting Chinese GPT to Pinyin Input Method](https://aclanthology.org/2022.acl-long.133/)**

* 拼音拆解，完整：wo；开始：w；结束：o
* 让模型利用上下文信息和拼音信息；增强模型对同声母字的辨析



**[Enhancing Chinese Pre-trained Language Model via Heterogeneous Linguistics Graph](https://aclanthology.org/2022.acl-long.140/)**

* 使用异构语言图谱



## 2021 ACL Chinese





## Text Classification

**[Improving Gradient Trade-offs between Tasks in Multi-task Text Classification](https://aclanthology.org/2023.acl-long.144/)**

**[Hierarchical Verbalizer for Few-Shot Hierarchical Text Classification](https://aclanthology.org/2023.acl-long.164/)**

**[Peer-Label Assisted Hierarchical Text Classification](https://aclanthology.org/2023.acl-long.207/)**

**[Randomized Smoothing with Masked Inference for Adversarially Robust Text Classifications](https://aclanthology.org/2023.acl-long.282/)**

**[HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification](https://aclanthology.org/2023.acl-long.432/)**

**[Efficient Shapley Values Estimation by Amortization for Text Classification](https://aclanthology.org/2023.acl-long.483/)**

**[TART: Improved Few-shot Text Classification Using Task-Adaptive Reference Transformation](https://aclanthology.org/2023.acl-long.617/)**

**[Hybrid Uncertainty Quantification for Selective Text Classification in Ambiguous Tasks](https://aclanthology.org/2023.acl-long.652/)**

**[PESCO: Prompt-enhanced Self Contrastive Learning for Zero-shot Text Classification](https://aclanthology.org/2023.acl-long.832/)**

**[Prototype-Guided Pseudo Labeling for Semi-Supervised Text Classification](https://aclanthology.org/2023.acl-long.904/)**

**[Linear Classifier: An Often-Forgotten Baseline for Text Classification](https://aclanthology.org/2023.acl-short.160/)**