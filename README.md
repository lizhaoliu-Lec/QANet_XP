# QANet For XP

[QANet](https://arxiv.org/abs/1804.09541)

## QANet
- **残差块**: 使用了残差网络来加深网络深度
- **自注意**: Google采用了自家的[multihead attention](https://arxiv.org/abs/1706.03762) 来计算`self attention`
- **强化位置信息**： QANet强化了位置信息，在每个卷积块中都加入了时间序列信息，可查看`layers / residual_block / add_timing_signal_ld`

### 小小改进
加入原始位置信息(position embedding)在decoder层做Attention计算

<div align=center><image src='./images/equation.png' alt=''/></div>

## 模型

### requirements
```
tensorflow 1.6+
jieba
```

### 语料预处理
包括生成词典，使用预训练词向量，模型支持[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)中预训练词向量，下载该模型词向量后在`my_cli.py`中指定即可.

```bash
python my_cli.py --prepro
```

### 训练
```bash
python my_cli.py --train [arguments]
```

或者直接采用封装好的bash训练
```bash
bash train.sh
```

## Reference
- [DuReader](https://github.com/baidu/DuReader)
- [QANet Baseline](https://github.com/NLPLearn/QANet)
- [QANet_dureader](https://github.com/SeanLee97/QANet_dureader)
