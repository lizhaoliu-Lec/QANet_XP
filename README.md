# QANet For XP

[QANet Paper](https://arxiv.org/abs/1804.09541)

## QANet
- **Residual block**: For gradient flow.
- **Self attention**: Google's [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Enhance position embedding**: Inject time sequence information in every convolutional block.

### Small change
Add position embedding in decoder by doing Attention.
![image](https://github.com/lizhaoliu-Lec/QANet_dureader/blob/master/images/equation.png)

### requirements
```
tensorflow >= 1.12.0
jieba >= 0.39
numpy >= 1.16.4
```

### Corpus preprocessing
Including dictionary generation, pretrained word vectors [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors), specify it in cli.py.

```bash
python cli.py --prepro
```

### Train
```bash
python cli.py --train [arguments]
```

or
```bash
bash train.sh
```

### Evaluate
```bash
python cli.py --evaluate [arguments]
```

or
```bash
bash evalutate.sh
```

### Predict
```bash
python cli.py --predict [arguments]
```

or
```bash
bash predict.sh
```

## Reference
- [DuReader](https://github.com/baidu/DuReader)
- [QANet Baseline](https://github.com/NLPLearn/QANet)
- [QANet_dureader](https://github.com/SeanLee97/QANet_dureader)
