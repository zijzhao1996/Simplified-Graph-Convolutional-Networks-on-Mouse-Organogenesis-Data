## Organogenesis_SGC

#### Authors
Zijie Zhao, Akshay Balsubramani, Prof. Anshul Kundaje

#### Abstract
Apply [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) to the [Shendure mouse data](https://www.ncbi.nlm.nih.gov/pubmed/30787437) - predict markers of organogenesis across timepoints to recapitulate known drivers of organ trajectories, and then look backward-infer causal genes from the models.

#### Result
```
zijzhao@kali:~$ python SGC.py
Loading the data...
Load data sucessfully.
--------------------------------------------
Stage1: 9.5, stage2: 10.5:
Training accuracy : 0.9728, test accuracy: 0.9583.
Running time: 76.68 seconds.
--------------------------------------------
Stage1: 9.5, stage2: 11.5:
Training accuracy : 0.9743, test accuracy: 0.9553.
Running time: 75.62 seconds.
--------------------------------------------
Stage1: 9.5, stage2: 12.5:
Training accuracy : 0.9735, test accuracy: 0.9589.
Running time: 76.27 seconds.
--------------------------------------------
Stage1: 9.5, stage2: 13.5:
Training accuracy : 0.9728, test accuracy: 0.9557.
Running time: 76.57 seconds.
--------------------------------------------
Stage1: 10.5, stage2: 11.5:
Training accuracy : 0.9199, test accuracy: 0.8689.
Running time: 114.97 seconds.
--------------------------------------------
Stage1: 10.5, stage2: 12.5:
Training accuracy : 0.9236, test accuracy: 0.8696.
Running time: 114.99 seconds.
--------------------------------------------
Stage1: 10.5, stage2: 13.5:
Training accuracy : 0.9232, test accuracy: 0.8693.
Running time: 110.22 seconds.
--------------------------------------------
Stage1: 11.5, stage2: 12.5:
Training accuracy : 0.9170, test accuracy: 0.8672.
Running time: 121.16 seconds.
--------------------------------------------
Stage1: 11.5, stage2: 13.5:
Training accuracy : 0.9188, test accuracy: 0.8682.
Running time: 125.53 seconds.
--------------------------------------------
Stage1: 12.5, stage2: 13.5:
Training accuracy : 0.9421, test accuracy: 0.9164.
Running time: 98.82 seconds.
--------------------------------------------
All work done.
```

In this example, we choose `cluster num=30` (Notochord and floor plate cells) and `k=2`. And the accuracy headmap is shown as below:  

<img src="/figure/heatmap.png"  width="500" height="450">

We could see that: 
* Simple graph convnet generally achieve a decent accuracy. (>87% with k=2)
* It is easier to classify E9.5 to all other time stamps. There might be some transcriptional change happen between E9.5 and E10.5.
* E12.5 and E13.5 could also be easily classified. Similarly, there might be some transcriptional variation.
* However, it is hard to sperate among E10.5, E11.5, E12.5 and E13.5.
