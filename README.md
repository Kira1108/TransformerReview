# Transformer

作为一个科学家，你应该使用下面的笔记本来训练这个Transformer(google colab TPU, 模型持久化在谷歌硬盘)：    
https://colab.research.google.com/drive/1jEfCu010BPWrepNl0nH2Hj10hQ8Ml3zH?usp=sharing    
训练好了把笔记本模型save到模型服务器上面去， 这个笔记本就是 notebooks/TransformerTraining.ipynb    

除此之外还要好好研究一下数据， 数据的笔记本在 notebooks/data.ipynb    

没有GPU就别玩了，爱咋咋地了， 学学就行。

## Transformer对我有啥用：

1. 以dot product 去做self attention的时候， 利用了vector similarty的思想，把similary matrix当作weight
做成了一个attention matrix, T * T）,similary matrix其实也不只在transformer才用到的，邻接矩阵也是一个N*N的矩阵，任意两个节点的结构相似性，也能表示成，相似性矩阵，然后用来做节点聚类，节点特征啥的， 卧槽， 这个可以做Node self attention, 妈的，我咋这么牛逼呢。

2. Transformer这个东西的结构确实比较难以想象， 没有中间的骚操作，整个transformer就是一堆屎。

3. 对于input来说，我可以随便折腾，所以在encoder layer的时候，从理解的角度来说，可以完全忽略padding mask，然后可以简化成一个简单结构 multihead attention + feed forward net, 这两个东西为了防止过拟合，都加入了dropout，然后添加了一个skip connection的layernorm， 这个结构可以在其他的地方复用。 

   [功能层] + [Dropout] + [Layernorm Skip connect]        
\+ [功能层] + [Dropout] + [Layernorm Skip connect]     
\+ [功能层] + [Dropout] + [Layernorm Skip connect]     
\+ [功能层] + [Dropout] + [Layernorm Skip connect]     
\+ Dense 变变形状输出


4. 看到和看不到：对于encoder layer来说，看padding是没有意义的，对于decoder来说， 看decoder input的padding是没有用的，看encoder 的padding位也是没有用的。 且在decoder预测某个step的时候，不能看到这个step后面的东西。 这些都被mask优雅的处理了，所以整个东西输入的时候，都是NTD的东西来来去去。

5. embedding + positional encoder + wq, wk, qv 融入位置信息后，经过dense的变换搞一下，这个目的是，word的底层表示embedding， 加入positional以后，有了词语 + 位置的表示， 但是query， key， value三者的意义，是通过wq,wk,wv三个dense层赋予的， 这样后续的self attention才有意义。 【positional encoding是可以在其他地方复用的】。


6. mask发生在attention matrix上面，我看不到你，你就是一堆屎， 这个很优雅，很合理。 但是create mask的时候，有些不同， 对于padding mask，我不需要知道这个矩阵的哪些位置是0，我需要原始的（N，T）的sequence， 看一看在T这个dimension上，哪些位置是0. 对于lookahead mask， 只需要知道，当前是第几步，后续的全被mask掉就好了，所以这个只需要知道一个decoder layer的max_token(T)，就可以创建出来了。 【mask用户处理stepwise的vectorized表示，是可以复用的】


7. 对于loss和accuracy， 要把padding位置的accuracy完全忽略，这时候，如果就傻呵呵的spasecategorical，就添加了很多无效信息进去。所以no reduction + mask + reduction over non_padding positions. 可以给出真实的accuracy和loss。 这个以前做的时候都没有好好去写这些东西，【以后要复用这个东西·】



