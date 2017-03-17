# TensorFlow-LSTM 

## Description   
This is my practice to understand LSTM and usage of tensorflow. 

The repo use LSTM cell to read novel "The Romance of the Condor Heroes" (神鵰俠侶), which is written by famous Chinese novelist Jin-Yong. I choose character-level language as model and generate a short paragraph to see if it works well or not.  

## Dataset
Dataset is available [here](https://drive.google.com/open?id=0BxIKcHMvdD_UR2d3TG51MmM3NDg) and all of them are downloaded from [this website](http://98book.com/books/novelbook_99970.html). The whole file of "The Romance of the Condor Heroes" is about 2.8M.

## How to use  

### Training   

```
python char_rnn.py -m train -t 100 -l 3 --num_epochs 20 -i data/romance_condor_heroes.txt
```
### Generating

```
python char_rnn.py -m generate -i data/romance_condor_heroes.txt
```

### Default parameters (if not specify)
* mode = train  
* number of epoch = 20
* number of time steps = 100  
* number of layers = 3  
* learning rate = 0.001  
* batch size = 64  
* state size = 256  
* number of characters to generate = 500  
* pick number of top probability of chars = 5  

If you want to change some of these parameters, please reference to the arguments list   

```
python char_rnn.py -h 
```


## Experiment Result  
### Configuration  
* number of epoch = 100  
* number of time steps = 100  
* number of layers = 3  
* learning rate = 0.001  
* batch size = 64  
* state size = 256  
* number of characters to generate = 500  
* pick number of top probability of chars = 5  

### Generated Result  

*楊陽兒齊馬的了一笑，他們自何有得罪他的。”郭靖道：“你這話說得好了？”郭靖一怔，問道：“怎地你不是？”郭芙道：“我也是不是。”武修文道：“我不說，你在他不願……”他話未說完，忽聽得遠處臉上有人低聲道：“我師妹，你這話可是我的師父，他們便是你的師姊，這是甚麼好手？” 周伯通不知如何是她，他自然不會，卻也決不致想起郭靖和武氏兄弟的一番事，他這一生竟會對他不起自己。楊過道：“我也是你師妹呢？你不知道麼？”郭襄道：“你說這事還是你師父，你說是你不是？”郭襄一笑，心道：“你……你……”郭芙道：“我……我……”他話說得甚是甚是，自是不能不及。 這時楊過和郭襄一齊出言譏戰楊過，只得說道：“我不想娶你。”郭襄道：“我不是爹爹，他不是我的師叔，他是在這兒……”郭襄道：“你怎麼要我答應了？你這位郭姑娘不知道。你不是好人，你就跟他們說過這個。”郭靖一怔，道：“我說甚麼？我爹爹媽媽也不是我師父的。”郭襄道：“他們也不是好人，我也不會。你們兩個一齊出來，我就跟你說了，是了。我是我爹爹的話，你不知道，我不知道。你爹爹媽媽也是一樣一樣，但你就是郭家哥哥。”黃蓉道：“我不知道。” 郭靖一怔：“我也是這樣的？你說不對我的*


## Reference  
I reference a lot of documents. Here are some of them that is highly recommended if you are interested in RNN.

### Papers
* [Long Short-Term Memory][1]  
* [Learning to Forget: Continual Prediction with LSTM][2]  
* [An Empirical Exploration of Recurrent Network Architectures][3]  
* [LSTM: A Search Space Odyssey][4]  
* [On the Computational Power of Neral Nets][5]  
* [Neural Turing Machines][6]  

### Blogs
* [The Unreasonable Effectiveness of Recurrent Neural Networks (Andrej Karpathy blog)][7]  
* [Understanding LSTM Networks (colah's blog)][8]  
* [Recurrent Neural Networks in Tensorflow (R2RT blog)][9]  
* [Written Memories: Understanding, Deriving and Extending the LSTM (R2RT blog)][10]  
* [Recurrent Neural Networks Tutorial (WildML blog)][11]

[1]: http://isle.illinois.edu/sst/meetings/2015/hochreiter-lstm.pdf
[2]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.5709&rep=rep1&type=pdf 
[3]: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf 
[4]: https://arxiv.org/abs/1503.04069 
[5]: http://binds.cs.umass.edu/papers/1995_Siegelmann_JComSysSci.pdf 
[6]: https://arxiv.org/pdf/1410.5401.pdf

[7]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[8]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[9]: http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
[10]: http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html
[11]: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns
