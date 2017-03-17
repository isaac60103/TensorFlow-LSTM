# Tensorflow-LSTM 

## Description   
This is my practice to understand LSTM and usage of tensorflow. 

The repo use LSTM cell to read novel "The Romance of the Condor Heroes" (神鵰俠侶), which is written by famous Chinese novelist Jin-Yong. I choose character-level language as model and generate a short paragraph to see if it works well or not.  

## Data 
Data is available [here](https://drive.google.com/open?id=0BxIKcHMvdD_UR2d3TG51MmM3NDg) and all of them are downloaded from [this website](http://98book.com/books/novelbook_99970.html)

## How to use  

### Training   

```
python char_rnn.py -m train -t 100 -l 3 --num_epochs 20 -i ./data/romance_condor_heroes.txt
```

#### Default parameters (if not specify):
* mode = train  
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

### Generating

```
python char_rnn.py -m predict -i data/total.txt
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



## Reference  
I reference a lot of documents. Here are some of them that is highly recommended if you are interested in RNN

[The Unreasonable Effectiveness of Recurrent Neural Networks (Andrej Karpathy blog)][1]  
[Understanding LSTM Networks (colah's blog)][2]  
[Recurrent Neural Networks in Tensorflow (R2RT blog)][3]  
[Written Memories: Understanding, Deriving and Extending the LSTM (R2RT blog)][4]  


[1]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[2]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[3]: http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
[4]: http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html
