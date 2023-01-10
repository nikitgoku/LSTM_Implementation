# Long Short-Term Memory Network (LSTMs) for Text Message Classification
This repository contains the implementation of LTSM network for Text Message Classification. Following will be the detailed article to understand Recurrent Neural Network (RNN) architecture from which LTSM architecture is derived, followed with the brief explanation of the implementation where I have used Keras. This code is built on <Kaggle>

## Contents
1. Introduction
2. RNN Architecture
3. LSTM Architecture
4. Implementation using Keras

## Introduction
Ordinary feedforward networks were only meant for data points that are independent of each other. However, if we have data in sequence such that one data point depends upon the previous data point, we need to modify the neural network to incorporate the dependencies between these data points. Recurrent Neural Network which has a different architecture than the traditional feed forward neural network, they were made and are very popular to handle data which are in sequential, efficiently used in Natural Language Processing (NLP) which is used for text classification tasks such as text classification, sentiment analysis, text generation and translation. A basic RNN architecture is built to be used for sequential or time-series data and contains networks with loops in them, allowing retaining past or historical information to forecast future values.

## RNN Architecture
An RNN can be viewed as many copies of a feed forward ANN executing in a chain, allowing information to persist. the chain-like architecture is what makes RNN suitable for sequential data as it allows the network to view the past information to understand the relation between data points.
![image](https://user-images.githubusercontent.com/114753615/211561803-e15ae916-bacf-4a56-a05d-73d6bdc5320a.png)
Looping allows the network to step through sequential input whilst persisting the state of the nodes in the hidden layer between steps - a sort of working memory.
The information persisting idea where to perform the current task is based on the previous information is intriguing, but what if to make relevance between the data points require information from further back, for example, while trying text prediction to predict the next word in "The clouds are in the ...", it is obvious that the next word probably will be _sky_ (if trained properly :p), as the prediction depends on the recent information, but what will be the prediction be here "I was born in France, I've lived there all my childhood and then moved out for my studies, I am fluent in ...". Obviously, the answer in _French_ (for a human) but for a machine trained on simply RNN cannot claim the answer to be _French_ as the relevant information is way behind with a long gap. This is where the requirement of storing long term memory comes in with an additional feature of disposing off irrelevant data points is required.

## LSTM Architecture
Long Short-Term Memory Networks are specially derived RNNs which can learn and memorise long term dependencies, recalling information for long periods is the default behaviour. It has a chain-like structure where four interacting layers communicate in a unique way. The properties of a LSTM network are:

    1. They forget irrelevant information of the previous state.
    
    2. They selectively update the cell state.
    
    3. They output certain part of the cell state.
    
![image](https://user-images.githubusercontent.com/114753615/211574549-85a208e7-2e7c-4232-89cc-0061d68c4e24.png)
 The repeating layers has a very different structure compared to the RNN which only has a _tanh_ layer.
 The four components include:
 
 1. *Cell State:*
    This is the most important part of LSTM, enacting as a long-term memory persisting information (if required) for all the iteration of the node. This is where removal of unnecessary information and keeping the important contextual information happens making sure that the information from the early iteration is not lost over ling sequence.   
2. *Forget Gate:*
    This is used to decide what information should be removed from the cell state. This gate outputs a number between 0 and 1 where 1 means completely keep the information and 0 means completely remove the information.
3. *Input Gate:*
    Decides which information should be added to the cell state. the _sigmoid_ function is used to call the input gate which mentions which values in the cell will be updated followed by a _tanh_ function which creates a vector for the new candidate values which are to be added to the state.
4. *Output Gate:*
    Decides on what working memory this node will output, based on the calculations on the current cell state and concatenated working memory. This also contains a _sigmoid_ layer which decided the part of the cell state to be updated followed by the _tanh_ layer to push the values between -1 and 1 which is multiplied by the output of the _sigmoid_ funtions output.
    
    
## Implementation using Keras
The model will be built on using Tensorflow and Keras integrated library whose purpose will be to detect spam messages using binary classification as 1 (SPAM) or 0 (HAM).

The notebook goes through the following:

    1. Load, explore and prepare data

    2. Build the LSTM model
    
    3. Train the model on training datasets
    
    4. Check the accuracy and loss
    
### Data
This a text data downloaded from UCI datasets. It contains 5574 SMS phone messages. The data were collected for the purpose of mobile phone spam research. The data have been labelled as either 1(SPAM) or 0(HAM).
WordCloud is being used here to visualise the most frequent words in the text list. This has been done for both HAM and SPAM messages.
The distribution of messages shows that there are more HAM messages than SPAM, through which it is required to downsample the data to have similar types of data.
As this data is being used in a deep learning model, text pre-processing is required so that numerical instances of the words are provided to the model. Text pre-processing in the code includes: 

- Tokenisation: encoding words into integers

- Sequencing: representing sentences by sequences of numbers

- Padding: To have same length of each text of every iteration

### Model: LSTM Layer Architecture
We build a sequential model which includes an Embedding Layer, LSTM layers and a Dense Layer.

The Embedding Layer maps each word into an N-dimensional vector of real number, where in the relation between words is defined by closeness of the word placement in the vector. Higher the dimension the embedding layer, higher is the linguistic relation between words.

The LSTM layer the fast version of LSTM which is optimised for GPUs. This layer looks at the sequence of words in the review, along with their word embeddings and uses both of these to determine to sentiment of a given review.

All LSTM units are connected to a single node in the dense layer. A sigmoid activation function determines the output from this node - a value between 0 and 1. Closer to 0 indicates a negative review. Closer to 1 indicates a positive review.
    
### References
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [A Gentle Introduction to RNN Unrolling](https://machinelearningmastery.com/rnn-unrolling/)
- [Explaining Recurrent Neural Networks](https://www.bouvet.no/bouvet-deler/explaining-recurrent-neural-networks)
