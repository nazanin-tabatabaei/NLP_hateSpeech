# Text Classification for Hate Speech
Hate speech is a  
(a) deliberate attack,  
(b) directed towards a specific group of people,  
(c) motivated by aspects of the group’s identity.  
The three premises must be true for a sentence to be categorized as HATE. Here are two examples:  
(a) “Poor white kids being forced to treat apes and parasites as their equals.”  
(b) “Islam is a false religion however unlike some other false religions it is crude and appeals to crude people such as arabs.”  
In (a), the speaker uses “apes” and “parasites” to refer to children of dark skin and implies they are not equal to “white kids”. That is, it is an attack to the group composed of children of dark skin based on an identifying characteristic, namely, their skin colour. Thus, all the premises are true and (a) is a valid example of HATE. Example (b) brands all people of Arab origin as crude. That is, it attacks the group composed of Arab people based on their origin. Thus, all the premises are true and (b) is a valid example of HATE.

The goal is to build a Naive Bayes model and a logistic regression model on a real-world hate speech classification dataset.The dataset used here is collected from Twitter online. Each example is labeled as 1 (hatespeech) or 0 (Non-hatespeech).  
![image](01.JPG)  

## Train and Test datasets
We start by dividing the data into train and test datasets.
```python
from sklearn.model_selection import train_test_split
train_frame, test_frame = train_test_split(train_frame, test_size=0.2)
```
## Text Processing
### Text Tokenization
First, we tokenize the text into tokens:
```python
tokenized_text = []
for i in range(0, len(train_frame['text'])):
   tokenized_text.append(tokenize(train_frame['text'][i]))
```
### Convert text into features
Then we convert the text into features. The example used here uses Unigram Features to convert the tokenized text into features. The first part creates the vocabulary.
```python
feat_extractor = UnigramFeature()
feat_extractor.fit(tokenized_text)
```
The next part shows how many of each word we  have in each document
```python
X_train = feat_extractor.transform_list(tokenized_text)
Y_train = train_frame['label']
```
We tokenize and conver the test data into features, as well.
```python
tokenized_text = []
for i in range(0, len(test_frame['text'])):
    tokenized_text.append(tokenize(test_frame['text'][i]))
X_test = feat_extractor.transform_list(tokenized_text)
Y_test = test_frame['label']
```

#### Credits
This project is taken from Georgia Tech's NLP class, Fall 2020.
