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


Credits
This project is taken from Georgia Tech's NLP class, Fall 2020.
