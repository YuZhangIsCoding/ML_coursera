# Machine Learning

Today I started a classic Coursera course [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome). I will update weekly about the topics introduced in the lectures and interesting problems I encountered.
## Week 1
1. Introduction to Machine Learning
   * Arthur Sammuel:
    > Field of study that gives computers the ability to learn without being explictly programmed.
   * Tom Mitchell:
    > A computer is said to learn from experience E with respect to some task T and some performance measure P,
    if its performance on T, as measured by P, improves with experience E.
   * Popular ML algorithms:
      * Supervised learning
      * Unsupervised learning
      * Others: reinforcement learning, recommender systems
   * Supervised learning:
      * Regression: predict continuous values, in other words, mapping input to some continuous function.
      * Classification: discrete values
   * Unsupervised learning:
      * No given answer, as as google news clustering and clustering people using genes
      * Could approach problems with little or no idea what the result should look like
      * Could not know the effects of variables
      * No feedback on the prediction results
2. Linear Regression with One Variable
   * Housing price
   
   ![Housing price](images/Diagram_lecture_2.png)
   
   * Hypothesis: h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> x + &theta;<sub>1</sub>x
   * Cost function J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = $$1\over 2m$$
  
