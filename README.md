# Machine Learning

Today I started a classic Coursera course [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome). I will update weekly about the topics introduced in the lectures and interesting problems I encountered. Here, I will just include a few bullet points from each week. Detailed summaries can be found in each week's README.md file. All diagrams were created from [draw.io](https://www.draw.io/), and all math notations were typed using inline html code or generated from [CodeCogs](http://latex.codecogs.com/).
## [Week 1](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week1/README.md)
1. Introduction to Machine Learning
   * Popular ML algorithms
   * Supervised and unsupervised learning
2. Linear Regression with One Variable
   * Housing price
   
   ![Housing price](images/Diagram_lecture_2.png)
   
   * Hypothesis: h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> x + &theta;<sub>1</sub>x
   * Cost function: J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
   * Gradient descent
   * Learning rate &alpha;     
3. Linear Algebra
   * Matrix, vector
   * Matrix addition: same dimension, element-wise
   * Scalar multiplication: multiply by real number
   * Matrix-vector, matrix-matrix multiplication
   * Transpose: B = A<sup>T</sup>, then B<sub>ij</sub> = A<sub>ji</sub>
      
## [Week 2](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week2/README.md)
1. Linear Regression with Multiple Variables
    * Often times we have multiple features (variables)
h<sub>&theta;</sub>(x) = &theta;<sub>0</sub>+&theta;<sub>1</sub>x<sub>1</sub>+&theta;<sub>2</sub>x<sub>2</sub>+...
    * The problem can be expressed as: h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>&sdot;x

1. Gradient Descent for Multivariate Linear Regression
    * &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>J(&theta;)
    * Feature Scaling: make sure that features are on a similar scale
    * Mean normalization: (x<sub>i</sub>-&mu;)/s
    * Learning rate: J(&theta;) should decrease after every iteration for sufficiently small &alpha;.
    * Features can be combined, e.g., combine frontage and depth to area.
    * Polynomial regression

1. Normal Equation: computing parameters analytically
    * &theta; = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y
    * Feature scaling is not needed
    * Noninvertibility (singular/degenerated)
        
1. Vectorization
    * Use matrix operations may reduce the computation times.
    * Wrap it all up, we can use this form for the matrix iterations: **&theta; = &theta;-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;(X&sdot;&theta;-y)**
    * The cost function can also be vectorized: **J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;(X&sdot;&theta;-y)<sup>T</sup>&sdot;(X&sdot;&theta;-y)**

## [Week 3](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week3/README.md)

1. Classification and Representation
    * Logistic regression: idea is to use threshold classifier output h<sub>&theta;</sub>(x) at 0.5
    * Hypothesis representation **h<sub>&theta;</sub>(x) = <sup>1</sup>&frasl;<sub>(1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup>)</sub>**, and **h<sub>&theta;</sub>(x) = P(y=1|x;&theta;)**
    * Decision boundary - a property of hypothesis
    * Simplified cost function and gradient descent **J(&theta;) = -<sup>1</sup>&frasl;m[&sum;(ylog(h<sub>&theta;</sub>(x))+(1-y)log(1-h<sub>&theta;</sub>(x)))]**
    * Iterations: &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>(J(&theta;))
        
    * Vectorized implementation:
        
        **h<sub>&theta;</sub>(x) = g(X&sdot;&theta;)**
        
        **J(&theta;) = <sup>1</sup>&frasl;<sub>m</sub>&sdot;(-y<sup>T</sup>&sdot;log(h)-(1-y)<sup>T</sup>log(1-h))**
        
        **&theta; = &theta;-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;(g(X&sdot;&theta;)-y)**

    * Advanced optimization (Conjugate gradient, BFGS, L-BFGS)
    
1. Multicalss classification
    * one-vs-all
    
        ![one-vs-all](images/BinaryClassVsMultiClass.png)
        
        **h<sub>&theta;</sub><sup>(i)</sup>(x) = P(y=i|x;&theta;), (i = 1, 2, 3, ...)**, 
        and **max h<sub>&theta;</sub><sup>(i)<sup>(x)**

1. Rugularization
   * The problem of overfitting
      
      ![overfitting](images/Overfitting.png)
      
   * Addressing overfitting (Reduce number of features or Regularization)

   * Cost function:**J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;[&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>+&lambda;&sdot;&sum;&theta;<sub>j</sub><sup>2</sup>], *j = 1, 2, ..., n***
   * Regulalized linear regression
   
      **&theta;<sub>0</sub> = &theta;<sub>0</sub>-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x)-y)&sdot;x<sub>0</sub>**
      
      **&theta;<sub>j</sub> = &theta;<sub>j</sub>-&alpha;&sdot;[<sup>1</sup>&frasl;<sub>m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x)-y)&sdot;x<sub>0</sub>+<sup>&lambda;</sup>&frasl;<sub>m</sub>&sdot;&theta;<sub>j</sub>], *j = 1, 2, ..., n***
      
      }
   * Normal equation:
      **&theta; = (X<sup>T</sup>&sdot;X+&lambda;[<sup>0</sup>1<sub>1</sub>])<sup>-1</sup>&sdot;X<sup>T</sup>&sdot;y**
            
   * Regularized logistic regression:
      **J(&theta;) = -<sup>1</sup>&frasl;<sub>m</sub>&sdot;[&sum;(ylog(h<sub>&theta;</sub>(x))+(1-y)log(1-h<sub>&theta;</sub>(x)))]+<sup>&lambda;</sup>&frasl;<sub>2m</sub>&sdot;&sum;&theta;<sub>j</sub><sup>2</sup>, *j = 1, 2, ..., n***
## [Week 4](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week4/README.md)
1. Non-linear classification
    * Too many features in hypothesis, logistic regression suffers overfitting or large computational cost.
    * Computer vision: Car detection may result in millions of features if just include the quadratic features.

2. Neural networks: algorithms that try to mimic the brain
    * The "One learning algorithm" hypothesis: (Auditory cortex learns to see; Somatosensory cortex learns to see, etc)
    * Model representation
        * A [neuron](https://en.wikipedia.org/wiki/Neuron) is composed of nucleus, dendrites, axons, etc., where dendrites and axons serves as the "input wires" and "output wires", respectively.
        
            ![neuron_from_wikipedia](images/Neuron.png)
        * Logistic unit can be built similarily on this neuron model:
           
            ![NeuronModel](images/NeuronModel.jpg) 
        * Neural networks:

            ![NeuralNetworks](images/NeuralNetworks.jpg)
        * Forward propagation: just like logistic regressions, but do it on every layer
            
            z<sup>(2)</sup> = &Theta;<sup>(1)</sup>x

            a<sup>(2)</sup> = g(z<sup>(2)</sup>)

            Add a<sub>0</sub><sup>(2)</sup> = 1 &rarr; a<sup>(2)</sup> at this layer has n+1 features

            z<sup>(3)</sup> = &Theta;<sup>(2)</sup>a<sup>(2)</sup>

            ...
    * Examples: logic gates (AND, OR, NOT, (NOT A) AND (NOT B), XNOR, XOR)
    * Multiclass classification
        * Multiple output units: one-vs-all
        * Instead of output as discrete values such as y &isin; {1, 2, 3, ...}, we have y &isin; [[1;0;0;0],[0;1;0;0],[0;0;1;0], ...]

## [Week 5](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week5/README.md)
1. Backpropagation

   * Cost function:
  
      <img src="https://latex.codecogs.com/svg.latex?J(\Theta)=-\frac{1}{m}\Bigg[\sum_{t=1}^{m}\sum_{k=1}^{K}y_k^{(t)}\textrm{log}h_{\Theta}(x)_k^{(t)}+(1-y_k^{(t)})\textrm{log}(1-h_{\Theta}(x)_k^{(t)})\Bigg]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{S_l}\sum_{j=1}^{S_{l+1}}(\Theta_{ji}^{l})^2"/>

   * Gradients
      
      <img src="https://latex.codecogs.com/svg.latex?\frac{\partial}{\partial\Theta_{ij}^{l}}J(\Theta)=\frac{1}{m}\sum_{t=1}^{m}(a_j^{l})^{(t)}\cdot(\delta_i^{l+1})^{(t)}+\frac{\lambda}{m}\Theta_{ij}^{l}"/>
      
      For the output unit:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^{L}=a_j^{L}-y_j"/>
      
      For inner layers:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^{l}=(\Theta^{l})^T\cdot\delta^{l+1}.*g'(z^l)=(\Theta^{l})^T\cdot\delta^{l+1}.*a^{l}.*(1-a^{l})"/>

   * Backpropagation intuition

      ![BackPropIntuition](images/BackPropIntuition.jpg)

   * Unrolling Parameters: Unroll the &Theta; matrices into the just one matrix and recover after backpropogation.

   * Gradient checking

       <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}J(\Theta)}{\Theta_{ij}^{l}}\approx\frac{J(...,\Theta_{ij}^{l}+\epsilon,...)-J(...,\Theta_{ij}^{l}-\epsilon,...)}{2\epsilon}"/>
    
   * Initial value of &Theta; Use random initialization for symmetry breaking.

   * Put it together
      * Pick a network architecture
      * Reasonable defaults: 1 hiddern layer or if >1 hidden layers, have same number of hidden units in every layer
      * Training a neural network
         1. Randomly initialize weights
         2. Implement forward propagation to get H<sub>&Theta;</sub>(x<sup>l</sup>) for each layer
         3. Compute cost function J(&Theta;)
         4. Implement backpropagation to compute partial derivatives
         5. Use gradient checking to compare partial derivaties compute using backpropagation vs. using numerical estimate of gradients
         6. Use gradient descent or advanced optimized method with backpropation to try to minimize J(&Theta;) as a function of parameters &Theta; (Notice that J(&Theta;) is non-convex, so we may end up finding a local minimum)
## Week 6
1. Spam Classifier

    Supervised learning
    
    x = features of email, e.g. deal, buy, discount, now, etc. In practice, take most frequently occuring n words (10,000 t0 50,000) in training set.

    y = spam (1) or not spam (0)

    * Advices to make your classifier have low error:
        * Collect lots of data, e.g. ["honey pot" project](https://en.wikipedia.org/wiki/Project_Honey_Pot).
        * Develop sophiscated features based on email routing infomation (email header).
        * Develop sophiscated features based on message bodies, e.g. distinguish synonyms, features about punctuations, etc.
        * Develop sophiscated algorithm to detect misspelling and improve input correctness.
    * Recommended approach
        * Start with a simple algorithm and test it on cross-validation data
        * Plot learning curve to decide if more data or more features needed, etc.
        * Error analysis: manually examine the examples that your algorithm made errors on. See if any systemetic trend of errors made.
    * Importance of numerical evaluation
        
        Should discount/discounts/discounted/discounting be taken as the same word? (Can be implemented by stemming softer, e.g. [porter stemmer](https://tartarus.org/martin/PorterStemmer/).

        Error analysis may not be helpful for deciding if this is likely to improve performance. Only solution is to try it and see if it works.

    * Error metrics for skewed classes
        
        Take the cancer classification as an example. Assume we find 1% error on test set, but the fact is that only 0.5% of patients have cancer. And if we have second classifier that always predicts the patients to be benign (y = 0), we will only have 0.5% error. But can we say that the second classifier is better?

        There's another example. Say an algorithm has 99.2% accuracy (0.8% error), and the other one has 99.5% accuracy (0.5% error), can we say the second is better?
    
    * Precision/Recall
    
        ![Precision/Recall](../images/Precision_Recall.jpg)
        
        * Precision: of all patients where we predicted y = 1, what fraction actually has cancer?

        * Recall: of all patients who has cancer, what fraction did we correctly detect as having cancer? In the example above, if we always predict y = 0, then recall = 0.

    * Trading off precision and recall
        
        Take logistic regression for example, 0 &le; h<sub>&theta;</sub>(x) &le; 1

        Predict 1 if h<sub>&theta;</sub>(x) &ge; threshold
        
        Predict 0 if h<sub>&theta;</sub>(x) < threshold

        * Suppose we want to predict y = 1 only if very confidently, then we may raise the threshold (0.7, 0.9, etc.). Then we will have high precision, but low recall (missing cases actually are 1)

        * Suppos we want to avoid missing too many cases of y = 1 (avoid false nagatives), we could lower threshold. Then we will have high recall, but low precision (many cases not 1 will be included)

        ![TradeOffPrecisionRecall](../images/TradeOffPrecsionRecall.png)

    * F<sub>1</sub> Score (F score)

        <img src="http://latex.codecogs.com/gif.latex?F_1Sore=2\frac{P\cdot{R}}{P+R}"/>
        
        How to compare precision/recall numbers?

        Algorithms|Precision|Recall|Average|F<sub>1</sub> Score
        -|-|-|-
        Algo1|0.5|0.4|0.45|0.444
        Algo2|0.7|0.1|0.4|0.175
        Algo3|0.02|1.0|0.51|0.0392

        * Average is not good because it fails to exclude the cases when precison or recall is extremely low.

    * Data for Machine Learning
        
        When designing a high accuracy learning system, sometime there are a lot of algorithms we can choose from.

        E.g. classify between confusing words

        {to, too, two}, {then, than}, etc. For breakfast, I ate ____ eggs.

        For this problem we could have following choices:
            * [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
            * [Winnow](https://en.wikipedia.org/wiki/Winnow_(algorithm))
            * [Memory-based](https://en.wikipedia.org/wiki/Instance-based_learning)
            * [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

    * Large Date Rationale
        
        Assume feature X &isin; R<sup>n+1</sup> has sufficient info to predict y accuracy
            
            Example: I ate (two) eggs.

            Counter example: Predicting housing price form only size and no other features

            Useful test: Given the input x, can human expert confidently predict y?
        
        Combine the strategies below will yield a small cost function J<sub>test</sub>(&theta;)
            1. Use a learning algorithm with many parameter (logisti/linear regression with many feature; neural network with many hidden layers), which gives low bias &rarr; J<sub>train</sub>(&theta;) will be small
            2. Use large training set, which makes it unlikely to overfit &rarr; J<sub>train</sub>(&theta;) &asymp; J<sub>test</sub>(&theta;)
