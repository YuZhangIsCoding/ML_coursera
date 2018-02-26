# Machine Learning

Today I started a classic Coursera course [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome). I will update weekly about the topics introduced in the lectures and interesting problems I encountered. Here, I will just include a few bullet points from each week. Detailed summaries can be found in each week's README.md file.
## Week 1
1. Introduction to Machine Learning
   * Popular ML algorithms:
      * Supervised learning
      * Unsupervised learning
      * Others: reinforcement learning, recommender systems
   * Supervised learning
      * Regression: predict continuous values, in other words, mapping input to some continuous function.
      * Classification: discrete values
2. Linear Regression with One Variable
   * Housing price
   
   ![Housing price](images/Diagram_lecture_2.png)
   
   * Hypothesis: h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> x + &theta;<sub>1</sub>x
   * Cost function: J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
   * Gradient descent
      * Learning rate &alpha;
          1. Too small &rarr; slow
          2. Too large &rarr; fail to converge or diverge
          3. As approaching a local minimum, gradient denscent will automatically take smaller steps &rarr; no need to reduce &alpha; over time
      * May reach a local minimum
      
3. Linear Algebra
   * Matrix, vector
   * Matrix addition: same dimension, element-wise
   * Scalar multiplication: multiply by real number
   * Matrix-vector, matrix-matrix multiplication
      * Often times solving Prediction = DataMatrix &sdot; Prameters is more computationally efficient
      * Not commutative: A&sdot;B &ne; B&sdot;A
      * Associative (A&sdot;B)&sdot;C = A&sdot;(B&sdot;C)
      * Identity matrix: *I*&sdot;A = A&sdot;*I*
      * Inverse: A&sdot;A<sup>-1</sup> = A<sup>-1</sup>&sdot;A = *I*
   * Transpose: B = A<sup>T</sup>, then B<sub>ij</sub> = A<sub>ji</sub>
      
## Week 2
1. Linear Regression with Multiple Variables
    * Often times we have multiple features (variables)
h<sub>&theta;</sub>(x) = &theta;<sub>0</sub>+&theta;<sub>1</sub>x<sub>1</sub>+&theta;<sub>2</sub>x<sub>2</sub>+...
    * The problem can be expressed as: h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>&sdot;x

1. Gradient Descent for Multivariate Linear Regression
    * Algorithm:
        Repeat{
        
        &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>(J(&theta;))
        
        }Simutaneously update for j = 0, 1, ..., n
    * Feature Scaling: make sure that features are on a similar scale
    * Mean normalization:
        * x<sub>i</sub> &larr; (x<sub>i</sub>-&mu;)/s, where s stands for range or standard deviation
    * Learning rate
        * J(&theta;) should decrease after every iteration for sufficiently small &alpha;.
    * Features can be combined, e.g., combine frontage and depth to area.
    * Polynomial regression:

1. Normal Equation: computing parameters analytically
    * &theta; = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y
    * Feature scaling is not needed
    * Noninvertibility (singular/degenerated)
        * Reduandant features (linear dependent)
        * Too many features (e.g. m &le; n)
        
1. Vectorization
    * Use matrix operations may reduce the computation times.
    * Wrap it all up, we can use this form for the matrix iterations: **&theta; = &theta;-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;(X&sdot;&theta;-y)**
    * The cost function can also be vectorized: **J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;(X&sdot;&theta;-y)<sup>T</sup>&sdot;(X&sdot;&theta;-y)**

## Week 3

1. Classification and Representation
    * Idea is to use threshold classifier output h<sub>&theta;</sub>(x) at 0.5:
        * If h<sub>&theta;</sub>(x) &ge; 0.5, predict y = 1
        * If h<sub>&theta;</sub>(x) &ge; 0.5, predict y = 0
    * Logistic regression
        * Hypothesis representation (we want 0 &le; h<sub>&theta;</sub>(x) &le; 1)

            &rArr; **h<sub>&theta;</sub>(x) = <sup>1</sup>&frasl;<sub>(1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup>)</sub>**
        * Interpretation of hypothesis

            **h<sub>&theta;</sub>(x) = P(y=1|x;&theta;)**

        * Decision boundary - a property of hypothesis
            
            h<sub>&theta;</sub>(x) &ge; 0.5 whenever &theta;<sup>T</sup>&sdot;x &ge; 0

    * Logistic Regressioin Model
        * Logistic regression cost function:
        
            **Cost(h<sub>&theta;</sub>(x), y) = -log(h<sub>&theta;</sub>(x)), *if y = 1***
            
            **Cost(h<sub>&theta;</sub>(x), y) = -log(1-h<sub>&theta;</sub>(x)), *if y = 0***
    * Simplified cost function and gradient descent
        * **Cost(h<sub>&theta;</sub>(x), y) = -ylog(h<sub>&theta;</sub>(x))-(1-y)log(1-h<sub>&theta;</sub>(x))**
        * **J(&theta;) = -<sup>1</sup>&frasl;m[&sum;(ylog(h<sub>&theta;</sub>(x))+(1-y)log(1-h<sub>&theta;</sub>(x)))]**
    * Algorithm:
    
        Repeat{
        
        &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>(J(&theta;))
        
        } Simultaneously update all &theta;<sub>j</sub>
        
    * Vectorized implementation:
        
        **h<sub>&theta;</sub>(x) = g(X&sdot;&theta;)**
        
        **J(&theta;) = <sup>1</sup>&frasl;<sub>m</sub>&sdot;(-y<sup>T</sup>&sdot;log(h)-(1-y)<sup>T</sup>log(1-h))**
        
        **&theta; = &theta;-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;(g(X&sdot;&theta;)-y)**

    * Advanced optimization
        * Different algorithms:
            
            1. Graident descent
            1. [Conjugate gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
            1. [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
            1. [L-BFGS](https://en.bywiki.com/wiki/L-BFGS)
1. Multicalss classification
    * one-vs-all
    
        ![one-vs-all](../images/BinaryClassVsMultiClass.png)
        
        **h<sub>&theta;</sub><sup>(i)</sup>(x) = P(y=i|x;&theta;), (i = 1, 2, 3, ...)**
        * Train a logistic regression classifier h<sub>&theta;</sub><sup>(i)</sup>(x) for each class i to predict the probability that y = i
            
            max h<sub>&theta;</sub><sup>(i)<sup>(x)

1. Rugularization
   * The problem of overfitting
      
      ![overfitting](../images/Overfitting.png)
      * If we have too many features, the learned hypothesis may fit the training set well, but fails to generalize to new examples.
   * Addressing overfitting
      1. Reduce number of features
         * Manually select which features to keep
         * Model selection algorithm
      2. Regularization
         * Keep all the features, but reduce magnitude/values of &theta;<sub>j</sub>
         * Works well when we have a lot of features, and each contributes a bit to predicting y
   * Cost function
      
      **J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;[&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>+&lambda;&sdot;&sum;&theta;<sub>j</sub><sup>2</sup>], *j = 1, 2, ..., n***
      
      * If &lambda; is set to extremely large value, will cause "underfitting"
      * Else if &lambda; is too small, may "overfitting"
   * Regulalized linear regression
   
      Repeat{
      
      **&theta;<sub>0</sub> = &theta;<sub>0</sub>-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x)-y)&sdot;x<sub>0</sub>**
      
      **&theta;<sub>j</sub> = &theta;<sub>j</sub>-&alpha;&sdot;[<sup>1</sup>&frasl;<sub>m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x)-y)&sdot;x<sub>0</sub>+<sup>&lambda;</sup>&frasl;<sub>m</sub>&sdot;&theta;<sub>j</sub>], *j = 1, 2, ..., n***
      
      }
   * Normal equation
      
      **&theta; = (X<sup>T</sup>&sdot;X+&lambda;[<sup>0</sup>1<sub>1</sub>])<sup>-1</sup>&sdot;X<sup>T</sup>&sdot;y**
      
      if &lambda; > 0, (X<sup>T</sup>&sdot;X+&lambda;[<sup>0</sup>1<sub>1</sub>]) will be invertible
      
   * Regularized logistic regression
      
      Same as before, but h<sub>&theta;</sub>(x) = <sup>1</sup>&frasl;<sub>(1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup>)</sub>
      
      **J(&theta;) = -<sup>1</sup>&frasl;<sub>m</sub>&sdot;[&sum;(ylog(h<sub>&theta;</sub>(x))+(1-y)log(1-h<sub>&theta;</sub>(x)))]+<sup>&lambda;</sup>&frasl;<sub>2m</sub>&sdot;&sum;&theta;<sub>j</sub><sup>2</sup>, *j = 1, 2, ..., n***
