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
   * Cost function: J(&theta;<sub>0</sub>, &theta;<sub>1</sub>)
   * Gradient descent
      * Start with some &theta;<sub>0</sub>, &theta;<sub>1</sub>
      * Keep changing (simutaneously) &theta;<sub>0</sub>, &theta;<sub>1</sub> to reduce J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) until hopefully end up at minimum
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
    * For convinience, define x<sub>0</sub> = 1
    * The problem can be expressed as: h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>&sdot;x

1. Gradient Descent for Multivariate Linear Regression
    * Algorithm:
        Repeat{
        
        &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>(J(&theta;))
        
        }
        
        Simutaneously update for j = 0, 1, ..., n
    * Feature Scaling: make sure that features are on a similar scale
        * Andrew's personal judgement: -3 ~ 3 and -&#8531; ~ &#8531; are appropriate
    * Mean normalization:
        * x<sub>i</sub> &larr; (x<sub>i</sub>-&mu;)/s, where s stands for range or standard deviation
    * Learning rate
        * Use plots to "debug", making sure that gradient denscent is working properly.
        * J(&theta;) should decrease after every iteration for sufficiently small &alpha;.
        * A large learning rate &alpha; may cause cost function to go up or experience oscillations over iterations.
        * Andrew's typical choice: 0.001, 0.003, 0.01, 0.03, ..., 0.3, 1, ...
    * Features can be combined, e.g., combine frontage and depth to area.
    * Polynomial regression:
        * h<sub>&theta;</sub>(x) = &theta;<sub>0</sub>+&theta;<sub>1</sub>x+&theta;<sub>2</sub>x<sup>2</sup>+&theta;<sub>3</sub>x<sup>3</sup> can be represented as h<sub>&theta;</sub>(x) = &theta;<sub>0</sub>+&theta;<sub>1</sub>x<sub>1</sub>+&theta;<sub>2</sub>x<sub>2</sub>+&theta;<sub>3</sub>x<sub>3</sub>
        * Feature scaling now is important
        * The choice of features is also very important

1. Normal Equation: computing parameters analytically
    * &theta; = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y
    * Feature scaling is not needed
    
      |Gradient Descent| Normal Equation|
      |--|--|
      |need to choose &alpha;|No need for &alpha;|
      |many iteractions|no need to iterate|
      |works well even n is large|slow if n is very large|
      |O(kn<sup>2</sup>)|O(kn<sup>3</sup>) for matrix operations|

    * Noninvertibility (singular/degenerated)
        * Reduandant features (linear dependent)
        * Too many features (e.g. m &le; n)
        
1. Vectorization
    * Use matrix operations may reduce the computation times.
    * Previously, we have h<sub>&theta;</sub>(x) = &sum;&theta;<sub>j</sub>&sdot;X<sub>j</sub> &rArr; &theta;<sup>T</sup>&sdot;X
    * Now for gradient descent, we have: &theta;<sub>j</sub> = &theta;<sub>j</sub>-&alpha;&sdot;<sup>1</sup>&frasl;<sub>m</sub>&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)&sdot;x<sub>j</sub><sup>(i)</sup>, and we want to apply the same idea. Vectorize it!
        * Let &theta; = &theta;-&alpha;&sdot;&delta;
        * &delta; = <sup>1</sup>&frasl;<sub>m</sub>&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)&sdot;x<sub>j</sub><sup>(i)</sup>
        * Use the matrix mentioned in normal equation, X =
        
            1|x<sub>1</sub><sup>1</sup>|...|x<sub>n</sub><sup>1</sup>
            --|--|--|--
            1|...|...|...|
            1|x<sub>1</sub><sup>m</sup>|...|x<sub>n</sub><sup>m</sup>
        
        * So h<sub>&theta;</sub>(x) = X&sdot;&theta;
        * And the summation can be also express by the dot product of two matrices: X<sup>T</sup>&sdot;(X&sdot;&theta;-y)
        * Then &delta; = <sup>1</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;(X&sdot;&theta;-y)
        * When we reached the globle minimum, &delta;&rarr;0, so X<sup>T</sup>&sdot;(X&sdot;&theta;-y) = 0 &rArr; X<sup>T</sup>&sdot;X&sdot; = X<sup>T</sup>&sdot;y &rArr; &theta; = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y, which proves the analytical solution mentioned about
    * Wrap it all up, we can use this form for the matrix iterations: **&theta; = &theta;-<sup>&alpha;</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;(X&sdot;&theta;-y)**

