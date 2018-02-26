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
   
   ![Housing price](../images/Diagram_lecture_2.png)
   
   * Hypothesis: h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> x + &theta;<sub>1</sub>x
   * Cost function: J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;&sum;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
        * Squared error function or mean squared error, the mean is halved as a convinience for the computation of gradient descent
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
