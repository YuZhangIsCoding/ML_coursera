## Week 2
1. Linear Regression with Multiple Variables
    * Often times we have multiple features (variables)
h<sub>&theta;</sub>(x) = &theta;<sub>0</sub>+&theta;<sub>1</sub>x<sub>1</sub>+&theta;<sub>2</sub>x<sub>2</sub>+...
    * For convinience, define x<sub>0</sub> = 1
    * The problem can be expressed as: h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>&sdot;x

1. Gradient Descent for Multivariate Linear Regression
    * Algorithm:
        Repeat{
        
        &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>J(&theta;)
        
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
    * Previously, we have h<sub>&theta;</sub>(x) = &sum;&theta;<sub>j</sub>&sdot;X<sub>j</sub> &rArr; &theta;<sup>T</sup>&sdot;x
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
    * The cost function can also be vectorized: **J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub>&sdot;(X&sdot;&theta;-y)<sup>T</sup>&sdot;(X&sdot;&theta;-y)**

