## Week 3

1. Classification and Representation

    * Examples: 
        * Emails: spam/not spam
        * Online transactions: fradulent or not
        * Medical judgements: malignant or not
        
    * Idea is to use threshold classifier output h<sub>&theta;</sub>(x) at 0.5:
        * If h<sub>&theta;</sub>(x) &ge; 0.5, predict y = 1
        * If h<sub>&theta;</sub>(x) &ge; 0.5, predict y = 0
    * Logistic regression
        * Hypothesis representation (we want 0 &ge; h<sub>&theta;</sub>(x) &gt; 1)

            h<sub>&theta;</sub>(x) = g(&theta;<sup>T</sup>&sdot;x)

            g(z) = <sup>1</sup>&frasl;<sub>(1+e<sup>-z</sup>)</sub> (Sigmoid Function or Logistic Function)

            &rArr; **h<sub>&theta;</sub>(x) = <sup>1</sup>&frasl;<sub>(1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup>)</sub>**
        * Interpretation of hypothesis

            h<sub>&theta;</sub>(x) = estimated probability that y = 1 on input x, parameterized by &theta;

            or **h<sub>&theta;</sub>(x) = P(y=1|x;&theta;)**

            and P(y=1|x;&theta;)+P(y=0|x;&theta;) = 1

        * Decision boundary - a property of hypothesis
            Since g(z) &ge; 0.5 when z &ge; 0
            
            h<sub>&theta;</sub>(x) &ge; 0.5 whenever &theta;<sup>T</sup>&sdot;x &ge; 0

            The decision boundary can also be non-linear
    * Logistic Regressioin Model
        * Cost function
        
            Linear regression has J(&theta;) = <sup>1</sup>&frasl;<sub>m</sub>&sdot;&sum;<sup>1</sup>&frasl;<sub>2</sub>&sdot;(h<sub>&theta;</sub>(x)-y)<sup>2</sup>,
            where Cost(h<sub>&theta;</sub>(x), y) &rarr; <sup>1</sup>&frasl;<sub>2</sub>&sdot;(h<sub>&theta;</sub>(x)-y)<sup>2</sup>
            
            We may want to use similar function, say Cost(h<sub>&theta;</sub>(x), y) = <sup>1</sup>&frasl;<sub>2</sub>&sdot;(h<sub>&theta;</sub>(x)-y)<sup>2</sup>, and h<sub>&theta;</sub>(x) = <sup>1</sup>&frasl;<sub>(1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup>)</sub>
            
            But if use this function, J(&theta;) will be "non-convex"
            
            Instead we will construct a logistic regression cost function
        * Logistic regression cost function:
        
            **Cost(h<sub>&theta;</sub>(x), y) = -log(h<sub>&theta;</sub>(x)), *if y = 1***
            
            **Cost(h<sub>&theta;</sub>(x), y) = -log(1-h<sub>&theta;</sub>(x)), *if y = 0***
        * Case 1: y = 1
        
            Intuition: 
            
            cost = 0 if y = 1 and h<sub>&theta;</sub>(x) = 1

            but as h<sub>&theta;</sub>(x) &rarr; 0, cost &rarr; &infin;
            
            Can be understood as, if h<sub>&theta;</sub>(x) = 0, hypothesis prdicts P(y=1|x;&theta;) = 0, but y is actually 1. Then we will penalize learning algorithm by a large cost
        * Case 2: y = 0
        
            Same idea, we will penalize the case h<sub>&theta;</sub>(x) &rarr; 1 when y = 0
    * Simplified cost function and gradient descent
        * **Cost(h<sub>&theta;</sub>(x), y) = -ylog(h<sub>&theta;</sub>(x))-(1-y)log(1-h<sub>&theta;</sub>(x))**
        * **J(&theta;) = -<sup>1</sup>&frasl;m[&sum;(ylog(h<sub>&theta;</sub>(x))+(1-y)log(1-h<sub>&theta;</sub>(x)))]**
        * Can be derived from the principle of [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
    * Algorithm:
    
        Repeat{
        
        &theta;<sub>j</sub> := &theta;<sub>j</sub>-&alpha;<sup>&part;</sup>&frasl;<sub>&part;&theta;<sub>j</sub></sub>(J(&theta;))
        
        } Simultaneously update all &theta;<sub>j</sub>
        
        The form is identical to linear regression, but h<sub>&theta;</sub>(x) = <sup>1</sup>&frasl;<sub>(1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup>)</sub>
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
        * Advantages:
            
            1. No need to manually pick &alpha;
            
            1. Ofter faster than gradient descent
        * Disadvantage: More complex
1. Multicalss classification
    * Examples:
        * Email tagging/folding: work, friends, hobby
        * Weather: sunny, cloudy, rainy
        * Medical diagram: not fill, cold, flu
    * one-vs-all
    
        ![one-vs-all](../images/BinaryClassVsMultiClass.png)
        
        **h<sub>&theta;</sub><sup>(i)</sup>(x) = P(y=i|x;&theta;), (i = 1, 2, 3, ...)**
        * Train a logistic regression classifier h<sub>&theta;</sub><sup>(i)</sup>(x) for each class i to predict the probability that y = i
        * On a new input x, to make a prediction, pick the class that maximizes
            
            max h<sub>&theta;</sub><sup>(i)<sup>(x)
