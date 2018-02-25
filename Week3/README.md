## Week 3
1. Classification
    Examples: 
        * Emails: spam/not spam
        * Online transactions: fradulent or not
        * Medical judgements: malignant or not
    * Idea is to use threshold classifier output h<sub>&theta;</sub>(x) at 0.5
        * If h<sub>&theta;</sub>(x) &ge; 0.5, predict y = 1
        * If h<sub>&theta;</sub>(x) &ge; 0.5, predict y = 0
    * Logistic regression
        * Hypothesis representation (we want 0 &ge; h<sub>&theta;</sub>(x) h<sub>&theta;</sub>(x) 1)

            h<sub>&theta;</sub>(x) = g(&theta;<sup>T</sup>&sdot;x)

            g(z) = <sup>1</sup>;frasl;<sub>1+e<sup>-z</sup></sub> (Sigmoid Function or Logistic Function)

            &rArr; **h<sub>&theta;</sub>(x) = <sup>1</sup>;frasl;<sub>1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup></sub>**
        * Interpretation of hypothesis

            &theta;<sup>T</sup>&sdot;x = estimated probability that y = 1 on input x

            or **&theta;<sup>T</sup>&sdot;x = P(y=1|x;&theta;)**

            and P(y=1|x;&theta;)+P(y=0|x;&theta;) = 1

        * Decision boundary - a property of hypothesis
            Since g(z) &ge; 0.5 when z &ge; 0, h<sub>&theta;</sub>(x) &ge; 0.5 whenever &theta;<sup>T</sup>&sdot;x &ge; 0

            The decision boundary can also be non-linear
    * Logistic Regressioin Model
        * Cost function
            Linear regression has J(&theta;) = <sup>1</sup>&frasl;<sub>m</sub>&sdot;&sum;<sup>1</sup>&frasl;<sub>2</sub>&sdot;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
            where Cost(h<sub>&theta;</sub>(x<sup>(i)</sup>), y<sup>(i)</sup>) &rarr; <sup>1</sup>&frasl;<sub>2</sub>&sdot;(h<sub>&theta;</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
            We may want to use similar function, say Cost(h<sub>&theta;</sub>(x), y) = <sup>1</sup>&frasl;<sub>2</sub>&sdot;(h<sub>&theta;</sub>(x)-y)<sup>2</sup>, and h<sub>&theta;</sub>(x) = <sup>1</sup>;frasl;<sub>1+e<sup>-&theta;<sup>T</sup>&sdot;x</sup></sub>
            but if use this function, J(&theta;) will be "non-convex"
            Instead we will construct a logistic regression cost function
        * Logistic regression cost function:
            Cost(h<sub>&theta;</sub>(x), y) = -log(h<sub>&theta;</sub>(x)), if y = 1
                                            = -log(1-h<sub>&theta;</sub>(x)), if y = 0
        * Case 1: y = 1
            Intuition: 
                        cost = 0 if y = 1 and h<sub>&theta;</sub>(x) = 1

                        but as h<sub>&theta;</sub>(x) &rarr; 0, cost &rarr; &infin;
            
            Can be understood as, if h<sub>&theta;</sub>(x) = 0, hypothesis prdicts P(y=1|x;&theta;) = 0, but y is actually 1. Then we will penalize learning algorithm by a large cost
        * Case 2: y = 0
            Same idea, we will penalize the case h<sub>&theta;</sub>(x) &rarr; 1 when y = 0


