## Week 7
1. Support Vector Machine (Large Margin Classifier)
    * Alternative view of logistic regression
        <img src="https://latex.codecogs.com/svg.latex?h_\theta(x)=\frac{1}{1+\textrm{e}^{-\theta^Tx}}"/>
        
        If y = 1, we want h<sub>&theta;</sub>(x) &asymp; 1 &rarr; &theta;<sup>T</sup>x >> 0
        
        If y =0, we want h<sub>&theta;</sub>(x) &asymp; 0 &rarr; &theta;<sup>T</sup>x << 0

        We can build cost function as following:

        <img src="https://latex.codecogs.com/svg.latex?J(\theta)=C\sum_{i=1}^{m}\big(y^{(i)}Cost_1(\theta^Tx^{(i)})+(1-y^{(i)})Cost_0(\theta^Tx^{(i)})+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2"/>

        Where C acts like 1/&lambda; in the logistic regression, and Cost<sub>1</sub>(z) and Cost<sub>0</sub>(z) look like below:

        ![SVM_cost](../images/SVM_cost.jpg)

        And the hypothesis is:

        *h<sub>&theta;</sub>(x) = 1, if &theta;<sup>T</sup>x &ge; 0*
        
        *h<sub>&theta;</sub>(x) = 0, otherwise*

    * Large Margin Intuition

        Suppose we have a very large C, e.g. C = 100,000, then

        If y = 1, we want &theta;<sup>T</sup>x &ge; 1 (not just &ge; 0)
        
        If y = 0, we want &theta;<sup>T</sup>x &le; -1 (not just < 0)

        Thus, whenever y<sup>(i)</sup> = 1, the first term in the cost function equals C&times;(1&times;0+0&times;Cost<sub>0</sub>) = 0

        And if y<sup>(i)</sup> = 0, we also have C&times;(0&times;Cost<sub>1</sub>+1&times0) = 0

        Then our objective function is to 

        **minimize 1/2&sum;&theta;<sub>j</sub><sup>2</sup>**

        such that

        **&theta;<sup>T</sup>x<sup>(i)</sup> &ge; 1, if y<sup>(i)</sup> = 1**

        **&theta;<sup>T</sup>x<sup>(i)</sup> &le; -1, if y<sup>(i)</sup> = 0**

    * SVM decision boundary
        * Linearly separable case
            
            ![SVM_linear_separable](../images/SVM_linear_Sep.jpg)

            Mathematically SVM select model with large margins

        * In presence of outliers

            ![SVM_outliers](../images/SVM_outliers)

    * The math behind SVM

        * Vector inner product

            Let u = [u1; u2], v = [v1, v2]

            Define ||u|| = length of vector = &radic;(u<sub>1</sub><sup>2</sup>+u<sub>2</sub><sup>2</sup>)

            and p is the length of the vector onto u (signed), as illustrated below:

            ![InnerProduct](../images/InnerProd.jpg)

            Thus u<sup>T</sup>&sdot;v = u<sub>1</sub>v<sub>1</sub>+u<sub>2</sub>v<sub>2</sub> = p&dsot;||u||

        * SVM decision boundary

            Let's simplify this problem to n = 2, and set &theta<sub>0</sub> = 0

            And the object function is to 

            minimize 1/2&sum;&theta;<sub>j</sub><sup>2</sup> = 1/2(&theta;<sub>1</sub><sup>2</sup>+&theta;<sub>2</sub><sup>2</sup>) = 1/2||&theta;||<sup>2</sup>

            Such that when C is very large:

            &theta;<sup>T</sup>x<sup>(i)</sup> = p<sup>(i)</sup>&sdot;&theta; &ge; 1, if y<sup>(i)</sup> = 1

            &theta;<sup>T</sup>x<sup>(i)</sup> = p<sup>(i)</sup>&sdot;&theta; &le; -1, if y<sup>(i)</sup> = 0

            As illustrated in the fig below, the vector of &theta; is perpendicular to the decision boundary.

            ![SVM_SmallerMargin](../images/SVM_SmallerMargin.jpg)

            If y<sup>(i)</sup> = 1, then we need p<sup>(i)</sup>&sdot;&theta; &ge; 1. If p<sup>(i)</sup> is small, then need ||&theta;|| to be large, which goes against our obejective.

            Same thing for y<sup>(i)</sup> = 0, and we need p<sup>(i)</sup>&sdot;&theta; &le; -1. If p<sup>(i)</sup> is close to 0, then need ||&theta;|| to be large.

            And in this following case, since the margin is large, ||&theta;|| can be smaller.

            ![SVM_LargerMargin](../images/SVM_LargerMargin.jpg)
