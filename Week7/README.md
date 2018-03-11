## Week 7
1. Support Vector Machine (Large Margin Classifier)
    * Alternative view of logistic regression
    
        <img src="https://latex.codecogs.com/svg.latex?h_\theta(x)=\frac{1}{1+\textrm{e}^{-\theta^Tx}}"/>
        
        If y = 1, we want h<sub>&theta;</sub>(x) &asymp; 1 &rarr; &theta;<sup>T</sup>x >> 0
        
        If y =0, we want h<sub>&theta;</sub>(x) &asymp; 0 &rarr; &theta;<sup>T</sup>x << 0

        We can build cost function as following:

        <img src="https://latex.codecogs.com/svg.latex?J(\theta)=C\sum_{i=1}^{m}\big(y^{(i)}Cost_1(\theta^Tx^{(i)})+(1-y^{(i)})Cost_0(\theta^Tx^{(i)})+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2"/>

        Where C acts like <sup>1</sup>&frasl;<sub>&lambda;</sub> in the logistic regression, and Cost<sub>1</sub>(z) and Cost<sub>0</sub>(z) look like below:

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

            ![SVM_outliers](../images/SVM_outliers.jpg)

    * The math behind SVM

        * Vector inner product

            Let u = [u1; u2], v = [v1; v2]

            Define ||u|| = length of vector = &radic;(u<sub>1</sub><sup>2</sup>+u<sub>2</sub><sup>2</sup>)

            and p is the length of the vector onto u (signed), as illustrated below:

            ![InnerProduct](../images/InnerProd.jpg)

            Thus u<sup>T</sup>&sdot;v = u<sub>1</sub>v<sub>1</sub>+u<sub>2</sub>v<sub>2</sub> = p&sdot;||u||

        * SVM decision boundary

            Let's simplify this problem to n = 2, and set &theta;<sub>0</sub> = 0

            And the object function is to 

            minimize 1/2&sum;&theta;<sub>j</sub><sup>2</sup> = 1/2(&theta;<sub>1</sub><sup>2</sup>+&theta;<sub>2</sub><sup>2</sup>) = 1/2||&theta;||<sup>2</sup>

            Such that when C is very large:

            &theta;<sup>T</sup>x<sup>(i)</sup> = p<sup>(i)</sup>&sdot;||&theta;|| &ge; 1, if y<sup>(i)</sup> = 1

            &theta;<sup>T</sup>x<sup>(i)</sup> = p<sup>(i)</sup>&sdot;||&theta;|| &le; -1, if y<sup>(i)</sup> = 0

            As illustrated in the fig below, the vector of &theta; is perpendicular to the decision boundary.

            ![SVM_SmallerMargin](../images/SVM_SmallerMargin.jpg)

            If y<sup>(i)</sup> = 1, then we need p<sup>(i)</sup>&sdot;||&theta;|| &ge; 1. If p<sup>(i)</sup> is small, then need ||&theta;|| to be large, which goes against our obejective.

            Same thing for y<sup>(i)</sup> = 0, and we need p<sup>(i)</sup>&sdot;||&theta;|| &le; -1. If p<sup>(i)</sup> is close to 0, then need ||&theta;|| to be large.

            And in this following case, since the margin is large, ||&theta;|| can be smaller.

            ![SVM_LargerMargin](../images/SVM_LargerMargin.jpg)

2. Kernels
    * Non-linear decision boundary
      
        Suppose we have a data set as following:
     
        ![NonLinearDecisionBoundary](../images/NonLinearDeciBound.jpg)
        
        One way to construct is add polynomial features, e.g. let f<sub>1</sub> = x<sub>1</sub>, f<sub>2</sub> = x<sub>2</sub>, f<sub>3</sub> = x<sub>1</sub>x<sub>2</sub>, ...
        
        And predict y = 1 if &theta;<sup>T</sup>f &ge; 0
     
        *Do we have other choices?*
      
    * Kernel
   
        ![Landmarks](../images/Landmarks.jpg)
        
        Given x:
        
        f<sub>1</sub> = Similarity(x, l<sup>(1)</sup>)
        
        f<sub>2</sub> = Similarity(x, l<sup>(2)</sup>)
        
        ...
      
    * Gaussian Kernal (RBF kernel)
   
        The Similarity function can be represented by a Gaussian kernel
        
        <img src="https://latex.codecogs.com/svg.latex?f_i=similarity(x,l^{(i)})=\textrm{exp}(-\frac{||x-l^{(i)}||}{2\sigma^2})"/>
        
        If x &asymp; l<sup>(i)</sup>, f &asymp; 1
        
        If x is far from l<sup>(i)</sup>, f &asymp; 0
        
        And &sigma; determines the shape of gaussian kernel
        
        ![Gaussian_Sigma](../images/Gaussian_Sigma.png)
      
    * Example
   
        Let &theta;<sub>0</sub> = -0.5, &theta;<sub>1</sub> = 1, &theta;<sub>2</sub> = 1 and &theta;<sub>3</sub> = 0
        
        If x is close to l<sup>(1)</sup> or l<sup>(2)</sup>, &theta;<sup>T</sup>f = -0.5+1 > 0 &rarr; y = 1
        
        Otherwise, &theta;<sup>T</sup>f = -0.5 < 0 &rarr; y = 0
        
        And the decision boundary will looks like below:
        
        ![Gaussian_DeciBound](../images/Gaussian_DeciBound.jpg)

    * Choose landmarks
      
        Given (x<sup>(1)</sup>, y<sup>(1)</sup>), ..., (x<sup>(m)</sup>, y<sup>(m)</sup>)
        
        Choose l<sup>(1)</sup> = x<sup>(1)</sup>, ..., l<sup>(m)</sup> = x<sup>(m)</sup>
        
        Given training/testing example x<sup>(i)</sup>
        
        f<sub>1</sub><sup>(i)</sup> = Similarity(x<sup>(i)</sup>, l<sup>(1)</sup>)
        
        ...
        
        f<sub>i</sub><sup>(i)</sup> = Similarity(x<sup>(i)</sup>, l<sup>(i)</sup>) = 1
        
        ...
        
        f<sub>m</sub><sup>(i)</sup> = Similarity(x<sup>(i)</sup>, l<sup>(m)</sup>)
        
        Hypothesis:
        
        Given X, compute f &isin; R<sup>m+1</sup>, and predict y = 1 if &theta;<sup>T</sup>f &ge; 0
      
    * SVM parameters
   
        * C (<sup>1</sup>&frasl;<sub>&lambda;</sub>)
           
           Large C: lower bias, high variance
           
           Small C: higher bias, low variance
        * &sigma;<sup>2</sup>
           
           Large &sigma;<sup>2</sup>: features f vary more smoothly &rarr; higher bias, low variance
           
           Small &sigma;<sup>2</sup>: features f vary less smoothly &rarr; lower bias, high variance
   
    * Choices of kernel
      
        * No kernel ("linear kernel")
         
            Predict y = 1 if &theta;<sup>T</sup>x &ge; 0, when n is large and m is small
      
        * Gaussian kernel
         
            when n is small and/or m is large
         
            **need to choose appropriate  &sigma;<sup>2</sup>**
         
            **feature scaling before using Gaussian kernel**
      
        * Other choices
         
            Need to satifsy ["Mercer's theorem"](https://en.wikipedia.org/wiki/Mercer%27s_theorem) to make sure SVM package's optimization run correctly and do not diverge
         
            [Polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel)
         
            [String kernel](https://en.wikipedia.org/wiki/String_kernel)
         
            [chi-square kernel](https://en.wikipedia.org/wiki/Chi-squared_distribution)
         
            [histogram intersection kernel](http://ieeexplore.ieee.org/document/1247294/)
         
            ...
         
    * Multiclass Classification
   
        Many SVM packages already have builtin multi-class classification functionality
      
        Otherwise, use one-vs-all method
      
    * Logistic Regression vs. SVM
      
        n = features, m = training examples
      
        * If n is large (relative to m), e.g. n = 10,000, m = 10-1000
            
            Use logistic regression or SVM without a kernel
      
        * If n is small, m is intermediate, e.g. n = 1-1000, m = 10-10,000
         
            Use SVM with Gaussian Kernel
      
        * If n is small, m is large, e.g. n = 1-1000, m = 50,000+
         
            Create/add more features, then use logistic regression or SVM without a kernel
      
        Neural Network likely to work well for most of these settings, but maybe slow to train.
   
