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
## [Week 6](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week6/README.md)
1. Debugging a learning algorithm
    * If found unacceptable large errors in its prediction, try these:
        1. Get more training examples &rarr; fixes high variance
        2. Try smaller sets of features &rarr; fixes high variance
        3. Try additional features &rarr; fiexes high bias
        4. Try adding polynomial features &rarr; fixes high bias
        5. Try descreasing &lambda; &rarr; fixes high bias
        6. Try increasing &lambda; &rarr; fixes high variance
    * Evaluate a hypothesis
        
        * Split data set to 2 sets: 70% training set and 30% test set are common defaults many people uses
        * Learn parameter &theta; from training data (minimize training error J<sub>train</sub>(&theta;))
        * Compute test set error J<sub>test</sub>(&theta;)

    * Model selection
        * Instead of dividing into just 2 data sets, we can split the data into 3 sets: 60% training set, 20% cross-validation set and 20% test set, which are common defaults many people use
        * Learn parameter &theta; and obtain training error
        * Calculate cross-validation error
        * Pick model from cross-validation step and calculate test error

    * Bias vs. Variance
    
        ![BiasVsVariance](images/BiasVsVariance.jpg)     
        
        * High bias: underfitting, J<sub>test</sub>(&theta;) &asymp; J<sub>train</sub>(&theta;)
        * High variance: overfitting, J<sub>train</sub>(&theta;) will be low and J<sub>cv</sub>(&theta;) >> J<sub>train</sub>(&theta;)

    * Regularization and Bias/Variance: Bias/Variance as a function of regularization parameter &lambda;

        ![RegularizationBiasVariance](images/RegularizationBiasVariance.jpg)

    * Learning curves: training/cross-validation error as a function of training size
        
        * High bias: training error will be closer to the cross-validation error, and getting more data will not help too much
            
            ![LearningCurveBias](images/LearningCurveBias.jpg)

        * High variance: training error is much lower than cross-validation error, and getting more data is likely to help

            ![LearningCurveVariance](images/LearningCurveVariance.jpg)

    * Neural Network and overfitting
        * "Small" neural network (fewer features), computationally cheaper, but more prone to underfitting
        * "Large" neural network (more parameters), computationally more expensive, more prone to overfitting, use regularization to address overfitting.

1. Spam Classifier
    * Advices to make your classifier have low error:
        * Collect lots of data, e.g. ["honey pot" project](https://en.wikipedia.org/wiki/Project_Honey_Pot).
        * Develop sophiscated features based on email routing infomation (email header).
        * Develop sophiscated features based on message bodies, e.g. distinguish synonyms, features about punctuations, etc.
        * Develop sophiscated algorithm to detect misspelling and improve input correctness.
    * Recommended approach
        * Start with a simple algorithm and test it on cross-validation data
        * Plot learning curve to decide if more data or more features needed, etc.
        * Error analysis: manually examine the examples that your algorithm made errors on. See if any systemetic trend of errors made.
    * Precision/Recall
    
        ![Precision/Recall](images/Precision_Recall.jpg)

    * Trading off precision and recall

        ![TradeOffPrecisionRecall](images/TradeOffPrecsionRecall.png)

    * F<sub>1</sub> Score (F score)
        
        <img src="https://latex.codecogs.com/svg.latex?F_1Sore=2\frac{P\cdot{R}}{P+R}"/>
    
    * Data for Machine Learning
        
        > "It's not who has the best algorithm that wins. It's who has the most data."

    * Large Date Rationale
        
         1. Use a learning algorithm with many parameter (logisti/linear regression with many feature; neural network with many hidden layers), which gives low bias &rarr; J<sub>train</sub>(&theta;) will be small
         2. Use large training set, which makes it unlikely to overfit &rarr; J<sub>train</sub>(&theta;) &asymp; J<sub>test</sub>(&theta;)
## [Week 7](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week7/README.md)
1. Support Vector Machine (Large Margin Classifier)
    * Alternative view of logistic regression
        We can build cost function as following:

        <img src="https://latex.codecogs.com/svg.latex?J(\theta)=C\sum_{i=1}^{m}\big(y^{(i)}Cost_1(\theta^Tx^{(i)})+(1-y^{(i)})Cost_0(\theta^Tx^{(i)})+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2"/>

        Where C acts like <sup>1</sup>&frasl;<sub>&lambda;</sub> in the logistic regression, and Cost<sub>1</sub>(z) and Cost<sub>0</sub>(z) look like below:

        ![SVM_cost](images/SVM_cost.jpg)

        And the hypothesis is:

        *h<sub>&theta;</sub>(x) = 1, if &theta;<sup>T</sup>x &ge; 0*
        
        *h<sub>&theta;</sub>(x) = 0, otherwise*

    * Large Margin Intuition

        **minimize 1/2&sum;&theta;<sub>j</sub><sup>2</sup>**

        such that

        **&theta;<sup>T</sup>x<sup>(i)</sup> &ge; 1, if y<sup>(i)</sup> = 1**

        **&theta;<sup>T</sup>x<sup>(i)</sup> &le; -1, if y<sup>(i)</sup> = 0**

    * SVM decision boundary
        * Linearly separable case
            
            ![SVM_linear_separable](images/SVM_linear_Sep.jpg)

            Mathematically SVM select model with large margins

        * In presence of outliers

            ![SVM_outliers](images/SVM_outliers.jpg)

    * The math behind SVM

        * Vector inner product

            ![InnerProduct](images/InnerProd.jpg)

            Thus u<sup>T</sup>&sdot;v = u<sub>1</sub>v<sub>1</sub>+u<sub>2</sub>v<sub>2</sub> = p&sdot;||u||

        * SVM decision boundary: p<sup>(i)</sup>&sdot;||&theta;|| &ge; 1 or p<sup>(i)</sup>&sdot;||&theta;|| &le; -1

            minimize 1/2&sum;&theta;<sub>j</sub><sup>2</sup> = 1/2||&theta;||<sup>2</sup>

            And larger margin can decrease ||&theta;||

            ![SVM_SmallerMargin](images/SVM_SmallerMargin.jpg)
            ![SVM_LargerMargin](images/SVM_LargerMargin.jpg)

2. Kernels
    * Kernel: f<sub>i</sub> = Similarity(x, l<sup>(i)</sup>), i = 1, ..., m
   
        ![Landmarks](images/Landmarks.jpg)
        
    * Gaussian Kernal (RBF kernel): If x &asymp; l<sup>(i)</sup>, f &asymp; 1; if x is far from l<sup>(i)</sup>, f &asymp; 0
   
        <img src="https://latex.codecogs.com/svg.latex?f_i=similarity(x,l^{(i)})=\textrm{exp}(-\frac{||x-l^{(i)}||}{2\sigma^2})"/>
        
        ![Gaussian_Sigma](images/Gaussian_Sigma.png)
      
    * Choose landmarks: 
        
        Given (x<sup>(1)</sup>, y<sup>(1)</sup>), ..., (x<sup>(m)</sup>, y<sup>(m)</sup>), 
        
        Choose l<sup>(1)</sup> = x<sup>(1)</sup>, ..., l<sup>(m)</sup> = x<sup>(m)</sup>
        
        Compute f<sub>i</sub><sup>(j)</sup> = Similarity(x<sup>(j)</sup>, l<sup>(i)</sup>), where i, j = 1, ..., m
      
    * SVM parameters
   
        * C (<sup>1</sup>&frasl;<sub>&lambda;</sub>)
           
           Large C: lower bias, high variance
           
           Small C: higher bias, low variance
        * &sigma;<sup>2</sup>
           
           Large &sigma;<sup>2</sup>: features f vary more smoothly &rarr; higher bias, low variance
           
           Small &sigma;<sup>2</sup>: features f vary less smoothly &rarr; lower bias, high variance
   
    * Choices of kernel
      
        * No kernel ("linear kernel"): n is large and m is small
      
        * Gaussian kernel: n is small and/or m is large; need to choose appropriate  &sigma;<sup>2</sup>; feature scaling before using Gaussian kernel
      
        * Other choices: [Polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel); [String kernel](https://en.wikipedia.org/wiki/String_kernel); [chi-square kernel](https://en.wikipedia.org/wiki/Chi-squared_distribution); [histogram intersection kernel](http://ieeexplore.ieee.org/document/1247294/)
    * Multiclass Classification: Builtin or one-vs-all
    * Logistic Regression vs. SVM

        * If n is large: logistic regression or SVM without a kernel
        * If n is small, m is intermediate: SVM with Gaussian Kernel
        * If n is small, m is large: Create/add more features, then use logistic regression or SVM without a kernel

        Neural Network likely to work well for most of these settings, but maybe slow to train.
   
## [Week 8](https://github.com/YuZhangIsCoding/ML_coursera/blob/master/Week8/README.md)
1. Clustering
    * Unsupervised learning

        ![UnsupervisedLearning](images/UnsupervisedLearning.jpg)

    * Optimization objective

        <img src="https://latex.codecogs.com/svg.latex?J(c^{(1)},...,c^{(m)},\mu_1,...,\mu_K)=\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-\mu_{c^{(i)}}||^2"/>

    * K-means algorithm
        
        ![Kmeans](images/Kmeans.jpg)

        **Randomly initialize K cluster centroids: randomly pick K (K < m) traning examples and set &mu;<sub>1</sub>, ..., &mu;<sub>K</sub> to these K examples.**

        **Repeat{**

        **for i = 1 to m**
        
        &nbsp;&nbsp;&nbsp;&nbsp;**C<sup>(i)</sup> = index (from 1 to K) of cluster cnetroid closest to x<sup>(i)</sup>** &larr; *Cluster assignment step* &rarr; minimize J(...) with respect to C<sup>(1)</sup>, ..., C<sup>(m)</sup>, while holding &mu;<sub>1</sub>, ..., &mu;<sub>K</sub> fixed

        **for k = 1 to K**

        &nbsp;&nbsp;&nbsp;&nbsp;**&mu;<sub>k</sub> = average of points assigned to cluster k** &larr; *Move centroid step* &rarr; minimize J(...) with respect to &mu;<sub>1</sub>, ..., &mu;<sub>K</sub>

        **}**


        * *Note: if no sample assigned to a centroid, we can*

            1. *Delete that centroid, and decrease the number of centroids to K-1 (more common).*
    
            2. *Randomly reinitialize the centroid if need K clusters.*

    * Local optima: try initializing several different times and **pick clustering that gives lowest cost function J(...)**

        * K = 2-10, random initialization works pretty well

        * K >> 10, just slight improve after random initialization

    * Choosing the number of clusters: 

        * Mainly by hand, human judgement
            
        * Elbow method: plot cost function vs. K

            ![Elbow_method](images/Elbow_Method.jpg)

            "Worth a shot, but won't have a high expectation", because many times, there are no clear elbow.

        * Later/downstream purpose

2. Dimensionality Reduction: Principal Component Analysis (PCA)
    * Motivation
        * Data compression: reduce data from 2D to 1D if datas fall near a line, or reduce data from 3D to 2D if data falls near a plane, etc.
            ![DataCompression](images/DataCompression.jpg)

        * Data visualization: simplify the features to 2 or 3 most comprehensive and important features
    * PCA problem formulation: Reduce from n-dimension to K-dimension:

        Find K vectors u<sup>(1)</sup>, ..., u<sup>(K)</sup> onto which to project the data, so as to minimize the projection error.

    * PCA is not linear regression: **Linear regression minimizes vetical distances, while PCA minimizes orthogonal distances, where data is not labeled and treated equally**

        ![PCAnotLinearReg](images/PCAnotLinearReg.jpg)

    * PCA algorithm with vectorized implementation
        
        **After mean normalization and feature scaling**

        **Compute ["Covariance matrix"](https://en.wikipedia.org/wiki/Covariance_matrix): &sum; = <sup>1</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;X**

        **Compute "eigenvectors" of matrix &sum;: [U, S, V] = svd(&sum;)**

        **Select first K eigenvectors as U<sub>reduce</sub>: U<sub>reduce</sub> = U(:, 1:K)**

        **Compute the projection from X to Z: Z = X&sdot;U<sub>reduce</sub>**

    * Reconstruction from compressed representation: **X<sub>approx</sub> = Z&sdot;U<sub>reduce</sub>**

    * Choose the number of pricinple components: choose K to be the smallest value, so that **"99% of variance is retained"**

        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\frac{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-x_{approx}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2}\le0.01"/>

    * Choose K algorithm:
        
        **Try PCA with K = 1, 2, ...**

        **Compute [U, S, V] = svd(&sum;)**

        **Pick the smallest K that satisfies:**
            
        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\frac{\sum_{i=1}^{K}S_{ii}}{\sum_{i=1}^{n}S_{ii}}\ge0.99"/>

    * Application of PCA: *speed up supervised learning*, *reduce memory/disk needed to store data*, *visualization with K = 2 or 3*
        
    * Bad use of PCA: prevent overfitting, blindly use PCA without testing raw data
