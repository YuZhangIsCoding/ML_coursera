## Week 10

1. Learning With Large Datasets

    * Perform sanity check (learning curve) before building algorithm for large dataset

        If the algorithm suffers from large variance, adding more datasets are likely to help

        ![LargeVariance](../images/LearningCurveVariance.jpg)

        If the algorithm suffers from large bias, we may want to create more features before increase the datasets

        ![LargeBias](../images/LearningCurveBias.jpg)

    * Stochastic Gradient Descent

        Take linear regression as an example,

        Define:

        <img src="http://latex.codecogs.com/svg.latex?Cost(\theta,(x^{(i)},y^{(i)}))=\frac{1}{2}(h_{\theta}(x^{(i)})-y^{(i)})^2"/>

        And the cost function can be expressed as:

        <img src="http://latex.codecogs.com/svg.latex?J_{train}(\theta)=\frac{1}{m}\sum_{i=1}^{m}Cost(y,(x^{(i)},y^{(i)}))"/>

        * Algorithm

            Randomly shuffle datasets

            Repeat{    &larr; 1-10 times

            &nbsp;&nbsp;&nbsp;&nbsp;for i = 1, ..., m {

            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\theta_j=\theta_j-\alpha(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}, for j = 0, ..., n

            &nbsp;&nbsp;&nbsp;&nbsp;}

            }

        Batch gradient descent will move directly to global minimum, but stochastic gradient descent will move in some region near the global minimum.

    * Mini-Batch Gradient Descent

        Use b examples in each iteration.

        *Algorithm:

            Repeat{

            &nbsp;&nbsp;&nbsp;&nbsp;for i = 1, 1+b ..., m-b+1 {

            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\theta_j=\theta_j-\alpha\frac{1}{b}\sum_{k=i}^{i+b-1}(h_\theta(x^{(k)})-y^{(k)})x_j^{(k)}, for j = 0, ..., n

            &nbsp;&nbsp;&nbsp;&nbsp;}

            }

    * Checking for convergence

        For batch gradient descent, we could plot J(&theta;) as a function of number of iterations.

        For stochastic gradient descent, plot Cost(*theta;, (x<sup>(i)</sup>, y<sup>(i))</sup>) averaged over the last 1000 examples processed by algorithm.

        Learning rate is typically hold constan, but we can also slowly decrease &alpha; over time if we want &theta; to converge. E.g.

        <img src="http://latex.codecogs.com/svg.latex?\alpha=\frac{const1}{iterationN+const2}"/>

1. Online Learning
