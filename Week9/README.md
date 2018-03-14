## Week 9
1. Anomaly Detection
    * Motivation

        Aircraft engine checks

        ![AircraftEngine](../images/AircraftEngine.jpg)

    * Density estimation
        
        Given dataset {x<sup>1</sup>, ..., x<sup>1</sup>), is x<sub>test</sub> anomalous?

        Model p(x):

        p(x<sub>test</sub>) < &epsilon; &rarr; anomaly

        p(x<sub>test</sub>) &ge; &epsilon; &rarr; OK

        ![DensityEstimation](../images/DenEsti.jpg)

    * Examples
        
        * Fraud detection
        * Manufacturing
        * Monitoring computers in a data center

    * Gaussian Distribution
        
        * Say x &isin; R, if x is distributed Gaussian with mean &mu; and variance &sigma;<sup>2</sup>

            x ~ N(&mu;, &sigma;<sup>2</sup>)

            <img src="http://latex.codecogs.com/svg.latex?p(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\textrm{exp}\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)"/>

        * Examples with different &mu; and &sigma;

            ![GaussianExamples](../images/GaussianExamples.png)

        * Parameter estimation

            <img src="http://latex.codecogs.com/svg.latex?\mu_j=\frac{1}{m}\sum_{i=1}^{m}x_j^{(i)}"/>

            <img src="http://latex.codecogs.com/svg.latex?\sigma^2=\frac{1}{m}\sum_{i=1}^{m}(x_j^{(i)}-\mu_j)^2"/>
        
        * Anomaly detection algorithm

            Choose features x<sub>i</sub> taht you think might be indicative of anomalous examples

            Fit parameters &mu;<sub>1</sub>, ..., &mu;<sub>n</sub>, &sigma;<sub>1</sub><sup>2</sup>, ..., &sigma;<sub>n</sub><sup>2</sup>

            Given new example x, compute p(x):

            &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?p(x)=\prod_{j=1}^{n}p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^{n}\frac{1}{\sqrt{2\pi}\sigma_j}\textrm{exp}\big(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}\big)"/>

            Anomaly if p(x) < &epsilon;

2. Building an anomaly detection system
    * Aircraft engines motivating example
        
        Assume that we have 10000 good engines and 20 flawed engines, we could build the detection system as :

        * Training set: 6000 good engines; Cross validation set: 2000 good engines and 10 anomalous ones; Test set: 2000 good engines and 10 anomalous ones

        * Alternatively even though not recommended by Andrew. Training set: 6000 good; CV: 4000+10; test set: 4000+10; Where same good engine examples or same anomalous engines were used for CV set and test set.

    * Algorithm evaluation:

        The data may be quite skewed, so we need additional evaluation metrics other than prediction accuracy

        * True positive, false positive, false negative and true negative
        * Precision and recall
        * F<sub>1</sub>-score

        The cross validation test can also be used to choose parameter &epsilon;

    * Anomaly Detection VS. Supervised Learning

        Anomaly Detection|Supervised Learning
        -|-
        Very small number of positive examples (0-20 is common) and large number of negative examples|Large number of positive and negative examples
        Contains many different types of anomalies, which are hard for any algorithm to learn from positive examples|Enough positive examples to get a sense of what examples are like
        Future anomalies may not look like any anomalies seen before|Future positive anomalies likely to be similar to ones in training set
        * Application
            
            Anomaly Detection|Supervised Learning
            -|-
            Fraud Detection|Email spam
            Manufacturing|Weather prediction
            Monitoring machines in data center|cancer classification

            We may swithc from anomaliy detection to supervised learning when enough examples were given.

    * Features to use

        * Non-Gaussian features

            Sometimes the histogram of training set does not look like Gaussian distributed we may try to "correct" it by applying additional functions, such as *log(x+c)*, *x<sup>c</sup>*, ...

            ![NonGaussian](../images/NonGaussian.png)

        * Error analysis

            Want p(x) large for normal and small for anomalous

            Most common problem: p(x) is comparable for normal and anomalous:

            **Look at examples that are close to cutoff and create new features**

            ![AnomalyAddFeature](../images/AnomalyAddFeature.jpg)

        * Monitoring computers in a data center

            Chose features that are unusually large or small in the event of an anomaly

            x1 = memory use

            x2 = number of disk accesses

            x3 = cpu load

            x4 = network traffic

            x5 = x3/x4

            ...
3. Multivariate Gaussian Distribution
    * Motivation
        
        ![MultivariateGaussian](../images/MultivariateGaussian.png)
        
        *The anomaly looks not so bad on both plots, but fails to predict*

        Instead of model p(x<sub>i</sub>) separately, model p(x) all in one go.

    * Parameters
        
        <img src="http://latex.codecogs.com/svg.latex?\mu_j=\frac{1}{m}\sum_{i=1}^{m}x_j^{(i)}"/>

        <img src="http://latex.codecogs.com/svg.latex?]\sum=\frac{1}{m}\sum_{i=1}^{m}(x^{(i))-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}{\cdot}X^T{\cdot}X"/>

        The distribution will change with respect to &mu; (location of center) and &sum; (the shape of distribution)

        ![MultiGaussian_Sigma](../images/MultiGaussian_Sigma.png)

        And p(x) is given by

        <img src="http://latex.codecogs.com/svg.latex?p(x;\mu,\sum)=\frac{1}{(2\pi)^{\frac{n}{2}}|\sum|^{\frac{1}{2}}}\textrm{exp}\big(-\frac{1}{2}(x-\mu)^T{\sum}^{-1}(x-\mu)\big)"/>

        If p(x) < &epsilon; flag an anomaly.

    * Original model is just a special case of multivariate Gaussian distribution

        <img src="http://latex.codecogs.com/svg.latex?\sum=\begin{matrix}\sigma_1^2&0&{\dots}&0\\0&\sigma_2^2&{\dots}&0\\\vdots&\vdots&\ddots&\vdots\\0&0&{\dots}&\sigma_n^2\end{bmatrix}"/>

    * Comparision

        Original Model|Multivariate Gaussian
        -|-
        Manually create features to capture anomalies when features take unusual combinations|Automatically captures correlations between features
        Computationally cheaper, n can go large (10,000 to 100,000)|Computationally more expensive because of the calculation of the reverse of &sum;
        Ok if m is small|Must have m > n, otherwise &sum; is non-invertible (may also be resulted from redundant features)
        
