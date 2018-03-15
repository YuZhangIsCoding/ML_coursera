## Week 9
1. Anomaly Detection
    * Motivation

        Aircraft engine checks

        ![AircraftEngine](../images/AircraftEngine.jpg)

    * Density estimation
        
        Given dataset {x<sup>(1)</sup>, ..., x<sup>(m)</sup>), is x<sub>test</sub> anomalous?

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

            <img src="http://latex.codecogs.com/svg.latex?\mu_j=\frac{1}{m}\sum_{i=1}^{m}x_j^{(i)},"/>

            <img src="http://latex.codecogs.com/svg.latex?\sigma_j^2=\frac{1}{m}\sum_{i=1}^{m}(x_j^{(i)}-\mu_j)^2"/>
        
        * Anomaly detection algorithm

            Choose features x<sub>i</sub> that you think might be indicative of anomalous examples

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

            We may switch from anomaly detection to supervised learning when enough examples were given.

    * Features to use

        * Non-Gaussian features

            Sometimes the histogram of training set does not look like Gaussian distributed. Even though the model probablity works fine, but we may try to "correct" it by applying additional functions, such as *log(x+c)*, *x<sup>c</sup>*, ...

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
        
        <img src="http://latex.codecogs.com/svg.latex?\mu_j=\frac{1}{m}\sum_{i=1}^{m}x_j^{(i)},"/>

        <img src="http://latex.codecogs.com/svg.latex?\sum=\frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}{\cdot}X^T{\cdot}X"/>

        The distribution will change with respect to &mu; (location of center) and &sum; (the shape of distribution)

        ![MultiGaussian_Sigma](../images/MultiGaussian_Simga.png)

        And p(x) is given by

        <img src="http://latex.codecogs.com/svg.latex?p(x;\mu,\sum)=\frac{1}{(2\pi)^{\frac{n}{2}}|\sum|^{\frac{1}{2}}}\textrm{exp}\big(-\frac{1}{2}(x-\mu)^T{\sum}^{-1}(x-\mu)\big)"/>

        If p(x) < &epsilon; flag an anomaly.

    * Original model is just a special case of multivariate Gaussian distribution

        <img src="http://latex.codecogs.com/svg.latex?\sum=\begin{bmatrix}\sigma_1^2&0&{\dots}&0\\0&\sigma_2^2&{\dots}&0\\\vdots&\vdots&\ddots&\vdots\\0&0&{\dots}&\sigma_n^2\end{bmatrix}"/>

    * Comparision

        Original Model|Multivariate Gaussian
        -|-
        Manually create features to capture anomalies when features take unusual combinations|Automatically captures correlations between features
        Computationally cheaper, n can go large (10,000 to 100,000)|Computationally more expensive because of the calculation of the reverse of &sum;
        Ok if m is small|Must have m > n, otherwise &sum; is non-invertible (may also be resulted from redundant features)
        
        Andrew's experience suggests when m > 10n, we could try Multivariate Gaussian

4. Recommender System        

    * Example: Predict Movie Rating, from 0 to 5 stars

        Movie|Alice(1)|Bob(2)|Carol(3)|Dave(4)
        -|-|-|-|-
        movie 1|5|5|0|0
        movie 2|5|?|?|?
        movie 3|?|4|0|4
        movie 4|0|0|5|?

        n<sub>u</sub> = no. of users

        n<sub>m</sub> = no. of movies

        r(i, j) = 1 if user j has rated movie i

        y<sup>(i, j)</sup> = rating given by user j to movie i (defined only if r(i, j) = 1)
    
    * Content Based Recommendations

        Build feature vectors x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup>

        For each user j, learn parameters &theta;<sup>(j)</sup> &isin; R<sup>n</sup> and predict user j's rating on movie i with (&theta;<sup>(j)</sup>)<sup>T</sup>&sdot;x<sup>(i)</sup> stars

        * Problem formulation

            To learn &theta;<sup>(j)</sup>

            &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\min_{\theta^{(j)}}\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^{n}(\theta^{(j)})^2"/>

            Get rid of m<sup>(j)</sup> and learn all &theta;:

            &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\min_{\theta^{(j)},\dots,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta^{(j)})^2"/>

            Gradient descent update:

            For k &ne; 0:

            &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\theta_k^{(j)}=\theta_k^{(j)}-\alpha\Big(\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)x_k^{(i)}+\lambda\theta_k^{(j)}\Big)"/>
            
            For k = 0:
            
            &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\theta_k^{(j)}=\theta_k^{(j)}-\alpha\Big(\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)x_k^{(i)}\Big)"/>

    * Collaborative Filtering

        Given x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup>, we can estimate &theta;<sup>(1)</sup>, ..., &theta;<sup>(n<sub>u</sub>)</sup>

        Given &theta;<sup>(1)</sup>, ..., &theta;<sup>(n<sub>u</sub>)</sup>, we can estimate x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup>

        We can actually optimize both simutaneously and cost function is
        
        <img src="http://latex.codecogs.com/svg.latex?J(x^{(1)},\dots,x^{(n_m)},\theta^{(1)},\dots,\theta^{(n_u)})=\frac{1}{2}\sum_{i,j:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta^{(j)})^2"/>

    * Algorithm

        **Initialize x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup> and &theta;<sup>(1)</sup>, ..., &theta;<sup>(n<sub>u</sub>)</sup> to small random values**

        **Minimize J(...) using gradient descent (or other methods):**

        &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?x_k^{(j)}=x_k^{(j)}-\alpha\Big(\sum_{j:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)\theta_k^{(j)}+\lambdax_k^{(i)}\Big)"/>

        &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\theta_k^{(j)}=\theta_k^{(j)}-\alpha\Big(\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T{\cdot}x^{(i)}-y^{(i,j)}\big)x_k^{(i)}+\lambda\theta_k^{(j)}\Big)"/>

        **For a user with parameter &theta; and a movie with feature x, pridict rating by &theta;<sup>T</sup>x**

    * Low Rank Matrix Factorization

        Define:

        &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}-&(x^{(1)})^T&-\\-&(x^{(2)})^T&-\\\vdots&\vdots&\vdots\\-&(x^{(n_m)})^T&-\end{bmatrix},"/>
        &nbsp;&nbsp;&nbsp;&nbsp;<img src="http://latex.codecogs.com/svg.latex?\Theta=\begin{bmatrix}-&(\theta^{(1)})^T&-\\-&(\theta^{(2)})^T&-\\\vdots&\vdots&\vdots\\-&(\theta^{(n_u)})^T&-\end{bmatrix}"/>

        Predicted rating = X&Theta;<sup>T</sup> &rarr; [Low Rank Matrix](https://en.wikipedia.org/wiki/Low-rank_approximation)

    * Finding related movies

        How to find movie j related to movie i?

        **if ||x<sup>(i)</sup>-x<sup>(j)</sup>|| is smaller &rarr; movie i and j are "similar"**

    * Mean normalilzation

        If we add a new user with no rating at all, then because of the regularization terms in our cost function, &theta; will be 0 for all features, which means &theta;<sup>T</sup>x is also 0, and all movies will assign 0 stars.
        
        To avoid such scenario, we first calculate the means for each movie and subtract each rating with the mean and then use the algorithm to learn. e.g.
    
        <img src="http://latex.codecogs.com/svg.latex?Y=\begin{bmatrix}5&5&0&0&?\\5&?&?&0&?\\?&4&0&?&?\\0&0&5&4&?\end{bmatrix}"/> &rarr;  <img src="http://latex.codecogs.com/svg.latex?\mu=\begin{bmatrix}2.5\\2.5\\2\\2.25\end{bmatrix}"/> &rarr;  <img src="http://latex.codecogs.com/svg.latex?Y=\begin{bmatrix}2.5&2.5&-2.5&-2.5&?\\2.5&?&?&-2.5&?\\?&?&2&-2&?\\-2.25&-2.25&2.75&1.75&?\end{bmatrix}">

        And for new user j that gave no rating, we will predict (&theta;<sup>(j)</sup>)<sup>T</sup>&sdot;x<sup>(i)</sup>+&mu;<sub>i</sub> = &mu;<sub>i</sub>
