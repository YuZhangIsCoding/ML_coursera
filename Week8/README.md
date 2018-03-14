## Week 8
1. Clustering
    * Unsupervised learning

        * Applications: *market segmentation*, *social network analysis*, *organize computer cluster*, *astronomical data analysis*

        ![UnsupervisedLearning](../images/UnsupervisedLearning.jpg)

    * K-means algorithm
        
        ![Kmeans](../images/Kmeans.jpg)

        Input: K (number of clusters), Training set {x<sup>(1)</sup>, ..., x<sup>(m)</sup>}, where x<sup>(i)</sup> &isin; R<sup>n</sup>(drop x<sub>0</sub> = 1 as convention)

        **Randomly initialize K cluster centroids &mu;<sub>1</sub>, ..., &mu;<sub>K</sub>**

        **Repeat{**

        **for i = 1 to m**
        
        &nbsp;&nbsp;&nbsp;&nbsp;**C<sup>(i)</sup> = index (from 1 to K) of cluster cnetroid closest to x<sup>(i)</sup>** &larr; *Cluster assignment step*

        **for k = 1 to K**

        &nbsp;&nbsp;&nbsp;&nbsp;**&mu;<sub>k</sub> = average of points assigned to cluster k** &larr; *Move centroid step*

        **}**


        * *Note: if no sample assigned to a centroid, we can*

            1. *Delete that centroid, and decrease the number of centroids to K-1 (more common).*
    
            2. *Randomly reinitialize the centroid if need K clusters.*


    * K-means for non-separated clusters

        ![Kmean_NonSep](../images/Kmean_NonSep.jpg)

        Even though in some cases, the date seems not separated, we can still use clustering to group them. In the case above, we divide the points into 3 clusters by their weight and height.

    * Optimization objective

        Define &mu;<sub>c<sup>(i)</sup></sub> = cluster centroid of cluster to which example x<sup>(i)</sup> has been assigned, e.g. x<sup>(i)</sup> &rarr; 5, c<sup>(i)</sup> = 5, &mu;<sub>c<sup>(i)</sup></sub> = &mu;<sub>5</sub>

        Define our distortion cost function as :

        <img src="https://latex.codecogs.com/svg.latex?J(c^{(1)},...,c^{(m)},\mu_1,...,\mu_K)=\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-\mu_{c^{(i)}}||^2"/>

    * Revisit the algorithm
        
        * The cluster assignment step is to minimize J(...) with respect to C<sup>(1)</sup>, ..., C<sup>(m)</sup>, while holding &mu;<sub>1</sub>, ..., &mu;<sub>K</sub> fixed
        * The move centroid step is to minimize J(...) with respect to &mu;<sub>1</sub>, ..., &mu;<sub>K</sub>

    * Random Initialization
       
        Randomly pick K (K < m) traning examples and set &mu;<sub>1</sub>, ..., &mu;<sub>K</sub> to these K examples. 

    * Local optima: try initializing several different times

        ![Kmeans_LocalOpt](../images/Kmeans_LocalOpt.jpg)
        
        for i = 1 to test_numbers (50-1000 is reasonable){

        &nbsp;&nbsp;&nbsp;&nbsp;*Randomly initialize K centroids*

        &nbsp;&nbsp;&nbsp;&nbsp;*Run k-means, get C<sup>(1)</sup>, ..., C<sup>(m)</sup>, &mu;<sub>1</sub>, ..., &mu;<sub>K</sub>*

        &nbsp;&nbsp;&nbsp;&nbsp;*Compute cost function J(...)*

        }

        **Pick clustering that gives lowest cost function J(...)**

        * K = 2-10, random initialization works pretty well

        * K >> 10, just slight improve after random initialization

    * Choosing the number of clusters: 

        * Mainly by hand, human judgement
            
            ![Cluster_manual](../images/Cluster_manual.jpg)

        * Elbow method: plot cost function vs. K

            ![Elbow_method](../images/Elbow_Method.jpg)

            "Worth a shot, but won't have a high expectation", because many times, there are no clear elbow.

        * Later/downstream purpose

            e.g. T-shirt sizing: K = 3 &rarr; S, M, L; K = 5 &rarr; XS, S, M, L, XL

2. Dimensionality Reduction
    * Motivation
        * Data compression
            
            ![DataCompression](../images/DataCompression.jpg)

            Reduce data from 2D to 1D if datas fall near a line, or reduce data from 3D to 2D if data falls near a plane, etc.

        * Data visualization

            Suppose we have a table like this:

            Country|GDP (x<sub>1</sub>)| Per capita GDP (x<sub>2</sub>)|...|x<sub>50</sub>
            --|--|--|--|--
            Canada|...|...|...|...
            China|...|...|...|...
            ...|...|...|...|...

            We can reduce x &isin; R<sup>50</sup> &rarr; z &isin; R<sup>2</sup>

            ![DataVisualization](../images/DataVisualizatioin.jpg)
    * Principle Component Analysis problem formulation

        Reduce from n-dimension to K-dimension:

        Find K vectors u<sup>(1)</sup>, ..., u<sup>(K)</sup> onto which to project the data, so as to minimize the projection error.

    * PCA is not linear regression

        **Linear regression minimizes vetical distances, while PCA minimizes orthogonal distances, where data is not labeled and treated equally**

        ![PCAnotLinearReg](../images/PCAnotLinearReg.jpg)

    * PCA algorithm

        *Preprocessing, feature scaling*
        
        Reduce data from n-d to K-d

        *Compute ["Covariance matrix"](https://en.wikipedia.org/wiki/Covariance_matrix), which is a way to represent the linear relationship between variables:**
        
        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\sum=\sum_{i=1}^{n}x^{(i)}(x^{(i)})^T"/>

        *Compute "eigenvectors" of matrix &sum; using [Singular-value decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition)*
            
        &nbsp;&nbsp;&nbsp;&nbsp;[U, S, V] = svd(&sum;)

        *Select first eigenvectors as U<sub>reduce</sub> = U(:, 1:K), and compute the projection of x to z by inner product:
        &nbsp;&nbsp;&nbsp;&nbsp;z = (U<sub>reduce</sub>)<sup>T</sup>&sdot;x, where z &isin; R<sup>K<sup>*

    * Vectorized implementation and summary
        
        **After mean normalization and feature scaling**

        **&sum; = <sup>1</sup>&frasl;<sub>m</sub>&sdot;X<sup>T</sup>&sdot;X**

        **[U, S, V] = svd(&sum;)**

        **U<sub>reduce</sub> = U(:, 1:K)**

        **Z = X&sdot;U<sub>reduce</sub>**

    * Reconstruction from compressed representation

        x<sub>approx</sub> = U<sub>reduce</sub>&sdot;z 

        Vectorized: **X<sub>approx</sub> = Z&sdot;U<sub>reduce</sub>**

    * Choose the number of pricinple components

        Average squared projection error:

        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-x_{approx}^{(i)}||^2"/>

        Total variation in the data:
    
        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2"/>

        Typically, choose K to be the smallest value, so that

        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\frac{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-x_{approx}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2}\le0.01"/>

        or **"99% of variance is retained"**

    * Choose K algorithm:
        
        **Try PCA with K = 1, 2, ...**

        **Compute [U, S, V] = svd(&sum;)**

        **Pick the smallest K that satisfies:**
            
        &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\frac{\sum_{i=1}^{K}S_{ii}}{\sum_{i=1}^{n}S_{ii}}\ge0.99"/>

    * Application of PCA:
        
        * Speed up supervised learning

            *Note: mapping x<sup>(i)</sup> &rarr; z<sup>(i)</sup> should be defined by training PCA only on the training set. This mapping can be applied to x<sub>cv</sub><sup>(i)</sup> and x<sub>test</sub><sup>(i)</sup>*

        * Reduce memory/disk needed to store data
        * Visualization: K = 2 or 3

    * Bad use of PCA:

        * Prevent overfitting: it might work ok, but isn't a good because PCA will throw away some data without knowing y. Use regularization instead.
        * Blindly use PCA in ML system: before implementing PCA, first try to run with original data. Only if that doesn't do what you want, then consider implementing PCA.
