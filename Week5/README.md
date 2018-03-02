## Week 5
1. Backpropagation

   * Cost function:
  
      <img src="https://latex.codecogs.com/svg.latex?J(\Theta)=-\frac{1}{m}\Bigg[\sum_{t=1}^{m}\sum_{k=1}^{K}y_k^{(t)}\textrm{log}h_{\Theta}(x)_k^{(t)}+(1-y_k^{(t)})\textrm{log}(1-h_{\Theta}(x)_k^{(t)})\Bigg]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{S_l}\sum_{j=1}^{S_{l+1}}(\Theta_{ji}^{l})^2"/>

      where *L* is the total number of layers in the network
    
      *S<sub>l</sub>* is the number of units (without bias unit) in layer
      
      K is the number of output units/classes
   * Gradients
      
      <img src="https://latex.codecogs.com/svg.latex?\frac{\partial}{\partial\Theta_{ij}^{l}}J(\Theta)=\frac{1}{m}\sum_{t=1}^{m}(a_j^{l})^{(t)}\cdot(\delta_i^{l+1})^{(t)}+\frac{\lambda}{m}\Theta_{ij}^{l}"/>
      
      For the output unit:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^{L}=a_j^{L}-y_j"/>
      
      For inner layers:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^{l}=(\Theta^{l})^T\cdot\delta^{l+1}.*g'(z^l)=(\Theta^{l})^T\cdot\delta^{l+1}.*a^{l}.*(1-a^{l})"/>

   
   * Algorithm:
   
      Training set {(x<sup>(1)</sup>, x<sup>(1)</sup>), ..., (x<sup>(m)</sup>, x<sup>(m)</sup>)}
      
      Set &Delta;<sub>ij</sub><sup>l</sup> = 0 for all i, j
      
      For i = 1 to m &larr; (x<sup>(i)</sup>, x<sup>(i)</sup>):
      
      &nbsp;&nbsp;&nbsp;&nbsp;a<sup>(1)</sup> = x<sup>(i)</sup>
      
      &nbsp;&nbsp;&nbsp;&nbsp;Forward propagation to compute a<sup>l</sup> for l = 2, ..., L
      
      &nbsp;&nbsp;&nbsp;&nbsp;Using y<sup>(i)</sup> to compute &delta;<sup>L</sup> = a<sup>L</sup>-y<sup>(i)</sup>
      
      &nbsp;&nbsp;&nbsp;&nbsp;Compute &delta;<sup>L-1</sup>, ..., &delta;<sup>(2)</sup>
      
      &nbsp;&nbsp;&nbsp;&nbsp;&Delta;<sub>ij</sub><sup>l</sup> = &Delta;<sub>ij</sub><sup>l</sup>+a<sub>j</sub><sup>l</sup>&delta;<sup>l+1</sup>, can be vectorized as &Delta;<sup>l</sup> = &Delta;<sup>l</sup>+&delta;<sup>l+1</sup>&sdot;(a<sup>l</sup>)<sup>T</sup>
      
      D<sub>ij</sub><sup>l</sup> = <sup>1</sup>&frasl;<sub>m</sub>&sdot;&Delta;<sub>ij</sub><sup>l</sup>+<sup>&lambda;</sup>&frasl;<sub>m</sub>&Theta;<sub>ij</sub><sup>l</sup>, if j &ne; 0
      
      D<sub>ij</sub><sup>l</sup> = <sup>1</sup>&frasl;<sub>m</sub>&sdot;&Delta;<sub>ij</sub><sup>l</sup>, if j = 0
 
   * Mathematical prove:
   
      First we define the cost of test t without regularization:
      
      <img src="https://latex.codecogs.com/svg.latex?Cost(\Theta,t)=-\sum_{k=1}^{K}\Big(y_k^{(t)}\textrm{log}h_{\Theta}(x)_k^{(t)}+(1-y_k^{(t)})\textrm{log}(1-h_{\Theta}(x)_k^{(t)})\Big)"/>
      
      Then the gradients can be expressed as:
      
      <img src="https://latex.codecogs.com/svg.latex?J(\Theta)=-\frac{1}{m}\sum_{t=1}^{m}Cost(\Theta,t)+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{S_l}\sum_{j=1}^{S_{l+1}}(\Theta_{ji}^{l})^2"/>
      
      For each test t, we will have: (Use *Cost(&Theta;)* for convinience)
        
      1. *For any inner layer, we have:*
      
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}Cost(\Theta)}{\partial\Theta_{ij}^l}=\frac{{\partial}Cost(\Theta)}{{\partial}z_i^{l+1}}\cdot\frac{{\partial}z_i^{l+1}}{\partial\Theta_{ij}^l}=\frac{{\partial}Cost(\Theta)}{{\partial}z_i^{l+1}}{\cdot}a_j^l"/>
      
         So we define a matrix &delta; for each inner layers and
      
         <img src="https://latex.codecogs.com/svg.latex?\delta_i^l=\frac{{\partial}Cost(\Theta)}{{\partial}z_i^{l}}"/>
      
         Then we have
      
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}Cost(\Theta)}{\partial\Theta_{ij}^l}=\delta_i^{l+1}{\cdot}a_j^l"/>
      
      2. *&delta; can be derived recursively from the next layer:*
            
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}Cost(\Theta)}{{\partial}z_i^{l}}=\sum_{k=0}^{S_{l+1}}\frac{{\partial}Cost(\Theta)}{{\partial}z_k^{l+1}}\cdot\frac{{\partial}z_k^{l+1}}{{\partial}z_i^{l}}=\sum_{k=0}^{S_{l+1}}\frac{{\partial}Cost(\Theta)}{{\partial}z_k^{l+1}}\cdot\frac{{\partial}(\Theta_k^l{\cdot}a^l)}{{\partial}a_i^l}\cdot\frac{{\partial}a_i^l}{{\partial}z_i^{l}}"/>
      
         The first term in the summation equation is just *&delta;<sub>i</sub><sup>l+1</sup>*, and the second term in this equation can be derived as:
      
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}(\Theta_k^l{\cdot}a^l)}{{\partial}a_i^l}=\frac{{\partial}(\cdots+\Theta_{ki}^l{\cdot}a_i^l+\cdots)}{{\partial}a_i^l}=\Theta_{ki}^l"/>
      
         And the last term is:
      
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}a_i^l}{{\partial}z_i^{l}}=g'(z_i^l)=a_i^l(1-a_i^l)"/>
      
         Put them together:
      
         <img src="https://latex.codecogs.com/svg.latex?\delta_i^l=\sum_{k=0}^{S_{l+1}}\delta_k^{l+1}{\cdot}\theta_{ki}^l{\cdot}g'(z_i^l)=((\theta^T)_i\cdot\delta^{l+1})g'(z_i^l)"/>
      
         Then, we can vectorize it as
      
         <img src="https://latex.codecogs.com/svg.latex?\delta^l=(\theta^T\cdot\delta^{l+1}).*g'(z^l)"/>
      
      3. *For the output layer, we have:*
      
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}Cost(\Theta)}{{\partial}z_i^{L}}=-\frac{{\partial}\big(y_i\textrm{log}h_{\Theta}(x)_i+(1-y_i)\textrm{log}(1-h_{\Theta}(x)_i)\big)}{{\partial}z_i^{L}}\\=-\frac{{\partial}\big(y_i\textrm{log}h_{\Theta}(x)_i+(1-y_i)\textrm{log}(1-h_{\Theta}(x)_i)\big)}{{\partial}h_{\Theta}(x^{L})_i}\frac{{\partial}h_{\Theta}(x^{L})_i}{{\partial}z_i^{L}}\\=-\Big(\frac{y_i}{h_{\Theta}(x^{L})_i}-\frac{1-y_i}{1-h_{\Theta}(x^L)_i}\Big)g'(z_i^L)\\=-\frac{y_i-h_{\Theta}(x^{L})_i}{h_{\Theta}(x^{L})_i(1-h_{\Theta}(x^{L})_i)}g'(z_i^L)"/>
      
         Since we have
      
         <img src="https://latex.codecogs.com/svg.latex?g'(z_i^L)=h_{\Theta}(x^{L})_i(1-h_{\Theta}(x^{L})_i)"/>
      
         We can derive that
      
         <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}Cost(\Theta)}{{\partial}z_i^{L}}=h_{\Theta}(x^{L})_i-y_i"/>
      
         Vectorize it, and we have:
      
         <img src="https://latex.codecogs.com/svg.latex?\delta^L=h_{\Theta}(x^{L})-y=a^L-y"/>
      
      * **Combine a, b, c together, we now see that the gradients in the algorithm are correct.**
         
         <img src="https://latex.codecogs.com/svg.latex?\frac{\partial}{\partial\Theta_{ij}^{l}}J(\Theta)=\frac{1}{m}\sum_{t=1}^{m}\frac{{\partial}Cost(\Theta,t)}{\partial\Theta_{ij}^{l}}+\frac{\lambda}{m}\Theta_{ij}^{l}=\frac{1}{m}\sum_{t=1}^{m}(a_j^{l})^{(t)}\cdot(\delta_i^{l+1})^{(t)}+\frac{\lambda}{m}\Theta_{ij}^{l}"/>

   * Backpropagation intuition

      ![BackPropIntuition](../images/BackPropIntuition.jpg)

   * Unrolling Parameters

      Unroll the &Theta; matrices into the just one matrix and recover after backpropogation.

   * Gradient checking

       <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}J(\Theta)}{\Theta_{ij}}\approx\frac{J(...,\Theta_{ij}+\epsilon,...)-J(...,\Theta_{ij}-\epsilon,...)}{2\epsilon}"/>
    
   * Initial value of &Theta;
      * Zero initialization or initialized with same value for each layer: after each update, each unit in the same layer will have identical parameters
      * Random initialization: symmetry breaking.
         
         Initialize each &Theta;<sub>ij</sub><sup>l</sup> to a random value in [-&epsilon;, &epsilon;]

         One effective strategy for choosing &epsilon; is to base it on the number of units in the network. A good choice of &epsilon; is 

         <img src="https://latex.codecogs.com/svg.latex?\epsilon=\frac{\sqrt{6}}{\sqrt{L_{in}+L_{out}}}"/>

         Where L<sub>In</sub> = S<sub>l</sub> and S<sub>Out</sub> = S<sub>l+1</sub>

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
