## Week 5
1. Backpropagation

   * Cost function:
  
      <img src="https://latex.codecogs.com/svg.latex?J(\Theta)=-\frac{1}{m}\Bigg[\sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)}\textrm{log}h_{\Theta}(x^{i})_k+(1-y_k^{(i)})\textrm{log}(1-h_{\Theta}(x^{i})_k)\Bigg]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{S_l}\sum_{j=1}^{S_{l+1}}(\Theta_{ji}^{(l)})^2"/>

      where *L* is the total number of layers in the network
    
      *S<sub>l</sub>* is the number of units (without bias unit) in layer
      
      K is the number of output units/classes
   * Gradients
      
      <img src="https://latex.codecogs.com/svg.latex?\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_j^{(l)}\cdot\delta_i^{(l+1)}"/>
      
      For the output unit:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^{(L)}=a_j^{(L)}-y_j"/>
      
      For inner layers:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^{(l)}=(\Theta^{(l)})^T\cdot\delta^{(l+1)}.*g'(z^l)=(\Theta^{(l)})^T\cdot\delta^{(l+1)}.*a^{(l)}.*(1-a^{(l)})"/>

   
   * Algorithm:
   
      Training set {(x<sup>(1)</sup>, x<sup>(1)</sup>), ..., (x<sup>(m)</sup>, x<sup>(m)</sup>)}
      
      Set &Delta;<sub>ij</sub><sup>(l)</sup> = 0 for all i, j
      
      For i = 1 to m &larr; (x<sup>(i)</sup>, x<sup>(i)</sup>):
      
      &nbsp;&nbsp;&nbsp;&nbsp;a<sup>(1)</sup> = x<sup>(i)</sup>
      
      &nbsp;&nbsp;&nbsp;&nbsp;Forward propagation to compute a<sup>(l)</sup> for l = 2, ..., L
      
      &nbsp;&nbsp;&nbsp;&nbsp;Using y<sup>(i)</sup> to compute &delta;<sup>(L)</sup> = a<sup>(L)</sup>-y<sup>(i)</sup>
      
      &nbsp;&nbsp;&nbsp;&nbsp;Compute &delta;<sup>(L-1)</sup>, ..., &delta;<sup>(2)</sup>
      
      &nbsp;&nbsp;&nbsp;&nbsp;&Delta;<sub>ij</sub><sup>(l)</sup> = &Delta;<sub>ij</sub><sup>(l)</sup>+a<sub>j</sub><sup>(l)</sup>&delta;<sup>(l+1)</sup>, can be vectorized as &Delta;<sup>(l)</sup> = &Delta;<sup>(l)</sup>+&delta;<sup>(l+1)</sup>&sdot;(a<sup>(l)</sup>)<sup>T</sup>
      
      D<sub>ij</sub><sup>(l)</sup> = <sup>1</sup>&frasl;<sub>m</sub>&sdot;&Delta;<sub>ij</sub><sup>(l)</sup>+&lambda;&Theta;<sub>ij</sub><sup>(l)</sup>, if j &ne; 0
      
      D<sub>ij</sub><sup>(l)</sup> = <sup>1</sup>&frasl;<sub>m</sub>&sdot;&Delta;<sub>ij</sub><sup>(l)</sup>, if j = 0
 
   * Mathematical prove:
      
      a. First we define
      
      <img src="https://latex.codecogs.com/svg.latex?Cost(i)=y^{(i)}\textrm{log}h_{\Theta}(x^{i})_k+(1-y^{(i)})\textrm{log}(1-h_{\Theta}(x^{i}))"/>
      
      
      
      b. For any inner layer, we have:
      
      <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}J(\Theta)}{\partial\Theta_{ij}^l}=\frac{{\partial}J(\Theta)}{{\partial}z_i^{l+1}}\cdot\frac{{\partial}z_i^(l+1)}{\partial\Theta_{ij}^l}=\frac{{\partial}J(\Theta)}{{\partial}z_i^{l+1}}{\cdot}a_j^l"/>
      
      So we define a matrix &delta; for each layer and
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^l=\frac{{\partial}J(\Theta)}{{\partial}z_i^{l}}"/>
      
      And we can derive recursively for each &delta;:
            
      <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}J(\Theta)}{{\partial}z_i^{l}}=\sum_{k=0}^{S_{l+1}}\frac{{\partial}J(\Theta)}{{\partial}z_k^{l+1}}\cdot\frac{{\partial}z_k^{l+1}}{{\partial}z_i^{l}}=\sum_{k=0}^{S_{l+1}}\frac{{\partial}J(\Theta)}{{\partial}z_k^{l+1}}\cdot\frac{{\partial}(\Theta_k^l{\cdot}a^l)}{{\partial}a_i^l}\cdot\frac{{\partial}a_i^l}{{\partial}z_i^{l}}"/>
      
      The first term in the summation equation is just *&delta;<sub>i</sub><sup>l+1</sup>*
      
      And the second term in this equation can be derived as:
      
      <img src="https://latex.codecogs.com/svg.latex?\because\frac{{\partial}(\Theta_k^l{\cdot}a^l)}{{\partial}a_i^l}=\frac{{\partial}(\cdots+\Theta_{ki}^l{\cdot}a_i^l+\cdots)}{{\partial}a_i^l}=\Theta_{ki}^l"/>
      
      And the last term is:
      
      <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}a_i^l}{{\partial}z_i^{l}}=g'(z_i^l)=a_i^l(1-a_i^l)"/>
      
      And put it together:
      
      <img src="https://latex.codecogs.com/svg.latex?\delta_i^l=\sum_{k=0}^{S_{l+1}}\delta_k^{l+1}{\cdot}\theta_{ki}^l{\cdot}g'(z_i^l)=((\theta^T)_i\cdot\delta^{l+1})g'(z_i^l)"/>
      
      And we can vectorize it as
      
      <img src="https://latex.codecogs.com/svg.latex?\delta^l=(\theta^T\cdot\delta^{l+1}).*g'(z^l)"/>
