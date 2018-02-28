## Week 5
1. Backpropagation

  * Prove that 
  
  <img src="https://latex.codecogs.com/svg.latex?\delta^l=\theta^T\delta^{l+1}.*g'(z^l)"/>
  
  <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}J(\theta)}{{\partial}z_i^{l}}=\sum_{k=0}^{S_{l+1}}\frac{{\partial}J(\theta)}{{\partial}z_k^{l+1}}\cdot\frac{{\partial}z_k^{l+1}}{{\partial}z_i^{l}}=\sum_{k=0}^{S_{l+1}}\frac{{\partial}J(\theta)}{{\partial}z_k^{l+1}}\cdot\frac{{\partial}(\theta_k^l{\cdot}a^l)}{{\partial}a_i^l}\cdot\frac{{\partial}a_i^l}{{\partial}z_i^{l}}"/>
  
  <img src="https://latex.codecogs.com/svg.latex?\because\frac{{\partial}(\theta_k^l{\cdot}a^l)}{{\partial}a_i^l}=\frac{{\partial}(\cdots+\theta_{ki}^l{\cdot}a_i^l+\cdots)}{{\partial}a_i^l}=\theta_{ki}^l"/>
  
  <img src="https://latex.codecogs.com/svg.latex?\frac{{\partial}a_i^l}{{\partial}z_i^{l}}=g'(z_i^l)=a_i^l(1-a_i^l)"/>
  
  let 
    
  <img src="https://latex.codecogs.com/svg.latex?\delta_i^l=\frac{{\partial}J(\theta)}{{\partial}z_i^{l}}"/>

  then
  
  <img src="https://latex.codecogs.com/svg.latex?\delta_i^l=\sum_{k=0}^{S_{l+1}}\delta_k^{l+1}{\cdot}\theta_{ki}^l{\cdot}g'(z_i^l)=((\theta^T)_i\cdot\delta^{l+1})g'(z_i^l)"/>
  
  <img src="https://latex.codecogs.com/svg.latex?\therefore\delta^l=(\theta^T\cdot\delta^{l+1}).*g'(z^l)"/>
