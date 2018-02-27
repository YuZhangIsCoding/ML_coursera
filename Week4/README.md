## Week 4
1. Non-linear classification
    * Previously we've successfully used logistic regression for classifications, but often times, we may have many features in our hypothesis. E.g. suppose we have x<sub>1</sub>, ..., x<sub>n</sub> features.
        * Just including the quadratic terms (x<sub>1</sub><sup>2</sup>, x<sub>1</sub>x<sub>2</sub>, ...) will have complexity of O(n<sup>2</sup>) &asymp; n<sup>2</sup>&frasl;2
        * Adding the cubic terms will make it O(n<sup>3</sup>) ...
        * Including all these my end up overfitting, and computationally expensive.
        * If we use less features, like just using x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, ... may not represent well.
    * Computer vision: Car detection
        * Pick two blocks of pixels and input learning algorithms, so that we can detect cars from new input.
        * Assume that a block has 50&times;50 pixels, then totally there will be 2500 pixels (7500 RGB)
        * Then if we just the quadratic features for logistic regression will need &asymp; 3 million features

2. Neural networks: algorithms that try to mimic the brain
    * The "One learning algorithm" hypothesis
    
        From neuron science, researches have shown that
        1. Auditory cortex can learn to see if cut off audio input and replace with visual input.[<sup>1</sup>](https://www.nature.com/articles/35009102)
        2. Somatosensory cortex learns to see.
    * Model representation
        * A [neuron](https://en.wikipedia.org/wiki/Neuron) is composed of nucleus, dendrites, axons, etc., where dendrites and axons serves as the "input wires" and "output wires", respectively.
        
            ![neuron_from_wikipedia](../images/Neuron.png)
        * Logistic unit can be built similarily on this neuron model:
           
            ![NeuronModel](../images/NeuronModel.jpg) 
        * Neural networks:

            ![NeuralNetworks](../images/NeuralNetworks.jpg)
            
            a<sub>1</sub><sup>(2)</sup> = g(&theta;<sub>10</sub><sup>(1)</sup>x<sub>0</sub>+&theta;<sub>11</sub><sup>(1)</sup>x<sub>1</sub>+&theta;<sub>12</sub><sup>(1)</sup>x<sub>2</sub>+&theta;<sub>13</sub><sup>(1)</sup>x<sub>3</sub>)

            a<sub>2</sub><sup>(2)</sup> = g(&theta;<sub>20</sub><sup>(1)</sup>x<sub>0</sub>+&theta;<sub>21</sub><sup>(1)</sup>x<sub>1</sub>+&theta;<sub>22</sub><sup>(1)</sup>x<sub>2</sub>+&theta;<sub>23</sub><sup>(1)</sup>x<sub>3</sub>)

            a<sub>3</sub><sup>(2)</sup> = g(&theta;<sub>30</sub><sup>(1)</sup>x<sub>0</sub>+&theta;<sub>31</sub><sup>(1)</sup>x<sub>1</sub>+&theta;<sub>32</sub><sup>(1)</sup>x<sub>2</sub>+&theta;<sub>33</sub><sup>(1)</sup>x<sub>3</sub>)
            
            If network has s<sub>j</sub> units in layer j, s<sub>j+1</sub> units in layer j+1, then &theta;<sup>(j)</sup> will have the dimension of s<sub>j+1</sub>&times;(s<sub>j</sub>+1).
    
            Output nodes will not include the bias nodes while the inputs will.
        * Mathematical representation: just like logistic regressions, but do it on every layer
            
            a<sub>1</sub><sup>(2)</sup> = g(z<sub>1</sub><sup>(2)</sup>)

            a<sub>2</sub><sup>(2)</sup> = g(z<sub>2</sub><sup>(2)</sup>)

            a<sub>3</sub><sup>(2)</sup> = g(z<sub>3</sub><sup>(2)</sup>)
        * Forward propagation: vectorized implementation
            
            z<sup>(2)</sup> = &Theta;<sup>(1)</sup>x

            a<sup>(2)</sup> = g(z<sup>(2)</sup>)

            Add a<sub>0</sub><sup>(2)</sup> = 1 &rarr; a<sup>(2)</sup> at this layer has n+1 features

            z<sup>(3)</sup> = &Theta;<sup>(2)</sup>a<sup>(2)</sup>

            ...
    * Examples: logic gate

        x<sub>1</sub>, x<sub>2</sub> &isin; {0, 1}, we have following logic operations:
        * AND
            
            ![AND](../images/AND.jpg)
            
            h<sub>&theta;</sub>(x) = g(-30+20x<sub>1</sub>+20x<sub>2</sub>)

            x<sub>1</sub>|x<sub>2</sub>|h<sub>&theta;</sub>(x)
            --|--|--
            0|0|g(-30)&asymp;0
            0|1|g(-10)&asymp;0
            1|0|g(-10)&asymp;0
            1|1|g(10)&asymp;1

        * OR
        
            ![OR](../images/OR.jgp)
            
            h<sub>&theta;</sub>(x) = g(-10+20x<sub>1</sub>+20x<sub>2</sub>)

            x<sub>1</sub>|x<sub>2</sub>|h<sub>&theta;</sub>(x)
            --|--|--
            0|0|g(-10)&asymp;0
            0|1|g(10)&asymp;1
            1|0|g(10)&asymp;1
            1|1|g(10)&asymp;1

        * NOT
            
            ![NOT](../images/NOT.jpg)
            
            h<sub>&theta;</sub>(x) = g(10-20x<sub>1</sub>)

            x<sub>1</sub>|h<sub>&theta;</sub>(x)
            --|--
            0|g(10)&asymp;1
            1|g(-10)&asymp;0

        * (NOT x<sub>1</sub>) AND (NOT x<sub>2</sub>)

            ![NotAandNotB](../images/NotAandNotB.jpg)
            
            h<sub>&theta;</sub>(x) = g(10-20x<sub>1</sub>-20x<sub>2</sub>)

            x<sub>1</sub>|x<sub>2</sub>|h<sub>&theta;</sub>(x)
            --|--|--
            0|0|g(10)&asymp;1
            0|1|g(-10)&asymp;0
            1|0|g(-10)&asymp;0
            1|1|g(-30)&asymp;0

        * XNOR (equals (x<sub>1</sub> AND x<sub>2</sub>) OR ((NOT x<sub>1</sub>) AND (NOT x<sub>2</sub>))
        
            ![XNOR](../images/XNOR.jpg)
            
            a<sub>1</sub><sup>(2)</sup> = g(-30+20x<sub>1</sub>+20x<sub>2</sub>)

            a<sub>2</sub><sup>(2)</sup> = g(10-20x<sub>1</sub>-20x<sub>2</sub>)

            h<sub>&theta;</sub>(x) = g(-10+20a<sub>1</sub><sup>(2)</sup>+20a<sub>2</sub><sup>(2)</sup>)

            x<sub>1</sub>|x<sub>2</sub>|a<sub>1</sub><sup>(2)</sup>|a<sub>2</sub><sup>(2)</sup>|h<sub>&theta;</sub>(x)
            --|--|--|--|--
            0|0|g(-30)&asymp;0|g(10)&asymp;1|g(10)&asymp;1
            0|1|g(-10)&asymp;0|g(-10)&asymp;0|g(-10)&asymp;0
            1|0|g(-10)&asymp;0|g(-10)&asymp;0|g(-10)&asymp;0
            1|1|g(10)&asymp;1|g(-30)&asymp;0|g(10)&asymp;1
            
        * XOR (equals (x<sub>1</sub> AND (NOT x<sub>2</sub>)) OR ((NOT x<sub>1</sub>) AND x<sub>2</sub>), another form (x<sub>1</sub> OR x<sub>2</sub>) AND ((NOT x<sub>1</sub>) OR (NOT x<sub>2</sub>)))
            
            Build based on XNOR:
        
            ![XOR](../images/XOR.jpg)
        
            a<sub>1</sub><sup>(2)</sup> = g(-30+20x<sub>1</sub>+20x<sub>2</sub>)
            
            a<sub>2</sub><sup>(2)</sup> = g(10-20x<sub>1</sub>-20x<sub>2</sub>)
            
            a<sub>1</sub><sup>(3)</sup> = g(-10+20a<sub>1</sub><sup>(2)</sup>+20a<sub>2</sub><sup>(2)</sup>)
            
            h<sub>&theta;</sub>(x) =  g(10-20a<sub>1</sub><sup>(3)</sup>)
    * Multiclass classification
        * Multiple output units: one-vs-all
        * Instead of output as discrete values such as y &isin; {1, 2, 3, ...}, we have y &isin; [[1;0;0;0],[0;1;0;0],[0;0;1;0], ...]
