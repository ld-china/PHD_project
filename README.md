
# parameters:

MiniSom(x=6, y=6, input_len=27, sigma=1, learning_rate=0.5, neighborhood_function='gaussian', activation_distance=activation_distance[i], random_seed=0)
### ps: activation_distance = ['euclidean', 'cosine', 'manhattan', 'chebyshev']

# result
(1) activation_distance: euclidean
Execution time: 0.08000016212463379 s
Quantization error: 0.2713008580420594
Model Accuracy Score:  0.886

(2) activation_distance: cosine
Execution time: 0.08999991416931152 s
Quantization error: 0.28798409570396005
Model Accuracy Score:  0.909

(3) activation_distance: manhattan
Execution time: 0.04999995231628418 s
Quantization error: 0.292011698366118
Model Accuracy Score:  0.818

(4) activation_distance: chebyshev
Execution time: 0.039999961853027344 s
Quantization error: 0.40794503784810554
Model Accuracy Score:  0.795

# figure
![distance matrix](https://user-images.githubusercontent.com/65076718/172353657-42840d72-874e-46ef-8a52-974e401f4643.png)

in the picture:  class 1 - Red circle, class 2 - yellow square, class 3 - blue diamond
