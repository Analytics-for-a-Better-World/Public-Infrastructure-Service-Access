### Model `maximal covering` as in the article by [Church and ReVelle](https://www.semanticscholar.org/paper/The-maximal-covering-location-problem-Church-Revelle/c3de804bbeb15b0d8570ee3d9f4cbdf432993cfa)

This model defines variables $z_i$ for each household $i\in I$ to indicate if that household can be served by a hospital that is opened at $j \in J$, leading to the complete model as follows:

https://latex.codecogs.com/svg.image?%5Cbegin%7Balign*%7D%5Cmax%5Cquad&%5Csum_%7Bi%5Cin%20I%7Dw_iz_i%5C%5C%5Ctext%7Bsubject%20to:%7D%5Cquad&z_i%5Cleq%5Csum_%7Bj%5Cin%20J_i%7Dx_j&%5Cforall%20i%5Cin%20I%5C%5C&%5Csum_%7Bj%5Cin%20J%7Dx_j%5Cleq%20p%5C%5C&x_j%5Cin%5C%7B0,1%5C%7D&%5Cforall%20j%5Cin%20J%5C%5C&z_i%5Cin%5C%7B0,1%5C%7D&%5Cforall%20i%5Cin%20I%5Cend%7Balign*%7D

The first line states the objective as to maximize the total headcount of the households served, while the second line (after _subject to:_) lists the first constraint: each household is only served if at least one hospital within reach is open. 
Then the number of hospitals to open constraints the selection and finally the binary nature of the variables used is specified. 

The model above selects up to $p$ hospitals. In the original paper Church and ReVelle selected exactly $p$ hospitals, but our model has advantages to be discussed later.
