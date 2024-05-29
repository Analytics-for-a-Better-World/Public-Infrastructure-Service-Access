### Model `maximal covering` as in the article by [Church and ReVelle](https://www.semanticscholar.org/paper/The-maximal-covering-location-problem-Church-Revelle/c3de804bbeb15b0d8570ee3d9f4cbdf432993cfa)

This model defines variables $z_i$ for each household $i\in I$ to indicate if that household can be served by a hospital that is opened at $j \in J$, leading to the complete model as follows:

\begin{align}
    \max\quad & \sum_{i\in I} w_iz_i  \\
    \text{subject to:}\quad & z_i \leq \sum_{j\in J_i } x_j & \forall i \in I \\
    & \sum_{j \in J} x_j \leq p \\
    & x_j \in \{0,1\} & \forall j \in J \\
    & z_i \in \{0,1\} & \forall i \in I
\end{align}

The first line states the objective as to maximize the total headcount of the households served, while the second line (after _subject to:_) lists the first constraint: each household is only served if at least one hospital within reach is open. 
Then the number of hospitals to open constraints the selection and finally the binary nature of the variables used is specified. 

The model above selects up to $p$ hospitals. In the original paper Church and ReVelle selected exactly $p$ hospitals, but our model has advantages to be discussed later.
