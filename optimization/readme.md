# Mathematical Optimization

We will now employ _Mathematical Optimization_ to determine the optimal subset of hospitals to open. 
For those unfamiliar with _Mathematical Optimization_, we recommend starting with a hands-on introduction available in [this Jupyter book](https://mobook.github.io/MO-book/intro.html).

At its core, _Mathematical Optimization_ involves creating mathematical models that act as digital twins of the real-world scenarios that we aim to optimize. 
After developing those models, we input relevant data to create specific instances of an optimization problem. These instances are then solved using an appropriate solver to discover the best of all feasible solutions, which we call the optimal solution.

Modeling is a conceptual process, while coding the models is a technical craft. For the latter, we use the package [`pyomo`](https://www.pyomo.org/) as the aforementioned Jupyter book does. 

Please note, the model discussed subsequently is featured as Exercise $3.1$ in a forthcoming textbook from Cambridge University Press, of which the aforementioned Jupyter book is the online companion.

A mathematical optimization model can be seen as a blueprint for the intended optimal solution. Once instantiated, the model is processed by a solver that seeks the optimal solution, if one exists.

For solving instances, we can utilize powerful solvers. For problems such as the one we will describe, [`Gurobi`](https://www.gurobi.com/) is an outstanding commercial solver, while [HiGHS](https://highs.dev/), an excellent open-source alternative, is available under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).

The modeling process begins with identifying the decisions to be made, leading to the definition of decision variables. After naming these decisions and variables, we formalize the objective and constraints using functions of those variables.

- The **objective function** measures the quality of a solution.
- **Constraints** ensure that a solution adheres to all necessary rules to be considered feasible.

In our case, and many others, the functions will be linear, and our variables will have a _binary_ nature to represent _yes/no_ decisions.

For example, we need one variable per household to determine if it is served by an accessible open hospital, and another variable per hospital to indicate whether it is open.

Typical mathematical notation for expressing models starts by naming the sets that support the indices of variables and the model parameters derived from the data.

For our optimization challenge, these include:

### Sets
- **$I$** - the set of households
- **$J$** - the set of potential hospital locations
- **$J_i$** - the set of potential hospital locations within reach of household $i \in I$. Note: $J_i \subseteq J$.

### Parameters
- **$v_i$** - the headcount of household $i \in I$.

### Model `maximal covering` as in the article by [Church and ReVelle](https://www.semanticscholar.org/paper/The-maximal-covering-location-problem-Church-Revelle/c3de804bbeb15b0d8570ee3d9f4cbdf432993cfa)

This model defines variables $z_i$ for each household $i\in I$ to indicate if that household can be served by a hospital that is opened at $j \in J$, leading to the complete model as follows:

$$
\begin{align}
    \max\quad & \sum_{i\in I} w_iz_i  \\
    \text{subject to:}\quad & z_i \leq \sum_{j\in J_i } x_j & \forall i \in I \\
    & \sum_{j \in J} x_j \leq p \\
    & x_j \in \{0,1\} & \forall j \in J \\
    & z_i \in \{0,1\} & \forall i \in I
\end{align}
$$

The first line states the objective as to maximize the total headcount of the households served, while the second line (after _subject to:_) lists the first constraint: each household is only served if at least one hospital within reach is open. 
Then the number of hospitals to open constraints the selection and finally the binary nature of the variables used is specified. 

The model above selects up to $p$ hospitals. In the original paper Church and ReVelle selected exactly $p$ hospitals, but our model has advantages to be discussed later.

# Implementing the mathematical model

After the model is designed, implementing it amounts to translating the concepts from the mathematical expressions above into code.

The translation is more or less one on one, the main difference being that the variables are declared prior to using them, as always in programming, while the mathematical formulation traditionally declares the variables at the end. 

We use the package `Pyomo` to code our model.
