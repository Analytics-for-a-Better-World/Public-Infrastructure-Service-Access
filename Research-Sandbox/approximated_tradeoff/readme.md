# Fast Approximate Pareto Curves for Maximum Covering Problems

This repository provides a highly scalable method to generate **approximate Pareto curves for facility location and service access problems** without requiring any optimization solver such as Gurobi.

The approach is designed for **national scale datasets with hundreds of thousands of demand points and candidate facilities**, where traditional mixed integer optimization may take hours or days.

Using efficient data structures and vectorized numerical computation, the method produces high quality tradeoff curves **in minutes or seconds** on a standard laptop.

---

# Main Idea

The method consists of three main stages.

## 1 Greedy solution with all facilities

First, a greedy algorithm is run over the **full set of candidate facilities**.

At each step the facility that increases coverage the most is opened.  
This continues until no additional population can be covered.

This produces a solution achieving the **maximum attainable coverage**.

The greedy implementation used here is inspired by:

Fleur Theulen, Master Thesis  
https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/blob/main/publications/Master%20Thesis%20Fleur%20Theulen_git.pdf

However, the new implementation is **several orders of magnitude faster**, mainly due to

* compressed sparse row storage (CSR) for catchment relationships  
* carefully designed vectorized NumPy operations  
* minimal memory movement

More about CSR format:  
https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

---

## 2 Local search reduction

The greedy solution often opens more facilities than necessary.

A fast **local search improvement phase** is therefore applied:

- repeatedly attempt **swaps of one open facility with one closed facility**
- accept the first improving move found
- stop when no further improvement is possible

Additionally, a **reduction step** removes redundant facilities while maintaining the same coverage.

The result is a **smaller solution with identical coverage**.

---

## 3 Approximate Pareto curve generation

Finally, the greedy algorithm is executed again but **restricted to the facilities selected in the reduced solution**.

This produces a sequence of incremental deployments that approximate a **Pareto frontier between budget and coverage**.

This idea of incremental deployment was introduced in:

https://www.thelancet.com/journals/lansea/article/PIIS2772-3682(24)00086-6/fulltext

---

# Approximation Quality

A true Pareto curve consists of optimal solutions for each budget.

The curve generated here is therefore **an approximation**, because solutions extend one another and are not solved independently.

However, empirical results show that the difference from optimal coverage is typically:

**less than 0.2 percent**, mostly at the very beginning of the curve.

The benefit is enormous computational savings.

What would normally require **many solver runs and several days of computation** can be obtained in **a few minutes**.

---

# Instances Used in Experiments

The experiments include both benchmark and national scale instances.

## Benchmark instance

Kings dataset

```

kings: (7,730 × 7,730) density = 1.304%

```

Source:

Máximo, Nascimento, Carvalho  
Intelligent guided adaptive search for the maximum covering location problem  
Computers & Operations Research, 2017.

The instance represents **Brooklyn in New York**, with population nodes and facilities distributed along the street network.

---

## Vietnam national instances

Population dataset:

Meta for Good population distribution dataset.

Total demand points:

```

406,784 population locations

```

Facilities include existing facilities and grid generated candidates.

Grid resolutions used:

```

10 km grid
5 km grid
1 km grid

```

Accessibility thresholds tested:

```

5 km
10 km
50 km
100 km
200 km

```

These instances contain **hundreds of thousands of candidate facilities**.

# Performance

The approximate Pareto curves are computed extremely quickly.

Even the largest instances with **356,000 candidate facilities** are solved in a few seconds.

All curves for all instances can be generated in **less than 90 seconds on a laptop**.

---

# Key Advantages

- No optimization solver required
- Extremely scalable
- Works for national scale planning problems
- Produces high quality approximate Pareto curves
- Very fast computation

This makes the approach particularly suitable for **policy analysis and large scale planning exercises** where many scenarios must be evaluated.

---

# References

Theulen, F.  
Master Thesis on public infrastructure access  
https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/blob/main/publications/Master%20Thesis%20Fleur%20Theulen_git.pdf

Lancet Regional Health Southeast Asia article on incremental deployment  
https://www.thelancet.com/journals/lansea/article/PIIS2772-3682(24)00086-6/fulltext

Máximo, V. R., Nascimento, M. C. V., Carvalho, A. C. P. L. F.  
Intelligent guided adaptive search for the maximum covering location problem  
Computers & Operations Research, 2017.

CSR sparse matrix format  
https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

