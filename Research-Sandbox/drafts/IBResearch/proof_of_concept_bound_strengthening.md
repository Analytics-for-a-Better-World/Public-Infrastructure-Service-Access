# Proof-of-Concept Bound Strengthening Ideas

This note records two small proof-of-concept designs for strengthening the lower
bound of the IB timetabling models.  Both are deliberately toy-first.  The goal
is to test whether the idea can move a bound before investing in a full-instance
implementation.

## 1. Lifted Pattern Clique Cuts

In the subject-pattern formulation, let

\[
\lambda_{bp} =
\begin{cases}
1, & \text{if subject } b \text{ uses pattern } p,\\
0, & \text{otherwise.}
\end{cases}
\]

The existing pattern model already fixes some pairwise pattern-combination
variables \(\mu_{bpcq}\) to zero when two patterns are impossible together, for
example because their exams overload a day.  The lifted clique idea projects
some of that information directly into the \(\lambda\)-space.

Construct a graph whose nodes are subject patterns \((b,p)\).  Add an edge
between two nodes when the corresponding patterns cannot both appear in any
feasible timetable.  In the current toy proof of concept, two patterns are
treated as incompatible if either:

- their combined daily length violates the daily-length rule; or
- that pair alone would exceed the global same-slot clash cap.

Every clique \(K\) in this graph gives the valid inequality

\[
\sum_{(b,p)\in K} \lambda_{bp} \le 1.
\]

The cut is valid because every pair of nodes in the clique is mutually
incompatible, so an integer timetable can select at most one of them.  This is
called "lifted" here in the practical modelling sense: incompatibilities that
were previously expressed through many pair variables are lifted/projection-cut
back into the subject-pattern selection variables.

### Current implementation

The toy driver now accepts:

```powershell
py -B run_toy_pattern_model.py `
  --relax-lambda `
  --clique-cut-rounds 5 `
  --clique-cut-max-cuts-per-round 5 `
  --summary-output toy_pattern_cliquecuts_exact_lp_summary.csv `
  --progress-output toy_pattern_cliquecuts_exact_lp_progress.csv `
  --plot-output toy_pattern_cliquecuts_exact_lp_bounds.png `
  --log-output toy_pattern_cliquecuts_exact_lp.log `
  --output toy_pattern_cliquecuts_exact_lp_timetable.csv `
  --output-flag 0
```

The separator solves a small exact maximum-weight clique problem at the current
LP solution.  If the maximum clique has fractional weight greater than one, the
corresponding cut is added and the LP is reoptimized.

### First toy result

The exact separator found no violated clique cuts on the current toy LP
solution.  The LP bound stayed at 15,370.0.

This does not invalidate clique lifting.  It says that the simple pairwise
incompatibility graph is too weak for this toy relaxation.  The next stronger
version should be a cover cut for the same-slot clash budget:

\[
\sum_{(b,p)\in S} \lambda_{bp} \le |S|-1
\quad
\text{if selecting all patterns in } S
\text{ forces same-slot clash mass above } 15.
\]

This cover version is not limited to pairwise conflicts; it can cut sets whose
total conflict is too large even though no single pair violates the cap.

## 2. Lagrangian Variable-Splitting Bound

The second proof of concept splits subject decisions from pair-interaction
decisions.  Let \(x_{bp}\) be the subject-pattern copy and let
\(z_{bpcq}\) be the pair-interaction copy.  A compact formulation is:

\[
\sum_p x_{bp}=1 \qquad \forall b,
\]

\[
\sum_{p,q} z_{bpcq}=1 \qquad \forall \{b,c\},
\]

\[
\sum_q z_{bpcq}=x_{bp}, \qquad
\sum_p z_{bpcq}=x_{cq}.
\]

The pair variables carry the cross-subject Furlong objective terms and the
subject variables carry the within-subject terms.  The global same-slot clash
cap can also be dualized.  Relaxing the agreement constraints with multipliers
\(u\), \(v\), and the clash cap with multiplier \(\eta\ge 0\), gives a separable
lower-bound problem:

\[
\begin{aligned}
L(u,v,\eta)=
\min_{x,z}\;&
\sum_{b,p}(\alpha_{bp}+\eta h_{bp})x_{bp}
\;+\;
\sum_{\{b,c\},p,q}(\beta_{bpcq}+\eta h_{bpcq})z_{bpcq}
-15\eta\\
&+
\sum_{\{b,c\},p}u_{bcp}
\left(x_{bp}-\sum_q z_{bpcq}\right)
+
\sum_{\{b,c\},q}v_{bcq}
\left(x_{cq}-\sum_p z_{bpcq}\right).
\end{aligned}
\]

For every multiplier vector, \(L(u,v,\eta)\) is a valid lower bound.  The
Lagrangian dual maximizes this value over the multipliers.

### Current implementation

The toy proof-of-concept script is:

```powershell
py -B run_toy_lagrangian_poc.py `
  --iterations 2000 `
  --step-scale 0.2 `
  --max-step 50 `
  --history-output toy_lagrangian_poc_s02_history.csv `
  --summary-output toy_lagrangian_poc_s02_summary.csv `
  --plot-output toy_lagrangian_poc_s02_bounds.png
```

It uses a simple projected subgradient method with a Polyak-style step using the
known toy incumbent 25,190 as an upper bound.

### First toy result

The best lower bound from this simple edge-split Lagrangian run was
13,328.0055.  This is valid, but it is below the subject-pattern LP bound
15,370.0.  The interpretation is useful: splitting only into single-subject and
pair-edge blocks is not enough.  This split has too much freedom to choose
mutually inconsistent pair solutions.

The promising Lagrangian version should use larger integral blocks:

- dense subject communities rather than single pair edges;
- exact community subproblems with joint pattern choices;
- agreement constraints between the global subject-pattern layer and community
copies;
- a bundle or cutting-plane method for the dual, rather than only raw
subgradient updates.

This is the version that could, in principle, beat the LP relaxation: the dense
community subproblem may fail the integrality property, so Geoffrion's
LP-equivalence obstruction need not apply.

## 3. Dense-Community Lagrangian Dual

The stronger proof of concept was implemented in
`run_toy_lagrangian_community_poc.py`.  It keeps the same subject-pattern layer,
but replaces selected pair-edge blocks by exact dense-community blocks.

Let \(B\) be the set of subjects and \(P_b\) the feasible patterns of subject
\(b\).  The subject layer has variables conceptually represented by
\(x_{bp}\).  For each selected block \(K\), either a subject pair or a dense
community, the block has exact joint choices

\[
  r \in R_K \subseteq \prod_{b\in K} P_b.
\]

The set \(R_K\) excludes joint choices that violate hard local feasibility, such
as impossible daily-duration combinations.  Objective terms are partitioned:
within-subject terms stay in the subject layer, and each cross-subject pair cost
is assigned to exactly one copy block.  This avoids double counting.

The agreement constraints are

\[
  x_{bp}
  =
  \sum_{r\in R_K: r_b=p} y_{Kr}
  \qquad
  \forall K,\; b\in K,\; p\in P_b.
\]

Dualizing these equalities with multipliers \(u_{Kbp}\) gives the Lagrangian
dual

\[
\max_{u,\eta\ge 0}
\left\{
  \sum_{b\in B}
  \min_{p\in P_b}
  \left(
    \alpha_{bp} + \eta h_{bp}
    + \sum_{K\ni b}u_{Kbp}
  \right)
  +
  \sum_K
  \min_{r\in R_K}
  \left(
    \beta_{Kr} + \eta h_{Kr}
    - \sum_{b\in K}u_{Kb r_b}
  \right)
  -15\eta
\right\}.
\]

Here \(\alpha\) and \(\beta\) are the Furlong objective contributions, and
\(h\) is the same-slot clash contribution.  The multiplier \(\eta\ge0\) dualizes
the global same-slot clash cap.

For the toy proof of concept, the full hypograph dual master can be loaded
explicitly.  This is a cutting-plane master with all cuts present:

\[
\theta_b
\le
\alpha_{bp}+\eta h_{bp}+\sum_{K\ni b}u_{Kbp}
\qquad \forall b,p,
\]

\[
\phi_K
\le
\beta_{Kr}+\eta h_{Kr}-\sum_{b\in K}u_{Kb r_b}
\qquad \forall K,r\in R_K,
\]

\[
\max\;
\sum_b\theta_b+\sum_K\phi_K-15\eta .
\]

This is the exact Lagrangian dual of the split formulation for the chosen
blocks.  On larger instances the same model should be treated as a delayed-cut
or bundle method: solve the master with a small set of cuts, call exact
community subproblems to find violated hypograph cuts, and repeat.

### Commands and results

One dense 3-subject community:

```powershell
py -B run_toy_lagrangian_community_poc.py `
  --dual-mode full `
  --community-count 1 `
  --community-size 3 `
  --community-max-tuples 250000 `
  --summary-output toy_lagrangian_community1x3_full_dual_summary.csv `
  --history-output toy_lagrangian_community1x3_full_dual_history.csv `
  --plot-output toy_lagrangian_community1x3_full_dual_bounds.png
```

Result:

- selected community: `Machine Learning|Deep Learning|Coding Lab`;
- copy blocks: 43;
- copy tuples: 143,936;
- dual rows: 144,314;
- dual columns: 3,316;
- best lower bound: 16,507.5;
- elapsed time including model construction: 37.68 seconds;
- Gurobi solve time after construction: 2.40 seconds.

This is the important positive result.  The bound is above the base
subject-pattern LP bound of 15,370.0 and matches the explicit dense-triple
extended LP bound.  Thus the community Lagrangian dual does recover the
strength of the corresponding extended formulation.

Two dense 3-subject communities:

```powershell
py -B run_toy_lagrangian_community_poc.py `
  --dual-mode full `
  --community-count 2 `
  --community-size 3 `
  --community-max-tuples 250000 `
  --summary-output toy_lagrangian_community2x3_full_dual_summary.csv `
  --history-output toy_lagrangian_community2x3_full_dual_history.csv `
  --plot-output toy_lagrangian_community2x3_full_dual_bounds.png
```

Result:

- selected communities:
  `Machine Learning|Deep Learning|Coding Lab` and
  `SBS|Language A Literature|Law and Ethics`;
- copy blocks: 41;
- copy tuples: 144,944;
- best lower bound: 16,507.5;
- elapsed time including model construction: 57.87 seconds.

The second dense triple did not improve the bound.  This suggests that the
first community contains the dominant missing local consistency for this toy
relaxation, while the second selected triple is not active at the dual optimum.

One dense 4-subject community was also attempted with all hypograph cuts loaded
directly.  That version did not finish within a 10-minute wall-clock timeout in
the current implementation.  The limiting factor appears to be direct
construction of the all-cuts master, not the idea itself: the earlier explicit
4-subject extended LP produced a stronger bound, but with over two million joint
tuples.  The next implementation should therefore use delayed cut generation
for community subproblems rather than materializing every community cut at once.
