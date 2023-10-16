# Two approaches to non-stationary entropy estimation

Entropy is often used to represent an aggregated measure of the information flow between two sources. Often estimators assume a stationary entropy rate with respect to time. In settings that only consider a short time frame, this assumption may be reasonable, but over longer time periods, this may be invalid.

In some settings, entropy estimators are used to detect changes in entropy over time.

Consider a directed information flow between a source, $S$, and a target, $T$.

We can consider the information flow between these two nodes $I(S,T)$ as a time-varying quantity. At any instant, this can be represented by the entropy rate $h(S,T)$ or as an overall measure of entropy, $H(S,T)$.

In practice, estimating the entropy rate from a continuous data source can be problematic when measurements cannot be repeated. To make a Shannon estimate of the entropy rate, the size of the typical set or the discrete probability space is used. When a single data stream exists, this results in an estimate over many parameters using a small sample size. This results in an estimate with high variance, however the size of this variance is often not included in final estimate values.
However, understanding how the entropy rate between two sources varies over time can reveal dynamics between two actors indicative of interesting or abnormal behaviours which would not be detected using an aggregated estimator.

In this work, we will present two temporal entropy estimators. These two estimators capture two different aspects. The first measures the entropy rate across time, using properties of existing estimators to impose smoothness constraints on the resulting estimate, reducing variance. 

The second approach is designed to capture temporal patterns created by delayed relationships in information flows between a source and a target.