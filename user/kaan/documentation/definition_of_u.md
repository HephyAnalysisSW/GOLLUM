Starting point is the *profile likelihood ratio* test statistic ($q_c$)

$$q_c(D) = -2 \log \frac{\max_{\nu} L(D \mid c, \nu)}{\max_{c', \nu'} L(D \mid c', \nu')} \quad (2.21)$$

and the extended likelihood:

$$L(D \mid c, \nu) = P(N_{\text{obs}} \mid L(\nu)\sigma(c, \nu)) \prod_{i=1}^{N_{\text{obs}}} p(x_i \mid c, \nu) \quad (2.22)$$

Define the Poisson mean

$$\lambda(c, \nu) = L(\nu)\sigma(c, \nu), \quad \lambda(0,0) = L(0)\sigma(0, 0).$$

**1) Take the log-likelihood ratio to the reference point $(0,0)$**

Ignoring the constant $\log(N_{\text{obs}}!)$ from the Poisson (it cancels in ratios),

$$\log \frac{L(D \mid c, \nu)}{L(D \mid 0, 0)} = \underbrace{\left[ -\lambda + \lambda_0 + N_{\text{obs}} \log \frac{\lambda}{\lambda_0} \right]}_{\text{Poisson term}} + \underbrace{\sum_{i=1}^{N_{\text{obs}}} \log \frac{p(x_i \mid c, \nu)}{p(x_i \mid 0, 0)}}_{\text{shape term}}$$

**2) Rewrite the shape term using $d\sigma = \sigma p \, dx$**

They introduce

$$d\sigma(x \mid c, \nu) = \sigma(c, \nu) p(x \mid c, \nu) \, dx. \quad (2.23)$$

So

$$p(x \mid c, \nu) = \frac{1}{\sigma(c, \nu)} \frac{d\sigma(x \mid c, \nu)}{dx} \quad \Rightarrow \quad \frac{p(x \mid c, \nu)}{p(x \mid 0, 0)} = \frac{d\sigma(x \mid c, \nu)}{d\sigma(x \mid 0, 0)} \cdot \frac{\sigma(0, 0)}{\sigma(c, \nu)}.$$

Therefore

$$\sum_{i=1}^{N_{\text{obs}}} \log \frac{p(x_i \mid c, \nu)}{p(x_i \mid 0, 0)} = \sum_{i=1}^{N_{\text{obs}}} \log \frac{d\sigma(x_i \mid c, \nu)}{d\sigma(x_i \mid 0, 0)} - N_{\text{obs}} \log \frac{\sigma(c, \nu)}{\sigma(0, 0)}.$$

**3) The $\sigma$ logs cancel between Poisson and shape terms**

From the Poisson part:

$$N_{\text{obs}} \log \frac{\lambda}{\lambda_0} = N_{\text{obs}} \log \frac{L(\nu)}{L(0)} + N_{\text{obs}} \log \frac{\sigma(c, \nu)}{\sigma(0, 0)}.$$

This **exactly cancels** the $-N_{\text{obs}} \log(\sigma(c,\nu)/\sigma(0,0))$ from the shape term, leaving

$$\log \frac{L(D \mid c, \nu)}{L(D \mid 0, 0)} = -(L(\nu)\sigma(c, \nu) - L(0)\sigma(0, 0)) + \sum_{i=1}^{N_{\text{obs}}} \log \left( \frac{L(\nu)}{L(0)} \frac{d\sigma(x_i \mid c, \nu)}{d\sigma(x_i \mid 0, 0)} \right).$$

Finally, define $u$ via

$$-\frac{1}{2} u(D \mid c, \nu) = \log \frac{L(D \mid c, \nu)}{L(D \mid 0, 0)},$$

which gives exactly

$$-\frac{1}{2} u(D \mid c, \nu) = -(L(\nu)\sigma(c, \nu) - L(0)\sigma(0, 0)) + \sum_{i=1}^{N_{\text{obs}}} \log \left( \frac{L(\nu)}{L(0)} \frac{d\sigma(x_i \mid c, \nu)}{d\sigma(x_i \mid 0, 0)} \right). \quad (2.25)$$

That's the derivation from (2.21)+(2.22) (plus the $d\sigma$ relation in (2.23)) to (2.25).