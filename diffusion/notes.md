Diffusion models.  Noise addition is performed with a noise scheduler, which describes the mean/variance of each successive sample as noise is generated additively.
x_t ~ N(sqrt(1-\beta_t)*x_t-1, \beta_t*I)

So variance at each step is the Beta value at that step.  Mean at each step is the last step's mean shifted toward zero by sqrt(1-beta) where beta changes over time.