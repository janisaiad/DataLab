# V11 symmetry-breaking / commit-once theorem note

V11 is built around the strengthened hypothesis:

1. The primitive event is not routing; it is a symmetry-breaking transition in
   the MCL expert dynamics.
2. Once the class-expert risk matrix A_{c,k}(t) is non-degenerate, a simple
   Bayes-risk or linear/logistic router is enough.
3. The reverse diffusion trajectory should use a shared/baseline score before
   the speciation time and commit once afterwards.

The training window is centered around t_star to avoid diluting the class signal
across all noise levels.  The anti-collapse terms are deliberately weak: they
are there to keep experts from merging or one expert from taking all usage, not
to hand-code a class partition.
