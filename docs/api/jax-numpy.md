# `duq.jax.numpy`

A unit-aware mirror of [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html),
quaxified on first access: [`duq.jax.Quantity`][duq.jax.Quantity] operands
dispatch through the duq primitive rules, while plain arrays behave exactly as
in `jax.numpy`.

::: duq.jax.numpy
