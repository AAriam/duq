# duq.jax primitive coverage

Generated from `duq.jax.PRIMITIVE_COVERAGE` (the module-level registry in
`src/duq/jax/_primitives.py`) by `docs/dev/generate_jax_coverage.py`;
`tests/jax/test_meta_jax.py` asserts this table stays in sync with the
registered rules.  Any `lax` primitive **not** listed here fails loud with
`UnsupportedOperationError` naming the primitive (via
`duq.jax.Quantity.default`); units are never silently dropped.

The control-flow primitives (`cond`, `while`, `scan`, `pjit`) are handled
by quax itself and work with quantity carries/branches out of the box.

| Primitive              | Unit rule |
|------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `abs`                  | unit preserved (affine rejected) |
| `acos`                 | dimensionless in, radian out |
| `acosh`                | dimensionless in (scale applied), dimensionless out |
| `add`                  | same dimension; right operand converted to the left unit |
| `argmax`               | plain index array out |
| `argmin`               | plain index array out |
| `asin`                 | dimensionless in, radian out |
| `asinh`                | dimensionless in (scale applied), dimensionless out |
| `atan`                 | dimensionless in, radian out |
| `atan2`                | same dimension in, radian out |
| `atanh`                | dimensionless in (scale applied), dimensionless out |
| `bitcast_convert_type` | plain array out (raw bits carry no unit) |
| `broadcast_in_dim`     | structural; unit preserved |
| `cbrt`                 | unit ** 1/3 |
| `ceil`                 | unit preserved |
| `concatenate`          | all operands converted to the first quantity's unit |
| `conj`                 | unit preserved |
| `convert_element_type` | structural; unit preserved |
| `copy`                 | structural; unit preserved |
| `cos`                  | angle in (rad/deg scale-converted at trace time), dimensionless out |
| `cosh`                 | dimensionless in (scale applied), dimensionless out |
| `cummax`               | cumulative; unit preserved |
| `cummin`               | cumulative; unit preserved |
| `cumprod`              | scale-1 dimensionless only (per-element unit is ambiguous) |
| `cumsum`               | cumulative; unit preserved |
| `div`                  | units divide |
| `dot_general`          | units multiply (matmul/dot/einsum contractions) |
| `dynamic_slice`        | indexing; unit preserved |
| `dynamic_update_slice` | quantity update converted to the operand's unit; plain update adopts it |
| `eq`                   | converted compare; incompatible dims -> all-False |
| `eq_to`                | total-order compare (converted) |
| `erf`                  | dimensionless in (scale applied), dimensionless out |
| `erf_inv`              | dimensionless in (scale applied), dimensionless out |
| `erfc`                 | dimensionless in (scale applied), dimensionless out |
| `exp`                  | dimensionless in (scale applied), dimensionless out |
| `exp2`                 | dimensionless in (scale applied), dimensionless out |
| `expm1`                | dimensionless in (scale applied), dimensionless out |
| `floor`                | unit preserved |
| `gather`               | indexing; unit preserved |
| `ge`                   | converted compare; incompatible dims raise |
| `gt`                   | converted compare; incompatible dims raise |
| `imag`                 | unit preserved |
| `integer_pow`          | unit ** y (static integer exponent) |
| `is_finite`            | plain bool array out |
| `le`                   | converted compare; incompatible dims raise |
| `le_to`                | total-order compare (converted) |
| `log`                  | dimensionless in (scale applied), dimensionless out |
| `log1p`                | dimensionless in (scale applied), dimensionless out |
| `logistic`             | dimensionless in (scale applied), dimensionless out |
| `lt`                   | converted compare; incompatible dims raise |
| `lt_to`                | total-order compare (converted) |
| `max`                  | same dimension; unit preserved |
| `min`                  | same dimension; unit preserved |
| `mul`                  | units multiply |
| `ne`                   | converted compare; incompatible dims -> all-True |
| `neg`                  | unit preserved (affine rejected) |
| `nextafter`            | same dimension; unit preserved |
| `pad`                  | quantity padding value converted to the operand's unit; plain adopts it |
| `pow`                  | unit ** concrete dimensionless exponent (traced exponent rejected) |
| `real`                 | unit preserved |
| `reduce_and`           | plain bool array out |
| `reduce_max`           | reduction; unit preserved |
| `reduce_min`           | reduction; unit preserved |
| `reduce_or`            | plain bool array out |
| `reduce_prod`          | unit ** reduced_size (shapes are static) |
| `reduce_sum`           | reduction; unit preserved |
| `rem`                  | same dimension; unit preserved |
| `reshape`              | structural; unit preserved |
| `rev`                  | structural; unit preserved |
| `round`                | unit preserved |
| `rsqrt`                | unit ** -1/2 |
| `scatter`              | quantity updates converted to the operand's unit; plain updates adopt it |
| `scatter-add`          | quantity updates converted to the operand's unit; plain updates adopt it |
| `scatter-max`          | quantity updates converted to the operand's unit; plain updates adopt it |
| `scatter-min`          | quantity updates converted to the operand's unit; plain updates adopt it |
| `scatter-sub`          | quantity updates converted to the operand's unit; plain updates adopt it |
| `select_n`             | quantity branches converted to the last quantity case's unit (np.where parity); plain branches adopt it; predicate plain |
| `sign`                 | plain array out (sign is scale-free) |
| `sin`                  | angle in (rad/deg scale-converted at trace time), dimensionless out |
| `sinh`                 | dimensionless in (scale applied), dimensionless out |
| `slice`                | structural; unit preserved |
| `sort`                 | per-operand; each output keeps its operand's unit |
| `split`                | structural; unit preserved on every output |
| `sqrt`                 | unit ** 1/2 |
| `square`               | unit squared |
| `squeeze`              | structural; unit preserved |
| `stack`                | all operands converted to the first quantity's unit |
| `stop_gradient`        | structural; unit preserved |
| `sub`                  | same dimension; degC - degC yields the coherent SI difference (K) |
| `tan`                  | angle in (rad/deg scale-converted at trace time), dimensionless out |
| `tanh`                 | dimensionless in (scale applied), dimensionless out |
| `tile`                 | structural; unit preserved |
| `transpose`            | structural; unit preserved |

## Known exclusions

- `jax.numpy.interp` compares its inputs against a hard-coded epsilon
  literal internally (dimensionally unsound at the primitive level); strip
  units with `duq.ustrip` around it.
- `jnp.deg2rad`/`rad2deg`/`radians`/`degrees` lower to a bare
  multiplication, so `duq.jax.numpy` ships curated unit-aware overrides
  for them instead of the quaxified originals.
- `lax.linalg` decompositions (`cholesky`, `svd`, `eig`, ...),
  `cumlogsumexp`, `scatter_mul`, FFTs and convolutions are uncovered and
  fail loud; convert to a canonical unit and strip before calling them.
- `cumprod`/`reduce_prod` differ: `reduce_prod` maps to `unit ** n`
  (static reduced size), while `cumprod` would give every element a
  different unit and is rejected for dimensional operands.
