# Core API (`duq`)

The top-level namespace re-exports the pure, array-free core: the `Quantity`,
`Unit`, `Dimension` and `UnitRegistry` types, the `duq.units` / `duq.dims`
attribute namespaces, the `uconvert` / `ustrip` free functions, and the error
hierarchy.

::: duq
    options:
      show_root_heading: false
      members:
        - Quantity
        - Unit
        - Dimension
        - UnitRegistry
        - QuantityLike
        - unit
        - dimension
        - uconvert
        - ustrip
        - default_registry
        - DuqError
        - DimensionalityError
        - AffineUnitError
        - AnalysisError
        - RegistryMismatchError
        - UndefinedUnitError
        - UnitParseError
        - UnsupportedOperationError

## Attribute namespaces

::: duq.units

::: duq.dims
