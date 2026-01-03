"""Generators for sample VNNLIB content for testing."""


def generate_simple_bounds_vnnlib(n_inputs=3, n_outputs=2):
    """Generate simple input bounds + output constraint VNNLIB.

    Args:
        n_inputs: Number of input variables
        n_outputs: Number of output variables

    Returns:
        VNNLIB content as string
    """
    lines = ["; Test property - simple bounds"]

    # Declare input variables
    lines.extend(f"(declare-const X_{i} Real)" for i in range(n_inputs))

    # Declare output variables
    lines.extend(f"(declare-const Y_{i} Real)" for i in range(n_outputs))

    lines.append("")

    # Add simple bounds for all inputs: 0 <= X_i <= 1
    lines.extend(f"(assert (and (<= X_{i} 1.0) (>= X_{i} 0.0)))" for i in range(n_inputs))

    # Add output constraint
    lines.append("(assert (<= Y_0 0.5))")

    return "\n".join(lines)


def generate_or_groups_vnnlib(n_groups=2, inputs_per_group=2):
    """Generate VNNLIB with multiple OR groups.

    Args:
        n_groups: Number of OR groups
        inputs_per_group: Number of input variables per group

    Returns:
        VNNLIB content as string
    """
    lines = ["; Test property - OR groups"]
    n_inputs = n_groups * inputs_per_group

    # Declare variables
    lines.extend(f"(declare-const X_{i} Real)" for i in range(n_inputs))
    lines.append("(declare-const Y_0 Real)")
    lines.append("")

    # Create OR groups with different bound ranges
    for group_idx in range(n_groups):
        constraints = []
        for i in range(inputs_per_group):
            var_idx = group_idx * inputs_per_group + i
            constraints.append(
                f"(and (<= X_{var_idx} {group_idx + 1}.0) (>= X_{var_idx} {group_idx * 0.5}))"
            )

        or_expr = f"(assert (or {' '.join(constraints)}))"
        lines.append(or_expr)

    lines.append("(assert (<= Y_0 1.0))")

    return "\n".join(lines)


def generate_complex_constraints_vnnlib():
    """Generate VNNLIB with complex linear constraints.

    Returns:
        VNNLIB content as string
    """
    return """; Test property - complex constraints
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (and (<= X_0 1.0) (>= X_0 -1.0)))
(assert (and (<= X_1 1.0) (>= X_1 -1.0)))
(assert (and (<= X_2 1.0) (>= X_2 0.0)))

(assert (<= (+ (* 2.0 X_0) (* 3.0 X_1)) 2.0))
(assert (>= (- Y_0 Y_1) 0.0))
(assert (<= (+ Y_0 Y_1) 1.5))
"""


def generate_linear_output_constraints_vnnlib():
    """Generate VNNLIB with linear output constraints.

    Returns:
        VNNLIB content as string
    """
    return """; Test property - linear output constraints
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (and (<= X_0 1.0) (>= X_0 0.0)))
(assert (and (<= X_1 2.0) (>= X_1 0.0)))

(assert (and
  (<= (- Y_0 (* 2.0 Y_1)) 0.5)
  (>= (+ Y_0 Y_1) 0.1)
))
"""


def generate_deep_nesting_vnnlib():
    """Generate VNNLIB with deeply nested AND/OR structures.

    Returns:
        VNNLIB content as string
    """
    return """; Test property - deep nesting
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)

(assert (and (<= X_0 1.0) (>= X_0 0.0)))
(assert (and (<= X_1 1.0) (>= X_1 0.0)))
(assert (and (<= X_2 1.0) (>= X_2 0.0)))

(assert (or
  (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.25))
  (and (<= X_0 1.0) (>= X_0 0.5) (<= Y_0 0.75))
  (and (or
    (and (<= X_1 0.3) (>= X_1 0.0))
    (and (<= X_1 0.7) (>= X_1 0.3))
  ) (<= Y_0 1.0))
))
"""


def generate_multiline_constraints_vnnlib():
    """Generate VNNLIB with multi-line constraints.

    Returns:
        VNNLIB content as string
    """
    return """; Test property - multi-line constraints
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)

(assert (and
  (<= X_0 1.0)
  (>= X_0 0.0)
  (<= X_1 1.0)
  (>= X_1 0.0)
))

(assert (<=
  (+
    (* 2.0 X_0)
    (* 3.0 X_1)
    (* 1.5 X_2)
  )
  5.0
))

(assert (>= Y_0 0.0))
"""


def generate_scientific_notation_vnnlib():
    """Generate VNNLIB with scientific notation values.

    Returns:
        VNNLIB content as string
    """
    return """; Test property - scientific notation
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (and (<= X_0 1.5e-10) (>= X_0 -1.5e-10)))
(assert (and (<= X_1 2.5e+03) (>= X_1 -2.5e+03)))

(assert (<= (* 1.0e-05 Y_0) 1.0e-04))
"""


def generate_negative_values_vnnlib():
    """Generate VNNLIB with negative coefficient values.

    Returns:
        VNNLIB content as string
    """
    return """; Test property - negative values
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (and (<= X_0 1.0) (>= X_0 -2.0)))
(assert (and (<= X_1 0.5) (>= X_1 -0.5)))

(assert (<= (+ (* -2.0 X_0) (* -3.0 X_1)) 1.0))
(assert (>= (- (* -1.0 Y_0) X_0) -0.5))
"""
