"""Common test data and constants for unit tests."""

# Common variable names used in tests
COMMON_INPUT_VARS = ["X_0", "X_1", "X_2", "X_3", "X_4"]
COMMON_OUTPUT_VARS = ["Y_0", "Y_1", "Y_2"]

# Common numeric values
EPSILON = 1e-6
LARGE_VALUE = 1e10
SMALL_VALUE = 1e-10

# Common bounds for testing
DEFAULT_INPUT_LOWER = 0.0
DEFAULT_INPUT_UPPER = 1.0
DEFAULT_OUTPUT_LOWER = -1.0
DEFAULT_OUTPUT_UPPER = 1.0

# Common coefficient values
SIMPLE_COEFFICIENTS = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
RANDOM_COEFFICIENTS = [0.123, -0.456, 1.789, -2.345, 3.456]

# Common expression patterns
SIMPLE_LINEAR_EXPR = "(<= (+ (* 2.0 X_0) (* 3.0 X_1)) 5.0)"
NESTED_EXPR = "(and (or (<= X_0 1.0) (>= X_0 0.0)) (<= Y_0 0.5))"
MULTILINE_EXPR = """(assert (and
  (<= X_0 1.0)
  (>= X_0 0.0)
  (<= X_1 1.0)
))"""
