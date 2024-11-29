import numpy as np
import re

import features

class FeatureSelector:
    def __init__(self, feature_names=features.feature_names):
        """
        Initialize the FeatureSelector with a list of feature names.

        Parameters:
        - feature_names: list of str, names of the features corresponding to the columns in the data.
        """
        self.feature_names = feature_names
        self.feature_index = {name: i for i, name in enumerate(feature_names)}  # Map feature names to indices
        self.conditions = []  # Store conditions for later representation

        raise RuntimeError("Let's not use the FeatureSelector")

    def _parse_string_condition(self, condition_str):
        """
        Parse a string condition into a tuple of bounds and feature.

        Parameters:
        - condition_str: str, condition in the form "200<x<=250".

        Returns:
        - A tuple representing the condition: (lower_bound, lower_op, feature, upper_op, upper_bound).
        """
        pattern = r'(?:(-?\d*\.?\d+)\s*([<>]=?)\s*)?([a-zA-Z_]\w*)\s*([<>]=?)?\s*(-?\d*\.?\d+)?'
        match = re.match(pattern, condition_str.strip())
        if not match:
            raise ValueError(f"Invalid condition string: {condition_str}")

        lower_val, lower_op, feature, upper_op, upper_val = match.groups()

        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature_names.")

        lower_bound = float(lower_val) if lower_val is not None else None
        upper_bound = float(upper_val) if upper_val is not None else None

        return lower_bound, lower_op, feature, upper_op, upper_bound

    def translate_condition(self, condition):
        """
        Translate a condition into a boolean selector function.

        Parameters:
        - condition: tuple or str, one of the following forms:
          - (x_low, "x"): x_low <= x
          - ("x", x_high): x < x_high
          - (x_low, "x", x_high): x_low <= x < x_high
          - "200<x<=250" (string): parsed into a condition.

        Returns:
        - A function that applies the condition to a NumPy array.
        """

        if isinstance(condition, str):
            condition = self._parse_string_condition(condition)

        self.conditions.append(condition)  # Store condition for string representation

        if len(condition) == 2:
            if isinstance(condition[0], str):  # ("x", x_high)
                feature = condition[0]
                upper_bound = condition[1]
                feature_idx = self.feature_index[feature]
                return lambda data: data[:, feature_idx] < upper_bound
            elif isinstance(condition[1], str):  # (x_low, "x")
                lower_bound = condition[0]
                feature = condition[1]
                feature_idx = self.feature_index[feature]
                return lambda data: lower_bound <= data[:, feature_idx]
        elif len(condition) == 3:  # (x_low, "x", x_high)
            lower_bound = condition[0]
            feature = condition[1]
            upper_bound = condition[2]
            feature_idx = self.feature_index[feature]
            return lambda data: (lower_bound <= data[:, feature_idx]) & (data[:, feature_idx] < upper_bound)
        elif len(condition) == 5:  # (lower_bound, lower_op, "x", upper_op, upper_bound)
            lower_bound, lower_op, feature, upper_op, upper_bound = condition
            feature_idx = self.feature_index[feature]

            # Map relational operators to lambda functions
            ops = {
                '<': lambda x, val: x < val,
                '<=': lambda x, val: x <= val,
                '>': lambda x, val: x > val,
                '>=': lambda x, val: x >= val,
                '==': lambda x, val: x == val,
            }

            def condition_func(data):
                column = data[:, feature_idx]
                lower_pass = ops[lower_op](column, lower_bound) if lower_op and lower_bound is not None else True
                upper_pass = ops[upper_op](column, upper_bound) if upper_op and upper_bound is not None else True
                return lower_pass & upper_pass

            return condition_func
        else:
            raise ValueError(f"Invalid condition: {condition}")

    def build_selector(self, conditions):
        """
        Build a selector function from a list of conditions.

        Parameters:
        - conditions: list of tuples or strings, where each is a condition in one of the supported forms.

        Returns:
        - A function that applies the combined conditions (AND) to a NumPy array.
        """
        self.conditions = []  # Clear previous conditions
        condition_functions = [self.translate_condition(cond) for cond in conditions]

        def selector(data):
            # Apply all condition functions and combine them with AND
            return np.logical_and.reduce([func(data) for func in condition_functions])

        return selector

    def __str__(self):
        """
        Return a string-based representation of the conditions.
        """
        condition_strs = []
        for cond in self.conditions:
            if len(cond) == 2:
                if isinstance(cond[0], str):  # ("x", x_high)
                    condition_strs.append(f"{cond[0]}<{cond[1]}")
                else:  # (x_low, "x")
                    condition_strs.append(f"{cond[0]}<={cond[1]}")
            elif len(cond) == 3:  # (x_low, "x", x_high)
                condition_strs.append(f"{cond[0]}<={cond[1]}<{cond[2]}")
            elif len(cond) == 5:  # (lower_bound, lower_op, "x", upper_op, upper_bound)
                parts = []
                if cond[0] is not None and cond[1] is not None:
                    parts.append(f"{cond[0]}{cond[1]}")
                parts.append(cond[2])
                if cond[3] is not None and cond[4] is not None:
                    parts.append(f"{cond[3]}{cond[4]}")
                condition_strs.append("".join(parts))

        return " AND ".join(condition_strs)

if __name__=="__main__":
    # Define feature names
    feature_names = ["x", "y", "z"]

    # Initialize the FeatureSelector
    selector = FeatureSelector(feature_names)

    # Define conditions (mix of strings and tuples)
    conditions = [
        "190<=x<300",        # String-based condition
        (0, "y", 5),         # Tuple-based condition
        ("z", 10)            # Upper bound on "z"
    ]

    # Build the selector
    select = selector.build_selector(conditions)

    # Example data
    data = np.array([
        [210, 4.0, 8.0],   # Satisfies all conditions
        [190, 4.0, 8.0],   # Does not satisfy 200<x
        [210, 6.0, 8.0],   # Does not satisfy y < 5
        [210, 4.0, 12.0]   # Does not satisfy z < 10
    ])

    # Apply the selector
    result = select(data)

    print("Data:")
    print(data)
    print("Selection result:", result)

