import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from collections import defaultdict, Counter
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_UNARY_OPS  = {"sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "asin", "acos", "atan"}
_BINARY_OPS = {"Add", "Mul", "Pow"}
_ALL_OPS = list(_UNARY_OPS | _BINARY_OPS)
_NUMERIC_LEAVES = {"Integer", "Float", "Rational", "Half", "NegativeOne", "One", "Zero", "NumberSymbol", "Pi", "Exp1"}

CONST_POOL = [
    sp.Integer(1), sp.Integer(2), sp.Integer(3), sp.Integer(4),
    sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4),
    sp.Integer(-1), sp.Integer(-2), sp.Rational(-1, 2),
    sp.pi,
]
UNARY_BUILDERS = {
    "sin":  sp.sin,  "cos":  sp.cos,  "tan":  sp.tan,
    "exp":  sp.exp,  "log":  sp.log,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "Abs":  sp.Abs,  "sqrt": sp.sqrt,
}

def characterise_domain(
    expr,
    n_vars:       int,
    n_probe:      int   = 10000,
    percentile:   float = 5.0, # 5% to trim bounding box edges, leaving a slightly smaller but more reliable domain (5% to 95% range)
    sep_accuracy: float = 0.75,
    rng:          np.random.Generator | None = None,
) -> dict | None:
    """
    Empirically characterise the applicability domain of a sympy expression
    by probing it with random inputs from U(0,1)^n and labelling each
    sample valid or invalid based on the output.

    A sample is invalid if the output is NaN, Inf, or has extreme magnitude
    (< 1e-12 or > 1e8 in absolute value).

    Three types of dependency rules are extracted:
        ratio:   xi < thr * xj           (e.g. v < c)
        product: xi < thr * xj * xk      (e.g. lambda < n*d)
        power:   xi < thr * xj^2

    Parameters
    ----------
    expr         : sympy.Expr
    n_vars       : number of input variables
    n_probe      : probe samples from U(0,1)^n  (default 1000)
    percentile   : percentile used to trim bounding box edges (default 5)
    sep_accuracy : minimum balanced accuracy for a rule to be kept (default 0.75)
    rng          : numpy random Generator

    Returns
    -------
    dict with keys:
        "bounds"       : np.ndarray (n_vars, 2)  per-variable [lo, hi]
        "rules"        : list of dicts with keys:
                           type, i, j, k, threshold, accuracy, str
        "valid_probe"  : float  fraction of probe points that were valid
        "n_valid"      : int    number of valid probe points
    or None if fewer than 20 valid probe points were found.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = rng.uniform(0.0, 1.0, size=(n_probe, n_vars))
    y = evaluate_equation(expr, X)

    valid = (
        np.isfinite(y) &
        (np.abs(y) < 1e8) &
        (np.abs(y) > 1e-12)
    )
    # if only a handful of valid points, this function cant be realiabily characterised — return None to signal this
    if valid.sum() < 20:
        return None

    X_valid   = X[valid]
    X_invalid = X[~valid]

    # Bounding box: percentile range of valid input values per variable
    bounds = np.column_stack([
        np.percentile(X_valid, percentile,       axis=0),
        np.percentile(X_valid, 100 - percentile, axis=0),
    ])

    # Dependency rules
    eps   = 1e-10
    rules = []

    def _balanced_acc(lhs_v, lhs_inv, rhs_v, rhs_inv, thr):
        pv = (lhs_v   < thr * rhs_v).mean()
        pi = (lhs_inv >= thr * rhs_inv).mean()
        return (pv + pi) / 2.0

    for i in range(n_vars):
        xi_v   = X_valid[:, i]
        xi_inv = X_invalid[:, i] if len(X_invalid) >= 10 else None
        if xi_inv is None:
            continue

        for j in range(n_vars):
            if i == j:
                continue
            xj_v   = X_valid[:, j]
            xj_inv = X_invalid[:, j]

            # Ratio: xi < thr * xj
            thr = float(np.percentile(xi_v / (xj_v + eps), 100 - percentile))
            acc = _balanced_acc(xi_v, xi_inv, xj_v, xj_inv, thr)
            if acc >= sep_accuracy:
                rules.append({
                    "type": "ratio", "i": i, "j": j, "k": None,
                    "threshold": thr, "accuracy": float(acc),
                    "str": f"x{i+1} < {thr:.3f} * x{j+1}",
                })

            # Power: xi < thr * xj^2 (only if better than ratio)
            thr2 = float(np.percentile(xi_v / (xj_v**2 + eps), 100 - percentile))
            acc2 = _balanced_acc(xi_v, xi_inv, xj_v**2, xj_inv**2, thr2)
            best_acc = max((r["accuracy"] for r in rules
                            if r["i"] == i and r["j"] == j), default=0)
            if acc2 >= sep_accuracy and acc2 > best_acc + 0.02:
                rules.append({
                    "type": "power", "i": i, "j": j, "k": None,
                    "threshold": thr2, "accuracy": float(acc2),
                    "str": f"x{i+1} < {thr2:.3f} * x{j+1}^2",
                })

            # Product: xi < thr * xj * xk
            for k in range(j + 1, n_vars):
                if i == k:
                    continue
                xk_v   = X_valid[:, k]
                xk_inv = X_invalid[:, k]
                prod_v   = xj_v   * xk_v
                prod_inv = xj_inv * xk_inv
                thr_p = float(np.percentile(xi_v / (prod_v + eps), 100 - percentile))
                acc_p = _balanced_acc(xi_v, xi_inv, prod_v, prod_inv, thr_p)
                if acc_p >= sep_accuracy:
                    rules.append({
                        "type": "product", "i": i, "j": j, "k": k,
                        "threshold": thr_p, "accuracy": float(acc_p),
                        "str": f"x{i+1} < {thr_p:.3f} * x{j+1} * x{k+1}",
                    })

    rules.sort(key=lambda r: -r["accuracy"])

    return {
        "bounds":      bounds,
        "rules":       rules,
        "valid_probe": float(valid.mean()),
        "n_valid":     int(valid.sum()),
    }

def sample_inputs(
    n_samples: int,
    n_variables: int,
    uniform_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Generate synthetic inputs for variables.

    Each variable gets a random sub-interval [a, b] ⊂ [0, 1], then is sampled
    from either U(a, b) or a truncated N(μ, σ) clipped to (a, b).

    Returns
    -------
    dict:
        "X"           : np.ndarray (n_samples, n_variables)
        "dist_types"  : list of "uniform" or "normal", length n_variables
        "sub_ranges"  : list of (a, b) tuples, length n_variables
        "dist_params" : list of dicts with distribution parameters
    """
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError(f"uniform_ratio must be in [0, 1], got {uniform_ratio}")

    if rng is None:
        rng = np.random.default_rng()

    # Assign distribution types
    n_uniform = round(n_variables * uniform_ratio)
    dist_types = ["uniform"] * n_uniform + ["normal"] * (n_variables - n_uniform)
    rng.shuffle(dist_types)

    X = np.empty((n_samples, n_variables))
    sub_ranges, dist_params = [], []

    for i, dist in enumerate(dist_types):

        # Step 1 & 2: Sample sub-range [a, b] jointly uniform over all valid pairs
        a, b = sorted(rng.uniform(0.0, 1.0, size=2))
        while b - a < 0.15:
            a, b = sorted(rng.uniform(0.0, 1.0, size=2))
        sub_ranges.append((a, b))

        if dist == "uniform":
            X[:, i] = rng.uniform(a, b, size=n_samples)
            dist_params.append({"a": a, "b": b})

        else:
            # Step 3: For truncated normal, sample mu and sigma within the sub-range
            # mu must be within [a, b]
            mu_lo     = a + 0.05 * (b - a)
            mu_hi     = b - 0.05 * (b - a)
            mu        = rng.uniform(mu_lo, mu_hi)

            # sigma derived so 3-sigma fits within [a, b]
            sigma_max = min(mu - a, b - mu) / 3.0
            if sigma_max < 0.01:
                # Range too tight for normal — fall back to uniform
                X[:, i] = rng.uniform(a, b, size=n_samples)
                dist_params.append({"a": a, "b": b})
                continue
            sigma = rng.uniform(sigma_max * 0.1, sigma_max)

            # clip to [a, b] not [0, 1]
            alpha = (a - mu) / sigma
            beta  = (b - mu) / sigma
            X[:, i] = stats.truncnorm.rvs(
                alpha, beta, loc=mu, scale=sigma,
                size=n_samples, random_state=rng,
            )
            dist_params.append({"mu": mu, "sigma": sigma, "a": a, "b": b})

    return {
        "X": X,
        "dist_types": dist_types,
        "sub_ranges": sub_ranges,
        "dist_params": dist_params,
    }

def sample_from_domain(
    domain:        dict | None,
    n_samples:     int,
    n_vars:        int,
    uniform_ratio: float = 0.5,
    rng:           np.random.Generator | None = None,
    max_attempts:  int   = 10,
) -> dict:
    """
    Sample input points that satisfy the applicability domain rules,
    using mixed uniform/truncated normal with random sub-ranges
    that are constrained to the domain bounds.

    The workflow is:
        1. For each variable, draw a random sub-range within the domain
           bounding box [lo_j, hi_j] instead of full [0,1]
        2. Assign each variable a uniform or truncated normal distribution
           within that sub-range
        3. Over-sample then apply dependency rules as a filter
        4. Return n_samples valid points

    Falls back to standard sample_inputs() if domain is None or if
    not enough valid points can be collected.

    Parameters
    ----------
    domain        : dict from characterise_domain(), or None
    n_samples     : number of points to return
    n_vars        : number of input variables
    uniform_ratio : fraction of variables assigned uniform distributions
    rng           : numpy random Generator
    max_attempts  : max rejection-sampling rounds before giving up

    Returns
    -------
    dict with keys:
        "X"           : np.ndarray of shape (n_samples, n_vars)
        "dist_types"  : list of "uniform" or "normal" per variable
        "sub_ranges"  : list of (a, b) tuples per variable
        "dist_params" : list of dicts per variable
    """
    if rng is None:
        rng = np.random.default_rng()

    if domain is None:
        return None

    bounds = domain["bounds"]   # shape (n_vars, 2)

    def _sample_batch(n_draw):
        """Sample n_draw points using sample_inputs logic within domain bounds."""
        n_uniform  = round(n_vars * uniform_ratio)
        dist_types = ["uniform"] * n_uniform + ["normal"] * (n_vars - n_uniform)
        rng.shuffle(dist_types)

        X           = np.empty((n_draw, n_vars))
        sub_ranges  = []
        dist_params = []

        for j, dist in enumerate(dist_types):
            lo, hi    = bounds[j, 0], bounds[j, 1]
            width     = hi - lo
            min_width = max(0.3 * width, 0.05)

            # Draw random sub-range within [lo, hi], try 20 times, if too narrow fallback to full range
            for _ in range(20):
                a_rel, b_rel = sorted(rng.uniform(0.0, 1.0, size=2))
                a = lo + a_rel * width
                b = lo + b_rel * width
                if b - a >= min_width:
                    break
            else:
                a, b = lo, hi   # fallback: use full domain range

            sub_ranges.append((a, b))

            if dist == "uniform":
                X[:, j] = rng.uniform(a, b, size=n_draw)
                dist_params.append({"a": a, "b": b})

            else:
                # Truncated normal within (a, b)
                mu_lo = a + 0.05 * (b - a)
                mu_hi = b - 0.05 * (b - a)
                mu    = rng.uniform(mu_lo, mu_hi)
                sigma_max = min(mu - a, b - mu) / 3.0
                if sigma_max < 1e-6:
                    # Range too tight for normal — fall back to uniform
                    X[:, j] = rng.uniform(a, b, size=n_draw)
                    dist_params.append({"a": a, "b": b})
                    continue
                sigma = rng.uniform(sigma_max * 0.1, sigma_max)
                alpha = (a - mu) / sigma
                beta  = (b - mu) / sigma
                X[:, j] = stats.truncnorm.rvs(
                    alpha, beta, loc=mu, scale=sigma,
                    size=n_draw, random_state=rng,
                )
                dist_params.append({"mu": mu, "sigma": sigma, "a": a, "b": b})

        return X, dist_types, sub_ranges, dist_params

    collected_X      = []
    last_dist_types  = []
    last_sub_ranges  = []
    last_dist_params = []
    attempts         = 0

    while sum(len(c) for c in collected_X) < n_samples and attempts < max_attempts:
        attempts += 1
        n_draw = max(n_samples * 5, 500)

        X_batch, dt, sr, dp = _sample_batch(n_draw)

        # Apply dependency rules (ratio, product, power)
        mask = np.ones(n_draw, dtype=bool)
        for rule in domain["rules"]:
            i, j, t = rule["i"], rule["j"], rule["threshold"]
            rtype   = rule.get("type", "ratio")
            if rtype == "ratio":
                mask &= X_batch[:, i] < t * X_batch[:, j]
            elif rtype == "product":
                mask &= X_batch[:, i] < t * X_batch[:, j] * X_batch[:, rule["k"]]
            elif rtype == "power":
                mask &= X_batch[:, i] < t * X_batch[:, j]**2

        if mask.any():
            collected_X.append(X_batch[mask])
            last_dist_types  = dt
            last_sub_ranges  = sr
            last_dist_params = dp

    if not collected_X:
        return None

    X_out = np.vstack(collected_X)
    idx   = rng.permutation(len(X_out))[:n_samples]
    if len(idx) < n_samples:
        repeats = int(np.ceil(n_samples / len(X_out)))
        X_out   = np.tile(X_out, (repeats, 1))[:n_samples]
    else:
        X_out = X_out[idx]

    return {
        "X":           X_out,
        "dist_types":  last_dist_types,
        "sub_ranges":  last_sub_ranges,
        "dist_params": last_dist_params,
    }

def _classify_node(expr) -> str:
    """Return the semantic role of a sympy node."""
    t = type(expr).__name__
    if t == "Symbol":
        return "variable"
    if t in _NUMERIC_LEAVES or isinstance(expr, sp.Number):
        return "constant"
    if t in _UNARY_OPS:
        return "unary"
    if t in _BINARY_OPS:
        return "binary"
    return "other"

def extract_features(expr_or_str) -> dict:
    """
    Extract structural features from a sympy expression or equation string.

    Parses the expression into a tree and computes features across three groups:

    Tree-level:
        depth               -- longest path from root to any leaf
        n_nodes             -- total node count (operators + leaves)
        n_leaves            -- variable + constant leaves
        n_operators         -- internal operator nodes

    Variables & constants:
        n_variables         -- total variable appearances (with repetition)
        n_unique_variables  -- number of distinct variable symbols
        n_constants         -- number of numeric/constant leaves
        variable_reuse      -- 1 if any variable appears more than once

    Operator composition:
        operator_counts     -- dict mapping operator name -> count
        n_unary             -- count of unary operators (sin, exp, ...)
        n_binary            -- count of binary operators (Add, Mul, Pow)
        unary_ratio         -- n_unary / n_operators
        binary_ratio        -- n_binary / n_operators

    Structural / shape:
        avg_operator_depth  -- mean depth of operator nodes (nesting level)
        avg_leaf_depth      -- mean depth of leaf nodes
        branching_factor    -- mean number of children per operator node

    Parameters
    ----------
    expr_or_str : str or sympy.Expr
        Equation as a string (parsed with sympy.sympify) or a sympy expression.

    Returns
    -------
    dict with the features listed above.
    """
    if isinstance(expr_or_str, str):
        expr = sp.sympify(expr_or_str)
    else:
        expr = expr_or_str

    operator_counts = Counter()
    leaf_depths     = []
    node_depths     = []
    variables_seen  = []
    constants_seen  = []

    def _traverse(node, depth):
        kind = _classify_node(node)

        if kind == "variable":
            variables_seen.append(str(node))
            leaf_depths.append(depth)

        elif kind == "constant":
            constants_seen.append(str(node))
            leaf_depths.append(depth)

        else:  # unary / binary / other operator
            operator_counts[type(node).__name__] += 1
            node_depths.append((type(node).__name__, depth))
            for child in node.args:
                _traverse(child, depth + 1)

    _traverse(expr, depth=0)

    n_variables   = len(variables_seen)
    n_unique_vars = len(set(variables_seen))
    n_constants   = len(constants_seen)
    n_operators   = sum(operator_counts.values())
    n_nodes       = n_variables + n_constants + n_operators
    n_leaves      = n_variables + n_constants
    tree_depth    = max(leaf_depths) if leaf_depths else 0

    n_unary  = sum(v for k, v in operator_counts.items() if k in _UNARY_OPS)
    n_binary = sum(v for k, v in operator_counts.items() if k in _BINARY_OPS)
    unary_ratio  = n_unary  / n_operators if n_operators > 0 else 0.0
    binary_ratio = n_binary / n_operators if n_operators > 0 else 0.0

    avg_operator_depth = (
        sum(d for _, d in node_depths) / len(node_depths) if node_depths else 0.0
    )
    avg_leaf_depth = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0.0

    total_children = sum(
        len(node.args)
        for node in sp.preorder_traversal(expr)
        if _classify_node(node) in ("unary", "binary", "other")
    )
    branching_factor = total_children / n_operators if n_operators > 0 else 0.0

    return {
        "equation": str(expr),   # canonical sympy string — makes the dict self-contained
        # Tree-level
        "depth":              tree_depth,
        "n_nodes":            n_nodes,
        "n_leaves":           n_leaves,
        "n_operators":        n_operators,
        # Variables & constants
        "n_variables":        n_variables,
        "n_unique_variables": n_unique_vars,
        "n_constants":        n_constants,
        "variable_reuse":     int(n_variables > n_unique_vars),
        # Operator composition
        "operator_counts":    dict(operator_counts),
        "n_unary":            n_unary,
        "n_binary":           n_binary,
        "unary_ratio":        round(unary_ratio, 4),
        "binary_ratio":       round(binary_ratio, 4),
        # Structural / shape
        "avg_operator_depth": round(avg_operator_depth, 4),
        "avg_leaf_depth":     round(avg_leaf_depth, 4),
        "branching_factor":   round(branching_factor, 4),
    }

def load_feynman_csv(path: str = "data/FeynmanEquations.csv") -> list[dict]:
    """
    Load the Feynman equations CSV and return a list of dicts, one per equation.

    Download the CSV manually from:
        https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv
    then pass the local path here.

    Each dict contains:
        "filename"    : str   — equation ID, e.g. "I.6.2a"
        "number"      : int   — row number in the original CSV
        "output"      : str   — output variable name
        "formula"     : str   — original formula with physical variable names
        "var_names"   : list  — variable names in declared CSV order
        "n_variables" : int   — number of variables
        "generic"     : str   — formula with names replaced by x1, x2, ...
                                None if sympy failed to parse the formula

    Parameters
    ----------
    path : str
        Local file path to FeynmanEquations.csv.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["Number", "Formula"])

    equations = []
    for _, row in df.iterrows():
        filename = str(row["Filename"]).strip()
        number   = int(row["Number"])
        output   = str(row["Output"]).strip()
        formula  = str(row["Formula"]).strip()
        n_vars   = int(row["# variables"])

        # Extract variable names in declared CSV order
        var_names = []
        for i in range(1, 11):
            name = row.get(f"v{i}_name", "")
            if pd.notna(name) and str(name).strip():
                var_names.append(str(name).strip())

        try:
            generic = _substitute_vars(formula, var_names)
        except Exception:
            generic = None

        equations.append({
            "filename":    filename,
            "number":      number,
            "output":      output,
            "formula":     formula,
            "var_names":   var_names,
            "n_variables": n_vars,
            "generic":     generic,
        })

    return equations

def _substitute_vars(formula: str, var_names: list[str]) -> str:
    """
    Replace physical variable names with x1, x2, ... using sympy substitution.
    Variable order follows the CSV declaration order.
    Sorts by length descending to prevent partial matches (e.g. theta1 before theta).
    """
    sorted_vars = sorted(var_names, key=len, reverse=True)
    index_map   = {name: var_names.index(name) + 1 for name in var_names}
    local_dict = {name: sp.Symbol(name) for name in var_names}
    expr       = sp.sympify(formula, locals=local_dict)
    subs        = {
        sp.Symbol(name): sp.Symbol(f"x{index_map[name]}")
        for name in sorted_vars
    }
    return str(expr.subs(subs))

def render_tree(expr_or_str) -> str:
    """
    Render a sympy expression as an ASCII tree.

    Each node displays:
        [OperatorName]  for internal nodes  e.g. [Mul], [exp], [Add]
        value           for leaf nodes      e.g. x1, 2, pi, -1/2

    Parameters
    ----------
    expr_or_str : str or sympy.Expr

    Returns
    -------
    str — multi-line ASCII tree, ready to print.

    Example
    -------
    >>> print(render_tree("sin(x1 / x2) + x3**2"))
    [Add]
    ├── [Pow]
    │   ├── x3
    │   └── 2
    └── [sin]
        └── [Mul]
            ├── x1
            └── [Pow]
                ├── x2
                └── -1
    """
    if isinstance(expr_or_str, str):
        expr = sp.sympify(expr_or_str)
    else:
        expr = expr_or_str

    lines = []

    def _label(node) -> str:
        if len(node.args) == 0:
            return str(node)
        return f"[{type(node).__name__}]"

    def _walk(node, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + _label(node))
        if node.args:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.args):
                _walk(child, prefix + extension, i == len(node.args) - 1)

    lines.append(_label(expr))
    for i, child in enumerate(expr.args):
        _walk(child, "", i == len(expr.args) - 1)

    return "\n".join(lines)

def render_tree_svg(expr_or_str) -> str:
    """
    Render a sympy expression as a top-down SVG tree.

    Nodes are colored by type:
        purple  -- binary operators (Add, Mul, Pow)
        teal    -- unary operators  (sin, cos, exp, sqrt, ...)
        amber   -- variable leaves  (x1, x2, ...)
        gray    -- constant leaves  (2, pi, -1/2, ...)

    Layout uses leaf-count weighting so subtrees get proportional
    horizontal space. Edges use straight diagonal lines.

    Parameters
    ----------
    expr_or_str : str or sympy.Expr

    Returns
    -------
    str -- SVG markup string, ready to write to a .svg file or embed in HTML.

    Usage
    -----
    svg = render_tree_svg("sin(x1/x2) + x3**2")

    # Save to file
    with open("tree.svg", "w") as f:
        f.write(svg)

    # Display inline in Jupyter
    from IPython.display import SVG, display
    display(SVG(svg))
    """
    if isinstance(expr_or_str, str):
        expr = sp.sympify(expr_or_str)
    else:
        expr = expr_or_str

    # ----- Layout -----
    positions = {}   # nid -> (x, y, label, node, child_nids)
    counter   = [0]

    def _count_leaves(node):
        if not node.args:
            return 1
        return sum(_count_leaves(c) for c in node.args)

    def _node_label(node):
        return str(node) if not node.args else type(node).__name__

    def _assign(node, x0, x1, depth):
        nid = counter[0]; counter[0] += 1
        x   = (x0 + x1) / 2
        y   = 60 + depth * 90
        child_nids = []
        if node.args:
            total = sum(_count_leaves(c) for c in node.args)
            cx = x0
            for child in node.args:
                w   = _count_leaves(child) / total
                cid = counter[0]
                _assign(child, cx, cx + w * (x1 - x0), depth + 1)
                child_nids.append(cid)
                cx += w * (x1 - x0)
        positions[nid] = (x, y, _node_label(node), node, child_nids)
        return nid

    _assign(expr, 40, 640, 0)

    max_y  = max(v[1] for v in positions.values())
    height = int(max_y + 70)

    # ----- Color classification -----
    def _color(node, label):
        if not node.args:
            if label.startswith("x") and label[1:].isdigit():
                return "amber"
            return "gray"
        if type(node).__name__ in _UNARY_OPS:
            return "teal"
        return "purple"

    # ----- Colors (self-contained hex, no CSS variables needed) -----
    FILLS   = {"purple": "#EEEDFE", "teal": "#E1F5EE",
               "amber":  "#FAEEDA", "gray": "#F1EFE8"}
    STROKES = {"purple": "#534AB7", "teal": "#0F6E56",
               "amber":  "#BA7517", "gray": "#5F5E5A"}
    TEXTS   = {"purple": "#3C3489", "teal": "#085041",
               "amber":  "#633806", "gray": "#444441"}

    W = 680
    R = 22

    o = []
    o.append('<svg width="100%" viewBox="0 0 ' + str(W) + ' ' + str(height) + '" xmlns="http://www.w3.org/2000/svg" role="img">')
    o.append('  <title>Expression tree</title>')
    o.append('  <desc>' + str(expr) + '</desc>')

    for nid, (x, y, label, node, cnids) in positions.items():
        for cid in cnids:
            cx, cy = positions[cid][0], positions[cid][1]
            o.append(
                '  <line x1="' + str(round(x, 1)) + '" y1="' + str(round(y, 1)) +
                '" x2="' + str(round(cx, 1)) + '" y2="' + str(round(cy, 1)) +
                '" stroke="#888780" stroke-width="1" opacity="0.6"/>'
            )

    for nid, (x, y, label, node, cnids) in positions.items():
        col  = _color(node, label)
        fill = FILLS[col]
        strk = STROKES[col]
        txt  = TEXTS[col]
        if node.args:
            o.append(
                '  <circle cx="' + str(round(x, 1)) + '" cy="' + str(round(y, 1)) +
                '" r="' + str(R) + '" fill="' + fill +
                '" stroke="' + strk + '" stroke-width="1.5"/>'
            )
            o.append(
                '  <text x="' + str(round(x, 1)) + '" y="' + str(round(y, 1)) +
                '" text-anchor="middle" dominant-baseline="central"' +
                ' font-family="monospace" font-size="11" font-weight="500"' +
                ' fill="' + txt + '">' + label + '</text>'
            )
        else:
            lw = max(len(label) * 8 + 16, 44)
            lh = 28
            lx = round(x - lw / 2, 1)
            ly = round(y - lh / 2, 1)
            o.append(
                '  <rect x="' + str(lx) + '" y="' + str(ly) +
                '" width="' + str(lw) + '" height="' + str(lh) +
                '" rx="6" fill="' + fill + '" stroke="' + strk + '" stroke-width="1"/>'
            )
            o.append(
                '  <text x="' + str(round(x, 1)) + '" y="' + str(round(y, 1)) +
                '" text-anchor="middle" dominant-baseline="central"' +
                ' font-family="monospace" font-size="11"' +
                ' fill="' + txt + '">' + label + '</text>'
            )

    o.append('</svg>')
    return '\n'.join(o)

def extract_production_rules(equations: list[dict], verbose: bool = False) -> dict:
    """
    Walk every equation tree and collect PCFG production rules.

    For each internal node encountered across all equations, records:
        parent_type -> children_types  (as a tuple)

    Then computes counts and normalized probabilities per parent type,
    plus distributions over root operators and leaf types.

    Uses the same node-type labels as render_tree_svg:
        Add, Mul, Pow, sin, cos, exp, log, sqrt, inv, 1/sqrt  -- operators
        VAR    -- variable leaf  (Symbol)
        CONST  -- constant leaf  (Integer, Rational, Float, pi, ...)

    Parameters
    ----------
    equations : list of dicts from load_feynman_csv()
        Only equations where "generic" is not None are used.

    Returns
    -------
    dict with keys:
        "rules"       : dict  parent -> {children_tuple -> count}
        "probs"       : dict  parent -> {children_tuple -> probability}
        "root_counts" : Counter  distribution over root operator types
        "root_probs"  : dict     normalized root operator probabilities
        "leaf_counts" : Counter  distribution over leaf types (VAR / CONST)
        "leaf_probs"  : dict     normalized leaf probabilities
        "n_equations" : int      number of equations successfully processed
        "failed"      : list     filenames that could not be parsed
    """

    def _node_type(node):
        t = type(node).__name__
        if not node.args:
            return "VAR" if t == "Symbol" else "CONST"
        if t == "Pow" and len(node.args) == 2:
            e = node.args[1]
            if e == sp.Rational(1, 2):   return "sqrt"
            if e == sp.Rational(-1, 2):  return "1/sqrt"
            if e == sp.Integer(-1):      return "inv"
        return t

    def _collect(node, rules, leaf_counts):
        if not node.args:
            leaf_counts[_node_type(node)] += 1
            return
        parent   = _node_type(node)
        children = tuple(_node_type(c) for c in node.args)
        rules[parent][children] += 1
        for child in node.args:
            _collect(child, rules, leaf_counts)

    rules       = defaultdict(Counter)
    root_counts = Counter()
    leaf_counts = Counter()
    failed      = []
    n_ok        = 0

    for eq in equations:
        if eq["generic"] is None:
            failed.append(eq["filename"])
            continue
        try:
            _ = extract_features(eq["generic"]) # sanity check that features are extrable, before collecting rules
            expr = sp.sympify(eq["generic"])
            root_counts[_node_type(expr)] += 1
            _collect(expr, rules, leaf_counts)
            n_ok += 1
        except Exception:
            failed.append(eq["filename"])

    if failed:
        print(f"Skipped {len(failed)} equations: {failed}")

    probs = {}
    for parent, child_rules in rules.items():
        total = sum(child_rules.values())
        probs[parent] = {children: count / total
                         for children, count in child_rules.items()}

    root_total = sum(root_counts.values())
    root_probs = {op: count / root_total for op, count in root_counts.items()}

    leaf_total = sum(leaf_counts.values())
    leaf_probs = {t: count / leaf_total for t, count in leaf_counts.items()}

    if verbose:
        print(f"\nProcessed {n_ok} equations")
        print(f"Unique parent types : {len(rules)}")
        print(f"Total rules         : {sum(len(v) for v in rules.values())}")
        print(f"Total rule instances: {sum(sum(v.values()) for v in rules.values())}")

        print(f"\n--- Root operator distribution ---")
        for op, p in sorted(root_probs.items(), key=lambda x: -x[1]):
            print(f"  {op:10s}  p={p:.3f}  (n={root_counts[op]})")

        print(f"\n--- Leaf type distribution ---")
        for t, p in sorted(leaf_probs.items(), key=lambda x: -x[1]):
            print(f"  {t:8s}  p={p:.3f}  (n={leaf_counts[t]})")

        print(f"\n--- Production rules per parent (top 3 each) ---")
        for parent in sorted(rules.keys()):
            print(f"  {parent}:")
            for children, prob in sorted(probs[parent].items(), key=lambda x: -x[1])[:3]:
                print(f"    {str(children):45s}  p={prob:.3f}  (n={rules[parent][children]})")

    return {
        "rules":       dict(rules),
        "probs":       probs,
        "root_counts": root_counts,
        "root_probs":  root_probs,
        "leaf_counts": leaf_counts,
        "leaf_probs":  leaf_probs,
        "n_equations": n_ok,
        "failed":      failed,
    }

def extract_production_rules_bayesian(equations: list[dict], alpha: float = 1.0, verbose: bool = False, optimize: bool = False) -> dict:
    """
    Walk every equation tree and collect PCFG production rules,
    using a Bayesian Dirichlet prior to smooth the probabilities.

    For each internal node encountered across all equations, records:
        parent_type -> children_types  (as a tuple)

    Then computes smoothed probabilities using the Dirichlet-Multinomial
    posterior mean:
        p(rule) = (count(rule) + alpha) / (total_count + alpha * K)

    where K is the number of unique rules for that parent type.
    This pulls rare rules toward a uniform distribution, reducing the
    dominance of frequent patterns like Mul -> (VAR, VAR).

    Parameters
    ----------
    equations : list of dicts from load_feynman_csv()
    alpha     : Dirichlet concentration parameter (default 1.0).
        alpha -> 0  : pure MLE (same as original, corpus-dominated)
        alpha = 1   : Laplace smoothing (one pseudocount per rule)
        alpha = 5   : moderate smoothing toward uniform
        alpha -> inf: completely uniform, ignores corpus

    Returns
    -------
    dict with keys:
        "rules"       : dict  parent -> {children_tuple -> count}
        "probs"       : dict  parent -> {children_tuple -> smoothed probability}
        "root_counts" : Counter  raw root operator counts
        "root_probs"  : dict     smoothed root operator probabilities
        "leaf_counts" : Counter  raw leaf type counts
        "leaf_probs"  : dict     smoothed leaf type probabilities
        "n_equations" : int      number of equations successfully processed
        "failed"      : list     filenames that could not be parsed
        "alpha"       : float    the concentration parameter used
    """

    def _node_type(node):
        t = type(node).__name__
        if not node.args:
            return "VAR" if t == "Symbol" else "CONST"
        if t == "Pow" and len(node.args) == 2:
            e = node.args[1]
            if e == sp.Rational(1, 2):   return "sqrt"
            if e == sp.Rational(-1, 2):  return "1/sqrt"
            if e == sp.Integer(-1):      return "inv"
        return t

    def _collect(node, rules, leaf_counts):
        if not node.args:
            leaf_counts[_node_type(node)] += 1
            return
        parent   = _node_type(node)
        children = tuple(_node_type(c) for c in node.args)
        rules[parent][children] += 1
        for child in node.args:
            _collect(child, rules, leaf_counts)

    rules       = defaultdict(Counter)
    root_counts = Counter()
    leaf_counts = Counter()
    failed      = []
    n_ok        = 0

    for eq in equations:
        if eq["generic"] is None:
            failed.append(eq["filename"])
            continue
        try:
            _ = extract_features(eq["generic"])
            expr = sp.sympify(eq["generic"])
            root_counts[_node_type(expr)] += 1
            _collect(expr, rules, leaf_counts)
            n_ok += 1
        except Exception:
            failed.append(eq["filename"])

    if failed:
        print(f"Skipped {len(failed)} equations: {failed}")

    # Bayesian smoothing — Dirichlet-Multinomial posterior mean
    # p(rule) = (count + alpha) / (total + alpha * K)

    # Production rules
    probs = {}
    for parent, child_rules in rules.items():
        K     = len(child_rules)
        total = sum(child_rules.values())
        probs[parent] = {
            rule: (count + alpha) / (total + alpha * K)
            for rule, count in child_rules.items()
        }

    # Root operator probabilities
    K_root     = len(root_counts)
    total_root = sum(root_counts.values())
    root_probs = {
        op: (count + alpha) / (total_root + alpha * K_root)
        for op, count in root_counts.items()
    }

    # Leaf type probabilities
    K_leaf     = len(leaf_counts)
    total_leaf = sum(leaf_counts.values())
    leaf_probs = {
        t: (count + alpha) / (total_leaf + alpha * K_leaf)
        for t, count in leaf_counts.items()
    }

    if verbose:
        print(f"\nProcessed {n_ok} equations  (alpha={alpha})")
        print(f"Unique parent types : {len(rules)}")
        print(f"Total rules         : {sum(len(v) for v in rules.values())}")
        print(f"Total rule instances: {sum(sum(v.values()) for v in rules.values())}")

        print(f"\n--- Root operator distribution (smoothed) ---")
        for op, p in sorted(root_probs.items(), key=lambda x: -x[1]):
            raw_p = root_counts[op] / total_root
            print(f"  {op:10s}  p={p:.3f}  (raw={raw_p:.3f}, n={root_counts[op]})")

        print(f"\n--- Leaf type distribution (smoothed) ---")
        for t, p in sorted(leaf_probs.items(), key=lambda x: -x[1]):
            raw_p = leaf_counts[t] / total_leaf
            print(f"  {t:8s}  p={p:.3f}  (raw={raw_p:.3f}, n={leaf_counts[t]})")

        print(f"\n--- Production rules per parent (top 3 each, smoothed) ---")
        for parent in sorted(rules.keys()):
            print(f"  {parent}:")
            for children, prob in sorted(probs[parent].items(),
                                        key=lambda x: -x[1])[:3]:
                raw_count = rules[parent][children]
                print(f"    {str(children):45s}  "
                    f"p={prob:.3f}  (n={raw_count})")

    return {
        "rules":       dict(rules),
        "probs":       probs,
        "root_counts": root_counts,
        "root_probs":  root_probs,
        "leaf_counts": leaf_counts,
        "leaf_probs":  leaf_probs,
        "n_equations": n_ok,
        "failed":      failed,
        "alpha":       alpha,
    }

def generate_equation(
    grammar:   dict,
    tau:       float = 6.0,
    max_depth: int   = 12,
    min_vars:  int   = 2,
    max_tries: int   = 20,
    rng:       np.random.Generator | None = None,
) -> sp.Expr | None:
    
    """Generate a new sympy expression using a given grammar."""

    if rng is None:
        rng = np.random.default_rng()

    var_counter = [0]

    def _new_var():
        var_counter[0] += 1
        return sp.Symbol("x" + str(var_counter[0]))

    def _sample_leaf():
        if rng.random() < grammar["leaf_probs"].get("VAR", 0.4):
            return _new_var()
        return CONST_POOL[int(rng.integers(len(CONST_POOL)))]

    def _sample_rule(op_type):
        rules = grammar["probs"].get(op_type)
        if not rules:
            return None
        keys  = list(rules.keys())
        probs = np.array([rules[k] for k in keys], dtype=float)
        probs /= probs.sum()
        return keys[int(rng.choice(len(keys), p=probs))]

    def _sample_root():
        rp    = grammar["root_probs"]
        keys  = list(rp.keys())
        probs = np.array([rp[k] for k in keys], dtype=float)
        probs /= probs.sum()
        return keys[int(rng.choice(len(keys), p=probs))]

    def _build(op_type, children):
        if op_type == "Add":    return sp.Add(*children)
        if op_type == "Mul":    return sp.Mul(*children)
        if op_type in ("Pow", "^"):
            return sp.Pow(children[0], children[1])
        if op_type == "inv":    return sp.Pow(children[0], sp.Integer(-1))
        if op_type == "sqrt":   return sp.sqrt(children[0])
        if op_type == "1/sqrt": return sp.Pow(children[0], sp.Rational(-1, 2))
        if op_type in UNARY_BUILDERS:
            return UNARY_BUILDERS[op_type](children[0])
        raise ValueError("Unknown op: " + op_type)

    def _is_valid(expr):
        for node in sp.preorder_traversal(expr):
            if node in (sp.I, sp.zoo, sp.oo, sp.nan):
                return False
            if isinstance(node, sp.core.numbers.ImaginaryUnit):
                return False
        return True

    def _expand(op_type, depth):
        p_leaf = 1.0 - np.exp(-depth / tau)
        if depth >= max_depth or (depth > 0 and rng.random() < p_leaf):
            return _sample_leaf()
        children_types = _sample_rule(op_type)
        if children_types is None:
            return _sample_leaf()
        children = []
        for ct in children_types:
            if ct == "VAR":
                children.append(_new_var())
            elif ct == "CONST":
                children.append(CONST_POOL[int(rng.integers(len(CONST_POOL)))])
            else:
                children.append(_expand(ct, depth + 1))
        if op_type in ("Pow", "^") and len(children) != 2:
            return _sample_leaf()
        try:
            return _build(op_type, children)
        except Exception:
            return _sample_leaf()

    for _ in range(max_tries):
        try:
            var_counter[0] = 0        # reset so each attempt starts at x1

            root_op = _sample_root()
            expr    = _expand(root_op, 0)

            if not _is_valid(expr):
                continue
            if len(expr.free_symbols) < min_vars:
                continue

            return expr

        except Exception:
            continue

    return None

def generate_random_expr(max_depth, rng, tau=4.0):
    """Generate a random sympy expression with a soft depth bias."""
    var_counter = [0]
    # create a new variable in sympy
    def _new_var():
        var_counter[0] += 1
        return sp.Symbol("x" + str(var_counter[0]))
    def _expand(depth):
        p_leaf = 1.0 - np.exp(-depth / tau) # tau=4 creates 8-10 depth trees with good variability, while tau=2 creates more shallow trees
        if depth >= max_depth or (depth > 0 and rng.random() < p_leaf):
            return (_new_var() if rng.random() < 0.5
                    else _CONST_POOL[int(rng.integers(len(_CONST_POOL)))])
        op = _ALL_OPS[int(rng.integers(len(_ALL_OPS)))]
        try:
            if op == "Add":  return sp.Add(_expand(depth+1), _expand(depth+1))
            if op == "Mul":  return sp.Mul(_expand(depth+1), _expand(depth+1))
            if op == "Pow":  return sp.Pow(_expand(depth+1), _CONST_POOL[int(rng.integers(len(_CONST_POOL)))])
            if op == "sin":  return sp.sin(_expand(depth+1))
            if op == "cos":  return sp.cos(_expand(depth+1))
            if op == "exp":  return sp.exp(_expand(depth+1))
            if op == "log":  return sp.log(_expand(depth+1))
            if op == "sqrt": return sp.sqrt(_expand(depth+1))
        except Exception:
            pass
        return _new_var()
    return _expand(0)

def evaluate_equation(expr, X: np.ndarray) -> np.ndarray:
    """
    Evaluate a sympy expression on an input matrix X.

    Variable symbols x1, x2, ... are mapped to columns X[:,0], X[:,1], ...
    in alphabetical order of symbol name. Numerical errors (division by zero,
    sqrt of negative, etc.) produce NaN rather than raising exceptions.

    Parameters
    ----------
    expr : sympy.Expr
    X    : np.ndarray of shape (n_samples, n_variables)

    Returns
    -------
    np.ndarray of shape (n_samples,), dtype float.
    May contain NaN where evaluation is undefined.
    """
    free   = sorted(expr.free_symbols, key=lambda s: s.name)
    n_vars = len(free)

    if n_vars == 0:
        try:
            return np.full(X.shape[0], float(expr))
        except Exception:
            return np.full(X.shape[0], np.nan)

    if n_vars > X.shape[1]:
        return np.full(X.shape[0], np.nan)

    f    = sp.lambdify(free, expr, modules="numpy")
    args = [X[:, i] for i in range(n_vars)]
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.asarray(f(*args), dtype=float)
        if y.shape == ():
            y = np.full(X.shape[0], float(y))
        return y
    except Exception:
        return np.full(X.shape[0], np.nan)
    
def generate_dataset(
    grammar:               dict,
    n_samples:             int   = 200,
    uniform_ratio:         float = 0.5,
    valid_ratio_threshold: float = 1,
    tau:                   float = 3.0,
    max_depth:             int   = 10,
    max_attempts:          int   = 50,
    n_probe:               int   = 10000,
    rng:                   np.random.Generator | None = None,
) -> dict | None:
    """
    Generate a random equation and dataset by sampling from the grammar and characterising its domain.

    Parameters
    ----------
    grammar               : dict from extract_production_rules()
    n_samples             : number of input samples to generate
    uniform_ratio         : fraction of uniform vs normal input variables
    valid_ratio_threshold : minimum fraction of finite (non-NaN) outputs
    tau                   : soft-forcing temperature for generate_equation()
    max_depth             : hard depth cap for generate_equation()
    max_attempts          : max generation attempts before giving up
    n_probe               : number of random probe points to characterise the domain
    rng                   : numpy random Generator

    Returns
    -------
    dict with keys:
        "expr"        : sympy.Expr   -- the generated equation
        "eq_str"      : str          -- canonical sympy string form
        "X"           : np.ndarray (n_samples, n_variables)
        "y"           : np.ndarray (n_samples,)  -- may contain NaN, if valid_ratio_threshold < 1
        "dist_types"  : list of "uniform" or "normal" per variable
        "sub_ranges"  : list of (a, b) tuples per variable
        "dist_params" : list of dicts per variable
        "n_vars"      : int          -- number of free variables
        "valid_ratio" : float        -- fraction of finite outputs
        "features"    : dict         -- from extract_features()
    or None if no valid equation was found within max_attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_attempts):

        # 1. Generate equation
        expr = generate_equation(grammar, tau=tau, max_depth=max_depth, rng=rng)
        if expr is None:
            continue

        free   = sorted(expr.free_symbols, key=lambda s: s.name)
        n_vars = len(free)
        if n_vars == 0:
            continue

        # 2. Sample inputs
        domain = characterise_domain(expr, n_vars, n_probe=n_probe, rng=rng)
        inp = sample_from_domain(domain, n_samples, n_vars,
                                uniform_ratio=uniform_ratio, rng=rng)
        if inp is None:
            continue   # burns one attempt and tries a new equation
        
        X   = inp["X"]

        # 3. Evaluate
        y = evaluate_equation(expr, X)

        # 4. Check valid ratio
        valid_ratio = np.isfinite(y).mean()
        if valid_ratio < valid_ratio_threshold:
            continue

        # 5. Extract structural features
        try:
            features = extract_features(expr)
        except Exception:
            features = {}

        return {
            "expr":        expr,
            "eq_str":      str(expr),
            "X":           X,
            "y":           y,
            "dist_types":  inp["dist_types"],
            "sub_ranges":  inp["sub_ranges"],
            "dist_params": inp["dist_params"],
            "n_vars":      n_vars,
            "valid_ratio": valid_ratio,
            "features":    features,
            "domain":      domain,
        }

    return None

def generate_datasets(
        equations:             dict | None = None,
        n_datasets:            int         = 100,
        n_samples_per_dataset: int         = 200,
        uniform_ratio:         float       = 0.5,
        tau:                   int         = 7,
        max_depth:             int         = 10,
        n_vars:                int         = 3,
        rng:                   np.random.Generator | None = None,
):
    """
    Generate a list of synthetic datasets.

    If equations is provided, learns a B-PCFG grammar from the corpus and
    optimises tau and max_depth automatically — user-supplied tau, max_depth,
    and n_vars are ignored in this case.

    If equations is None, generates datasets using random expression trees
    with the user-supplied tau, max_depth, and n_vars.

    Parameters
    ----------
    equations             : list of dicts from load_feynman_csv(), or None
    n_datasets            : number of datasets to generate
    n_samples_per_dataset : samples per dataset
    uniform_ratio         : fraction of uniform input variables
    tau                   : soft-forcing temperature (only used if equations is None)
    max_depth             : max expression tree depth (only used if equations is None)
    n_vars                : number of input variables (only used if equations is None)
    rng                   : numpy random Generator

    Returns
    -------
    list of dataset dicts
    """
    if rng is None:
        rng = np.random.default_rng()

    np.seterr(over="ignore")

    if equations is None:
        # Random generation — no grammar, user controls structure
        print(f"Generating {n_datasets} random datasets "
              f"(tau={tau}, max_depth={max_depth}, n_vars={n_vars})")

        datasets = []
        while len(datasets) < n_datasets:
            # Sample inputs for n_vars variables
            inp = sample_inputs(n_samples_per_dataset, n_vars,
                                uniform_ratio=uniform_ratio, rng=rng)
            X   = inp["X"]


            # Keep generating until expression uses exactly n_vars variables
            for _ in range(50):
                expr = generate_random_expr(max_depth=max_depth, rng=rng)
                if expr is not None and len(expr.free_symbols) == n_vars:
                    break
            else:
                continue   # gave up — try a new dataset

            # Rename variables to match X columns
            sym_map = {s: sp.Symbol(f"x{i+1}")
                       for i, s in enumerate(sorted(expr.free_symbols,
                                                     key=lambda s: s.name))}
            expr = expr.subs(sym_map)

            y = evaluate_equation(expr, X)
            if not np.isfinite(y).all():
                continue

            datasets.append({
                "expr":      expr,
                "eq_str":    str(expr),
                "X":         X,
                "y":         y,
                "n_vars":    n_vars,
                "dist_types":  inp["dist_types"],
                "sub_ranges":  inp["sub_ranges"],
                "dist_params": inp["dist_params"],
            })
    else:
        # extract production rules from the corpus
        tau, alpha, safe_max_depth = optimise_tau_alpha(equations, n_gen=500, n_trials=100)

        grammar = extract_production_rules_bayesian(equations, alpha=alpha, verbose=False)

        np.seterr(over="ignore")   # before the loop to avoid spurious warnings during generation and evaluation
        datasets = []
        while len(datasets) < n_datasets:
            ds = generate_dataset(
                grammar,
                n_samples=n_samples_per_dataset,
                uniform_ratio=uniform_ratio,
                tau=tau,
                max_depth=safe_max_depth,
                rng=rng
            )
            if ds is not None:
                datasets.append(ds)     
        np.seterr(over="warn")    

    return datasets

def optimise_tau_alpha(
    equations:              list,
    n_gen:                  int   = 100,
    n_trials:               int   = 100,
    max_depth_tolerance:    int   = 2,
    seed:                   int   = 43,
    verbose:                bool  = False,
) -> tuple[int, int, int]:
    """
    Optimise B-PCFG hyperparameters (alpha, tau) using Optuna TPE sampler.

    For each (alpha, tau) candidate:
        1. Extract production rules with Bayesian smoothing (alpha)
        2. Generate n_gen equations with soft-forcing temperature (tau)
        3. Compute fitness = sum(p-values) - sum(KS statistics) + n_passing_features

    Parameters
    ----------
    equations            : list of dicts from load_feynman_csv()
    n_gen                : equations to generate per fitness evaluation
    n_trials             : number of Optuna trials
    max_depth_tolerance  : extra depth levels beyond corpus max depth
    seed                 : random seed

    Returns
    -------
    tau             : int   best soft-forcing temperature
    alpha           : int   best concentration parameter
    safe_max_depth  : int   corpus max depth + tolerance
    """

    SCALAR_COLS = [
        "depth", "n_operators", "n_unique_variables", "n_constants",
        "unary_ratio", "avg_operator_depth",
        "avg_leaf_depth", "branching_factor",
    ]

    # Corpus features — computed once
    corpus_records = []
    for eq in equations:
        if eq["generic"] is None:
            continue
        try:
            corpus_records.append(extract_features(eq["generic"]))
        except Exception:
            continue

    corpus_df        = pd.DataFrame(corpus_records)
    max_depth_corpus = int(corpus_df["depth"].max())
    safe_max_depth   = max_depth_corpus + max_depth_tolerance

    def _generated_features(grammar, tau):
        records  = []
        call_rng = np.random.default_rng(seed)
        attempts = 0
        while len(records) < n_gen and attempts < n_gen * 10:
            attempts += 1
            expr = generate_equation(grammar, tau=tau,
                                     max_depth=safe_max_depth, rng=call_rng)
            if expr is None or not expr.free_symbols:
                continue
            try:
                records.append(extract_features(expr))
            except Exception:
                continue
        return pd.DataFrame(records)

    def _compute_fitness(alpha, tau):
        grammar = extract_production_rules_bayesian(equations, alpha=float(alpha))
        gen_df  = _generated_features(grammar, tau=float(tau))
        if gen_df.empty:
            return -999.0
        total_p, total_ks, n_pass = 0.0, 0.0, 0
        for col in SCALAR_COLS:
            if col not in corpus_df.columns or col not in gen_df.columns:
                continue
            c_vals = corpus_df[col].dropna().values
            g_vals = gen_df[col].dropna().values
            if len(c_vals) < 2 or len(g_vals) < 2:
                continue
            ks_stat, p_val = stats.ks_2samp(c_vals, g_vals)
            total_p  += p_val
            total_ks += ks_stat
            if p_val > 0.05:
                n_pass += 1
        return (total_p*5) - total_ks + (n_pass*15)

    # Optuna study
    best_so_far = [-999.0]

    def objective(trial):
        alpha = trial.suggest_int("alpha", 1,  50)
        tau   = trial.suggest_int("tau",   1,  25)
        f     = _compute_fitness(alpha, tau)

        if f > best_so_far[0]:
            best_so_far[0] = f
            if verbose:
                print(f"  trial {trial.number:4d}  alpha={alpha:4d}  "
                      f"tau={tau:3d}  fitness={f:.4f}  [new best]")
        elif trial.number % 10 == 0 and verbose:
            print(f"  trial {trial.number:4d}  alpha={alpha:4d}  "
                  f"tau={tau:3d}  fitness={f:.4f}")

        return -f   # Optuna minimises

    print(f"\nOptimising B-PCFG hyperparameters "
          f"(n_trials={n_trials}, n_gen={n_gen})...")

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_alpha = study.best_params["alpha"]
    best_tau   = study.best_params["tau"]
    best_fitness = -study.best_value

    # Final evaluation — individual KS results at best params
    best_grammar  = extract_production_rules_bayesian(equations, alpha=best_alpha)
    final_rng     = np.random.default_rng(seed)
    final_records = []
    attempts      = 0
    while len(final_records) < 1000 and attempts < 5000:
        attempts += 1
        expr = generate_equation(best_grammar, tau=best_tau,
                                 max_depth=safe_max_depth, rng=final_rng)
        if expr is None or not expr.free_symbols:
            continue
        try:
            final_records.append(extract_features(expr))
        except Exception:
            continue
    gen_df_final = pd.DataFrame(final_records)

    n_pass = 0
    print(f"\n  Final KS results  (alpha={best_alpha}, tau={best_tau})")
    print(f"  {'Feature':<25} {'KS stat':>8} {'p-value':>10} {'Result':>8}")
    print("  " + "-" * 55)
    for col in SCALAR_COLS:
        if col not in corpus_df.columns or col not in gen_df_final.columns:
            continue
        c_vals = corpus_df[col].dropna().values
        g_vals = gen_df_final[col].dropna().values
        ks_stat, p_val = stats.ks_2samp(c_vals, g_vals)
        passed = p_val >= 0.05
        if passed:
            n_pass += 1
        print(f"  {col:<25} {ks_stat:>8.3f} {p_val:>10.3f} "
              f"{'PASS' if passed else 'FAIL':>8}")

    print(f"\n  {n_pass}/{len(SCALAR_COLS)} features pass KS test")
    print(f"  Best alpha = {best_alpha}")
    print(f"  Best tau   = {best_tau}")

    return int(best_tau), int(best_alpha), safe_max_depth

def validate_generator(
    grammar:     dict,
    equations:   list[dict],
    n_generated: int   = 500,
    tau:         float = 6.0,
    max_depth:   int   = 9,
    p_threshold: float = 0.05,
    plot:        bool  = True,
    save_path:   str   = "results/generator_validation.png",
    rng:         np.random.Generator | None = None,
) -> dict:
    """
    Validate the generator by comparing structural feature distributions
    of generated equations against the Feynman corpus.

    For each scalar feature, runs a two-sample Kolmogorov-Smirnov test.
    A high p-value (> p_threshold) means the distributions are statistically
    indistinguishable -- which is the desired outcome.

    Parameters
    ----------
    grammar     : dict from extract_production_rules()
    equations   : list of dicts from load_feynman_csv() -- the corpus
    n_generated : number of equations to generate for comparison
    tau         : soft-forcing temperature passed to generate_equation()
    max_depth   : hard depth cap passed to generate_equation()
    p_threshold : KS p-value below which a feature is flagged as divergent
    plot        : whether to produce the comparison figure
    save_path   : where to save the figure
    rng         : numpy random Generator

    Returns
    -------
    dict with keys:
        "corpus_df"   : pd.DataFrame  -- corpus feature vectors
        "gen_df"      : pd.DataFrame  -- generated feature vectors
        "ks_results"  : dict  {feature: {"stat": float, "p": float, "pass": bool}}
        "op_corpus"   : Counter       -- pooled operator counts from corpus
        "op_gen"      : Counter       -- pooled operator counts from generated
        "n_generated" : int           -- actual number of equations generated
        "n_failed"    : int           -- equations rejected (None or no vars)
    """

    if rng is None:
        rng = np.random.default_rng()

    SCALAR_COLS = [
        "depth", "n_operators", "n_unique_variables", "n_constants",
        "unary_ratio", "avg_operator_depth",
        "avg_leaf_depth", "branching_factor",
    ]

    # --- Feynman equations, extract features structure ---
    corpus_records = []
    op_corpus = Counter()
    for eq in equations:
        if eq["generic"] is None:
            continue
        try:
            f = extract_features(eq["generic"])
            corpus_records.append(f)
            op_corpus.update(f["operator_counts"])
        except Exception:
            continue
    corpus_df = pd.DataFrame(corpus_records)

    # --- PCFG generation, then extract features structure ---
    gen_records = []
    op_gen      = Counter()
    n_failed    = 0
    attempts    = 0

    while len(gen_records) < n_generated:
        attempts += 1
        if attempts > n_generated * 10:
            break
        expr = generate_equation(grammar, tau=tau, max_depth=max_depth, rng=rng)
        if expr is None or not expr.free_symbols:
            n_failed += 1
            continue
        try:
            f = extract_features(expr)
            gen_records.append(f)
            op_gen.update(f["operator_counts"])
        except Exception:
            n_failed += 1

    gen_df   = pd.DataFrame(gen_records)
    n_actual = len(gen_records)

    # --- KS tests ---
    ks_results = {}
    print(f"\nComparing {len(corpus_df)} corpus equations vs {n_actual} generated")
    print(f"{'Feature':<25} {'KS stat':>8} {'p-value':>10} {'Result':>8}")
    print("-" * 55)
    for col in SCALAR_COLS:
        if col not in corpus_df.columns or col not in gen_df.columns:
            continue
        c_vals = corpus_df[col].dropna().values
        g_vals = gen_df[col].dropna().values
        if len(c_vals) < 2 or len(g_vals) < 2:
            continue
        ks_stat, p_val = stats.ks_2samp(c_vals, g_vals)
        passed = p_val >= p_threshold
        ks_results[col] = {"stat": ks_stat, "p": p_val, "pass": passed}
        flag = "PASS" if passed else "FAIL"
        print(f"  {col:<23} {ks_stat:>8.3f} {p_val:>10.3f} {flag:>8}")

    n_pass = sum(1 for v in ks_results.values() if v["pass"])
    print(f"\n  {n_pass}/{len(ks_results)} features pass KS test (p >= {p_threshold})")

    # --- Operator frequency comparison ---
    all_ops = sorted(set(list(op_corpus.keys()) + list(op_gen.keys())))
    total_c = sum(op_corpus.values()) or 1
    total_g = sum(op_gen.values()) or 1
    print(f"\n{'Operator':<12} {'Corpus %':>10} {'Generated %':>12}")
    print("-" * 36)
    for op in sorted(all_ops, key=lambda o: -op_corpus.get(o, 0)):
        pc = 100 * op_corpus.get(op, 0) / total_c
        pg = 100 * op_gen.get(op, 0) / total_g
        print(f"  {op:<10} {pc:>9.1f}% {pg:>11.1f}%")

    if plot:
        _plot_validation(corpus_df, gen_df, ks_results, op_corpus, op_gen,
                         SCALAR_COLS, save_path)

    return {
        "corpus_df":   corpus_df,
        "gen_df":      gen_df,
        "ks_results":  ks_results,
        "op_corpus":   op_corpus,
        "op_gen":      op_gen,
        "n_generated": n_actual,
        "n_failed":    n_failed,
    }

def _plot_validation(corpus_df, gen_df, ks_results, op_corpus, op_gen,
                     scalar_cols, save_path):
    """comparison of corpus vs generated."""

    BLUE   = "#325A9A"
    ORANGE = "#F17127"
    GREEN  = "#55A868"
    RED    = "#C44E52"

    # Scientific style settings
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         18,
        "axes.titlesize":    18,
        "axes.labelsize":    16,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    valid_cols = [c for c in scalar_cols
                  if c in corpus_df.columns and c in gen_df.columns]
    n_rows = (len(valid_cols) + 3) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows + 4))
    axes = axes.flatten()

    for idx, col in enumerate(valid_cols):
        ax = axes[idx]
        c_vals = corpus_df[col]
        g_vals = gen_df[col]

        # Use integer bins for discrete features, auto bins for continuous
        all_vals = np.concatenate([c_vals, g_vals])
        if np.all(all_vals == all_vals.astype(int)):
            # Discrete integer feature — align bins exactly to integers
            min_val = int(all_vals.min())
            max_val = int(all_vals.max())
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1.0)
        else:
            bins = 15

        ax.hist(c_vals, bins=bins, color=BLUE,   alpha=0.65, label="Corpus",
                density=True, edgecolor="white", linewidth=0.5)
        ax.hist(g_vals, bins=bins, color=ORANGE, alpha=0.65, label="Generated",
                density=True, edgecolor="white", linewidth=0.5)
        
        if np.all(all_vals == all_vals.astype(int)):
            min_val = int(all_vals.min())
            max_val = int(all_vals.max())
            ax.set_xticks(np.arange(min_val, max_val + 1))

        # Grey grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.7)

        res    = ks_results.get(col, {})
        stat   = res.get("stat", float("nan"))
        p      = res.get("p",    float("nan"))
        passed = res.get("pass", False)
        color  = GREEN if passed else RED

        def _format_col_name(col):
            name = col.replace("_", " ")
            if name.startswith("n "):
                name = name[2:]   # remove leading "n "
            return name.capitalize()
        
        ax.set_title(_format_col_name(col).title(), fontweight="bold")
        ax.set_ylabel("Density")
        # ax.legend()
        ax.text(0.97, 0.95, f"KS={stat:.2f}\np={p:.3f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=10,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"))

    # Operator frequency bar chart
    ax = axes[len(valid_cols)]
    all_ops = sorted(set(list(op_corpus.keys()) + list(op_gen.keys())))
    total_c = sum(op_corpus.values()) or 1
    total_g = sum(op_gen.values()) or 1
    top_ops = sorted(all_ops, key=lambda o: -op_corpus.get(o, 0))[:12]
    x = np.arange(len(top_ops))
    w = 0.38

    ax.bar(x - w/2, [100*op_corpus.get(o,0)/total_c for o in top_ops],
           width=w, color=BLUE,   alpha=0.9, label="Corpus",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, [100*op_gen.get(o,0)/total_g    for o in top_ops],
           width=w, color=ORANGE, alpha=0.9, label="Generated",
           edgecolor="white", linewidth=0.5)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(top_ops, rotation=40, ha="right")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Operator Frequencies", fontweight="bold")
    # ax.legend()

    for i in range(len(valid_cols) + 1, len(axes)):
        axes[i].axis("off")

    n_pass = sum(1 for v in ks_results.values() if v["pass"])
    fig.suptitle(
        f"Generator Validation — {n_pass}/{len(ks_results)} features pass KS test",
        fontsize=16, fontweight="bold", y=1.01,
    )

    legend_elements = [
    Patch(facecolor=BLUE,   alpha=0.9, label="Corpus"),
    Patch(facecolor=ORANGE, alpha=0.9, label="Generated"),
]
    fig.legend(handles=legend_elements, loc="lower center",
            ncol=2, fontsize=18, frameon=True,
            bbox_to_anchor=(0.5, -0.02))


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.show()

    # Reset rcParams to defaults after plotting
    plt.rcParams.update(plt.rcParamsDefault)