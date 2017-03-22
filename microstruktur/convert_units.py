"""Utils for converting between different units."""
import ast
import operator as op

units = {
    'm': 1.,
    's': 1.,
    'T': 1.,
}
prefixes = {
    'c': 1e-2,
    'm': 1e-3,
    'u': 1e-6,
    u"\u03BC": 1e-6,
    'n': 1e-10,
}

operators = {ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow,
             ast.USub: op.neg}


def eval_expr(expr):
    """Evaluate a unit expression to it's S.I. factor."""
    return eval_num(ast.parse(expr, mode='eval').body)


def eval_num(node):
    """Evaluate an AST expression node to it's S.I. factor."""
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](eval_num(node.operand))
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](
            eval_num(node.left),
            eval_num(node.right)
        )
    elif isinstance(node, ast.Name):
        strval = node.id
        if len(strval) > 2:
            raise ValueError('unit not recognised %s' % strval)
        try:
            if len(strval) == 1:
                return units[strval]
            else:
                return prefixes[strval[0]] * units[strval[1]]
        except KeyError:
            raise ValueError('unit not reconised %s' % strval)
    else:
        raise TypeError(node)


def unit_conversion_factor(units_in, units_out):
    """Given to unit expressions returns the conversion factor.

    Warning, it doesn't control for correctness of the units yet.
    """
    units_in_si = eval_expr(units_in)
    units_out_si = eval_expr(units_out)
    return units_in_si / units_out_si


def unit_conversion_factor_from_SI(units_out):
    """Return the conversion factor from SI to the specified unit."""
    units_out_si = eval_expr(units_out)
    return 1. / units_out_si


def unit_conversion_factor_to_SI(units_in):
    """Return the conversion factor to SI from the specified unit."""
    units_in_si = eval_expr(units_in)
    return units_in_si
