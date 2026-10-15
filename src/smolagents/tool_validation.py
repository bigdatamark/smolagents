# [MODULE DOC]
# Description: Static validation of Tool classes so they can be serialized and rebuilt from source. Provides
#   MethodChecker (undefined-name detection in method ASTs) and validate_tool_attributes (class-level checks).
# Inputs: depends on ast, builtins, itertools.zip_longest and smolagents.utils (BASE_BUILTIN_MODULES,
#   get_source, is_valid_name).
# Outputs: exports MethodChecker, validate_tool_attributes and the private _BUILTIN_NAMES set (names of all
#   builtins).
# Exceptions: none raised directly.
# Side effects: none at import.
import ast
import builtins
from itertools import zip_longest

from .utils import BASE_BUILTIN_MODULES, get_source, is_valid_name


_BUILTIN_NAMES = set(vars(builtins))


class MethodChecker(ast.NodeVisitor):
    # [CLASS DOC]
    # Description: AST visitor that inspects one method's source tree and records every name that is loaded or
    #   called without being defined locally, as an argument, as a class attribute, as an import, as a
    #   builtin, as a base builtin module, as a typing name (Any only) or as a class defined inside the
    #   method. Used by validate_tool_attributes to ensure Tool methods are self-contained so their source can
    #   be serialized.
    # Inputs: class_attributes (set of str, names of the tool's class-level attributes treated as defined);
    #   check_imports (bool, default True, stored only). Key attributes: undefined_names (unused set), imports
    #   (dict alias to module name), from_imports (dict alias to module/name pair), assigned_names (set),
    #   arg_names (set), class_attributes, errors (list of error strings), check_imports, typing_names,
    #   defined_classes.
    # Outputs: after visiting, the errors list holds messages of the form Name '<id>' is undefined.
    # Exceptions: none raised directly.
    # Side effects: mutates its own tracking sets/dicts while visiting.
    # KNOWN ISSUE: check_imports is stored but never consulted, and nothing checks for local (non-package)
    #   imports even though the class docstring says it does; undefined_names is never populated.
    # KNOWN ISSUE: Name tracking is flow-insensitive: a name assigned anywhere in the method (even after use)
    #   counts as defined.
    """
    Checks that a method
    - only uses defined names
    - contains no local imports (e.g. numpy is ok but local_script is not)
    """

    def __init__(self, class_attributes: set[str], check_imports: bool = True):
        # [FUNCTION DOC]
        # Description: Initializes empty tracking containers and stores the given class attributes and flag.
        # Inputs: class_attributes (set of str); check_imports (bool, default True).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets instance attributes; typing_names starts as a set containing only Any.
        self.undefined_names = set()
        self.imports = {}
        self.from_imports = {}
        self.assigned_names = set()
        self.arg_names = set()
        self.class_attributes = class_attributes
        self.errors = []
        self.check_imports = check_imports
        self.typing_names = {"Any"}
        self.defined_classes = set()

    def visit_arguments(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Records the names of a function's positional arguments plus its variadic
        #   positional and keyword arguments as defined argument names.
        # Inputs: node (ast.arguments).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: replaces arg_names with the new set (it does not accumulate).
        # KNOWN ISSUE: Positional-only and keyword-only arguments are ignored, so using them is reported as
        #   undefined.
        # KNOWN ISSUE: Because arg_names is replaced rather than extended, a nested function or lambda
        #   overwrites the outer function's argument names, and names from children visited after it may be
        #   misjudged. Child nodes (annotations, defaults) are not visited.
        """Collect function arguments"""
        self.arg_names = {arg.arg for arg in node.args}
        if node.kwarg:
            self.arg_names.add(node.kwarg.arg)
        if node.vararg:
            self.arg_names.add(node.vararg.arg)

    def visit_Import(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Records each imported module under its alias (or its full dotted name if no
        #   alias).
        # Inputs: node (ast.Import).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds entries to imports.
        # KNOWN ISSUE: For 'import a.b' the recorded key is the dotted string, while the loaded name is just
        #   'a', so later uses of 'a' are flagged undefined.
        for name in node.names:
            actual_name = name.asname or name.name
            self.imports[actual_name] = name.name

    def visit_ImportFrom(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Records each from-imported name under its alias with its source module and
        #   original name (module is empty string for relative imports with no module).
        # Inputs: node (ast.ImportFrom).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds entries to from_imports.
        module = node.module or ""
        for name in node.names:
            actual_name = name.asname or name.name
            self.from_imports[actual_name] = (module, name.name)

    def visit_Assign(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Marks plain-name targets and names inside tuple/list targets as assigned, then
        #   visits the assigned value.
        # Inputs: node (ast.Assign).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names; may append errors via visiting the value.
        # KNOWN ISSUE: Targets that are subscripts, attributes or starred expressions are not visited, so
        #   undefined names used inside them (for example an index variable) are never checked; nested tuple
        #   targets are not unpacked.
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)
        self.visit(node.value)

    def visit_With(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Marks a plain-name 'as' alias of each with-item as assigned, then visits all
        #   children.
        # Inputs: node (ast.With).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        # KNOWN ISSUE: Async with statements and tuple aliases are not handled.
        """Track aliases in 'with' statements (the 'y' in 'with X as y')"""
        for item in node.items:
            if item.optional_vars:  # This is the 'y' in 'with X as y'
                if isinstance(item.optional_vars, ast.Name):
                    self.assigned_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Marks the exception alias name (if any) as assigned, then visits all children.
        # Inputs: node (ast.ExceptHandler).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        """Track exception aliases (the 'e' in 'except Exception as e')"""
        if node.name:  # This is the 'e' in 'except Exception as e'
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Marks a plain-name annotated-assignment target as assigned and visits the value
        #   if present (the annotation itself is not visited).
        # Inputs: node (ast.AnnAssign).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        """Track annotated assignments."""
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Marks a plain-name or tuple-of-names loop target as assigned, then visits all
        #   children.
        # Inputs: node (ast.For).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        # KNOWN ISSUE: List-shaped and nested loop targets, and async for loops, are not handled.
        target = node.target
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators):
        # [FUNCTION DOC]
        # Description: HELPER: For each comprehension generator, marks a plain-name or tuple-of-names target
        #   as assigned.
        # Inputs: generators (list of ast.comprehension).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        """Helper method to handle generators in all types of comprehensions"""
        for generator in generators:
            if isinstance(generator.target, ast.Name):
                self.assigned_names.add(generator.target.id)
            elif isinstance(generator.target, ast.Tuple):
                for elt in generator.target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_ListComp(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Registers list-comprehension loop variables as assigned, then visits all
        #   children.
        # Inputs: node (ast.ListComp).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        # KNOWN ISSUE: Generator expressions have no handler, so their loop variables are reported as
        #   undefined.
        """Track variables in list comprehensions"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Registers dict-comprehension loop variables as assigned, then visits all
        #   children.
        # Inputs: node (ast.DictComp).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        """Track variables in dictionary comprehensions"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Registers set-comprehension loop variables as assigned, then visits all
        #   children.
        # Inputs: node (ast.SetComp).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to assigned_names.
        """Track variables in set comprehensions"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Skips attribute accesses whose base is the plain name self (so self attributes
        #   are never checked); otherwise visits children normally.
        # Inputs: node (ast.Attribute).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: none beyond child visits.
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: Records the name of a class defined inside the method, then visits its body.
        # Inputs: node (ast.ClassDef).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: adds to defined_classes.
        """Track class definitions"""
        self.defined_classes.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: For a loaded name, appends an undefined-name error unless it is a builtin, base
        #   builtin module, argument, self, class attribute, import, from-import, assigned name, typing name
        #   or locally defined class.
        # Inputs: node (ast.Name).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: may append to errors.
        # KNOWN ISSUE: Names declared global/nonlocal, walrus targets, and names defined in the module around
        #   the tool are not recognized, which yields false positives; a name used several times produces
        #   duplicate errors.
        if isinstance(node.ctx, ast.Load):
            if not (
                node.id in _BUILTIN_NAMES
                or node.id in BASE_BUILTIN_MODULES
                or node.id in self.arg_names
                or node.id == "self"
                or node.id in self.class_attributes
                or node.id in self.imports
                or node.id in self.from_imports
                or node.id in self.assigned_names
                or node.id in self.typing_names
                or node.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node):
        # [FUNCTION DOC]
        # Description: HELPER: When the call target is a plain name, appends an undefined-name error under the
        #   same rules as visit_Name (minus typing names); then visits all children, so the function name is
        #   also checked again by visit_Name.
        # Inputs: node (ast.Call).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: may append to errors.
        # KNOWN ISSUE: An undefined called name gets two identical error messages (once here, once in
        #   visit_Name).
        if isinstance(node.func, ast.Name):
            if not (
                node.func.id in _BUILTIN_NAMES
                or node.func.id in BASE_BUILTIN_MODULES
                or node.func.id in self.arg_names
                or node.func.id == "self"
                or node.func.id in self.class_attributes
                or node.func.id in self.imports
                or node.func.id in self.from_imports
                or node.func.id in self.assigned_names
                or node.func.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.func.id}' is undefined.")
        self.generic_visit(node)


def validate_tool_attributes(cls, check_imports: bool = True) -> None:
    # [FUNCTION DOC]
    # Description: Checks that a Tool subclass can be safely serialized from source. It parses the class
    #   source (via get_source), requires the first top-level statement to be a class, and collects errors:
    #   invalid 'name' class attribute (non-constant, non-string or not a valid identifier), class attributes
    #   whose value contains anything other than constants/dicts/lists/sets, __init__ parameters without
    #   defaults, __init__ defaults that are not literals, and undefined names in each method (via
    #   MethodChecker with the class attributes as known names). All errors are joined into one message.
    # Inputs: cls (a Tool class whose source is retrievable); check_imports (bool, default True, forwarded to
    #   MethodChecker where it has no effect).
    # Outputs: None if no problem is found.
    # Exceptions: ValueError if the source does not start with a class, or with message 'Tool validation
    #   failed for <ClassName>:' followed by one line per error; errors from get_source (e.g.
    #   OSError/TypeError) propagate.
    # Side effects: none (reads source code).
    # KNOWN ISSUE: The docstring says class attributes should only be strings or dicts, but the code also
    #   accepts other constants, lists and sets.
    # KNOWN ISSUE: Because ast.walk also yields context nodes, a class attribute whose value is a list literal
    #   contains a Load node and is wrongly flagged as complex; negative numbers and tuples are flagged too.
    # KNOWN ISSUE: Only synchronous methods directly in the class body are checked; async methods are skipped
    #   and their assignments are counted as class attributes. Annotated class attributes are not tracked.
    # KNOWN ISSUE: Error ordering from set-based collections is non-deterministic.
    """
    Validates that a Tool class follows the proper patterns:
    0. Any argument of __init__ should have a default.
    Args chosen at init are not traceable, so we cannot rebuild the source code for them, thus any important arg should be defined as a class attribute.
    1. About the class:
        - Class attributes should only be strings or dicts
        - Class attributes cannot be complex attributes
    2. About all class methods:
        - Imports must be from packages, not local files
        - All methods must be self-contained

    Raises all errors encountered, if no error returns None.
    """

    class ClassLevelChecker(ast.NodeVisitor):
        # [CLASS DOC]
        # Description: HELPER: AST visitor over the tool's class node that collects class attribute names,
        #   complex attributes, invalid name attributes and __init__ parameter problems.
        # Inputs: none. Attributes: imported_names (unused), complex_attributes, class_attributes,
        #   non_defaults, non_literal_defaults (sets of str), in_method (bool), invalid_attributes (list of
        #   str).
        # Outputs: results are read from its attributes after visiting.
        # Exceptions: none raised directly.
        # Side effects: mutates its own attributes.
        def __init__(self):
            # [FUNCTION DOC]
            # Description: HELPER: Initializes the empty collections and the in_method flag to False.
            # Inputs: none.
            # Outputs: none.
            # Exceptions: none raised directly.
            # Side effects: sets instance attributes.
            self.imported_names = set()
            self.complex_attributes = set()
            self.class_attributes = set()
            self.non_defaults = set()
            self.non_literal_defaults = set()
            self.in_method = False
            self.invalid_attributes = []

        def visit_FunctionDef(self, node):
            # [FUNCTION DOC]
            # Description: HELPER: Checks __init__ parameters when the function is __init__, then visits the
            #   function body with in_method set so assignments inside methods are not treated as class
            #   attributes, restoring the previous flag afterwards.
            # Inputs: node (ast.FunctionDef).
            # Outputs: none.
            # Exceptions: none raised directly.
            # Side effects: may add to non_defaults/non_literal_defaults; toggles in_method.
            # KNOWN ISSUE: Any nested function named __init__ (for example in a nested class) is also checked.
            if node.name == "__init__":
                self._check_init_function_parameters(node)
            old_context = self.in_method
            self.in_method = True
            self.generic_visit(node)
            self.in_method = old_context

        def visit_Assign(self, node):
            # [FUNCTION DOC]
            # Description: HELPER: Ignored inside methods. At class level, records plain-name targets as class
            #   attributes, flags them as complex if any node in the value is not a constant/dict/list/set
            #   node, and, when the first target is 'name', validates that the value is a string constant that
            #   is a valid identifier and not a keyword.
            # Inputs: node (ast.Assign).
            # Outputs: none.
            # Exceptions: none raised directly.
            # Side effects: adds to class_attributes, complex_attributes and invalid_attributes.
            # KNOWN ISSUE: A list literal value is considered complex because its Load context node fails the
            #   type check.
            if self.in_method:
                return
            # Track class attributes
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_attributes.add(target.id)

            # Check if the assignment is more complex than simple literals
            if not all(isinstance(val, (ast.Constant, ast.Dict, ast.List, ast.Set)) for val in ast.walk(node.value)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.complex_attributes.add(target.id)

            # Check specific class attributes
            if getattr(node.targets[0], "id", "") == "name":
                if not isinstance(node.value, ast.Constant):
                    self.invalid_attributes.append(f"Class attribute 'name' must be a constant, found '{node.value}'")
                elif not isinstance(node.value.value, str):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a string, found '{node.value.value}'"
                    )
                elif not is_valid_name(node.value.value):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found '{node.value.value}'"
                    )

        def _check_init_function_parameters(self, node):
            # [FUNCTION DOC]
            # Description: HELPER: Pairs __init__ positional parameters with their defaults (aligned from the
            #   end); parameters other than self with no default are recorded as non-defaults, and defaults
            #   that are not constant/dict/list/set nodes are recorded as non-literal.
            # Inputs: node (ast.FunctionDef for __init__).
            # Outputs: none.
            # Exceptions: none raised directly.
            # Side effects: adds to non_defaults and non_literal_defaults.
            # KNOWN ISSUE: Keyword-only and positional-only parameters are never checked, nor are variadic
            #   parameters.
            # Check defaults in parameters
            for arg, default in reversed(list(zip_longest(reversed(node.args.args), reversed(node.args.defaults)))):
                if default is None:
                    if arg.arg != "self":
                        self.non_defaults.add(arg.arg)
                elif not isinstance(default, (ast.Constant, ast.Dict, ast.List, ast.Set)):
                    self.non_literal_defaults.add(arg.arg)

    class_level_checker = ClassLevelChecker()
    source = get_source(cls)
    tree = ast.parse(source)
    class_node = tree.body[0]
    if not isinstance(class_node, ast.ClassDef):
        raise ValueError("Source code must define a class")
    class_level_checker.visit(class_node)

    errors = []
    # Check invalid class attributes
    if class_level_checker.invalid_attributes:
        errors += class_level_checker.invalid_attributes
    if class_level_checker.complex_attributes:
        errors.append(
            f"Complex attributes should be defined in __init__, not as class attributes: "
            f"{', '.join(class_level_checker.complex_attributes)}"
        )
    if class_level_checker.non_defaults:
        errors.append(
            f"Parameters in __init__ must have default values, found required parameters: "
            f"{', '.join(class_level_checker.non_defaults)}"
        )
    if class_level_checker.non_literal_defaults:
        errors.append(
            f"Parameters in __init__ must have literal default values, found non-literal defaults: "
            f"{', '.join(class_level_checker.non_literal_defaults)}"
        )

    # Run checks on all methods
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            method_checker = MethodChecker(class_level_checker.class_attributes, check_imports=check_imports)
            method_checker.visit(node)
            errors += [f"- {node.name}: {error}" for error in method_checker.errors]

    if errors:
        raise ValueError(f"Tool validation failed for {cls.__name__}:\n" + "\n".join(errors))
    return
