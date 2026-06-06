# Lean formalisation of CAR-MNAR Theorem 1

A self-contained Lean 4 project (Mathlib v4.28.0) that machine-verifies the
algebraic core of Theorem 1: under weak self-masking, positivity, complete-data
conditional independence, and Condition (TWD), the residual-independence test on
the test-wise complete subset preserves conditional independence and controls the
Type-I error.

## File
`CIPreservation.lean` — five theorems, all `sorry`-free:
`ci_preserved` (Steps 1–3), `ci_preserved_testwise` (Condition (TWD), Step 4),
`type_one_control` (Type-I control), and the assembled `theorem1` /
`theorem1_testwise`.

## Build and audit
```
lake exe cache get
lake build
```
The axiom audit `#print axioms CARMNAR.theorem1` reports only the standard Lean
axioms `propext`, `Classical.choice`, `Quot.sound` — no `sorry`, no
user-introduced axioms.
