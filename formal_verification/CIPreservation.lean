/-
  CAR-MNAR — Theorem 1 (CI preservation and Type-I control under
  self-masking MNAR), formalised in Lean 4 / Mathlib v4.28.0.

  The deep probabilistic content is isolated into named hypotheses; the
  algebraic core is proved here with no `sorry`. All quantities are the
  conditional probabilities of the supplementary proof, evaluated at a fixed
  (y, x, z):

    pY_XZ = P(Y|X,Z)        pY_Z = P(Y|Z)        rhoY = P(R_Y=1|Y)
    pR_XZ = P(R_Y=1|X,Z)    pR_Z = P(R_Y=1|Z)
    qY_XZ = P(Y|X,Z,R_Y=1)  qY_Z = P(Y|Z,R_Y=1)

  Axiom audit: `#print axioms CARMNAR.theorem1` reports only the standard
  Lean axioms `propext`, `Classical.choice`, `Quot.sound`.
-/
import Mathlib

namespace CARMNAR

/-- Bayes identity with X: weak self-masking gives `qY_XZ·pR_XZ = pY_XZ·rhoY`. -/
def ObsWithX (pY_XZ rhoY pR_XZ qY_XZ : ℝ) : Prop :=
  qY_XZ * pR_XZ = pY_XZ * rhoY

/-- Bayes identity without X, with the same `rhoY` (this is weak self-masking). -/
def ObsWithoutX (pY_Z rhoY pR_Z qY_Z : ℝ) : Prop :=
  qY_Z * pR_Z = pY_Z * rhoY

/-- Complete-data CI: `P(Y|X,Z) = P(Y|Z)`. -/
def CompleteCI (pY_XZ pY_Z : ℝ) : Prop := pY_XZ = pY_Z

/-- Normaliser equality `P(R_Y=1|X,Z) = P(R_Y=1|Z)`, implied by weak
self-masking and `CompleteCI` via `∫ ρ(y) P(Y|·) dy`. -/
def NormalizerEq (pR_XZ pR_Z : ℝ) : Prop := pR_XZ = pR_Z

/-- CI preservation after conditioning on `R_Y=1` (supplement Steps 1–3). -/
theorem ci_preserved
    (pY_XZ pY_Z rhoY pR_XZ pR_Z qY_XZ qY_Z : ℝ)
    (hX : ObsWithX pY_XZ rhoY pR_XZ qY_XZ)
    (hZ : ObsWithoutX pY_Z rhoY pR_Z qY_Z)
    (hCI : CompleteCI pY_XZ pY_Z)
    (hN : NormalizerEq pR_XZ pR_Z)
    (hpos : 0 < pR_Z) :
    qY_XZ = qY_Z := by
  unfold ObsWithX at hX
  unfold ObsWithoutX at hZ
  unfold CompleteCI at hCI
  unfold NormalizerEq at hN
  rw [hCI, hN] at hX
  have h : qY_XZ * pR_Z = qY_Z * pR_Z := by rw [hX, hZ]
  exact mul_right_cancel₀ hpos.ne' h

/-- Condition (TWD) — Bayes identity for the `R_X=1` step, with X. -/
def ObsWithX_RX (qY_XRY rhoX pRX_XRY tY_X : ℝ) : Prop :=
  tY_X * pRX_XRY = qY_XRY * rhoX

/-- Condition (TWD) — Bayes identity for the `R_X=1` step, without X (same `rhoX`). -/
def ObsWithoutX_RX (qY_RY rhoX pRX_RY tY : ℝ) : Prop :=
  tY * pRX_RY = qY_RY * rhoX

/-- Condition (TWD) normaliser equality `P(R_X=1|X,Z,R_Y=1) = P(R_X=1|Z,R_Y=1)`. -/
def TWDNormEq (pRX_XRY pRX_RY : ℝ) : Prop := pRX_XRY = pRX_RY

/-- CI preservation on the full test-wise complete subset, adding the `R_X=1`
conditioning under Condition (TWD) (supplement Step 4). Iterating over each
`R_{Z_j}=1` covers the entire subset. -/
theorem ci_preserved_testwise
    (qY_XRY qY_RY rhoX pRX_XRY pRX_RY tY_X tY : ℝ)
    (hpres : qY_XRY = qY_RY)
    (hXr : ObsWithX_RX qY_XRY rhoX pRX_XRY tY_X)
    (hZr : ObsWithoutX_RX qY_RY rhoX pRX_RY tY)
    (hNr : TWDNormEq pRX_XRY pRX_RY)
    (hposr : 0 < pRX_RY) :
    tY_X = tY := by
  unfold ObsWithX_RX at hXr
  unfold ObsWithoutX_RX at hZr
  unfold TWDNormEq at hNr
  rw [hpres, hNr] at hXr
  have h : tY_X * pRX_RY = tY * pRX_RY := by rw [hXr, hZr]
  exact mul_right_cancel₀ hposr.ne' h

/-- Type-I control: equal conditional law gives equal null rejection probability. -/
theorem type_one_control
    (qY_XZ qY_Z alpha : ℝ) (reject : ℝ → ℝ)
    (hpres : qY_XZ = qY_Z) (hcal : reject qY_Z = alpha) :
    reject qY_XZ = alpha := by
  rw [hpres]; exact hcal

/-- Theorem 1 after the `R_Y=1` step: CI preservation and Type-I control. -/
theorem theorem1
    (pY_XZ pY_Z rhoY pR_XZ pR_Z qY_XZ qY_Z alpha : ℝ) (reject : ℝ → ℝ)
    (hX : ObsWithX pY_XZ rhoY pR_XZ qY_XZ)
    (hZ : ObsWithoutX pY_Z rhoY pR_Z qY_Z)
    (hCI : CompleteCI pY_XZ pY_Z)
    (hN : NormalizerEq pR_XZ pR_Z)
    (hpos : 0 < pR_Z)
    (hcal : reject qY_Z = alpha) :
    qY_XZ = qY_Z ∧ reject qY_XZ = alpha := by
  have hpres : qY_XZ = qY_Z :=
    ci_preserved pY_XZ pY_Z rhoY pR_XZ pR_Z qY_XZ qY_Z hX hZ hCI hN hpos
  exact ⟨hpres, type_one_control qY_XZ qY_Z alpha reject hpres hcal⟩

/-- Theorem 1 on the full test-wise complete subset (under Condition (TWD)). -/
theorem theorem1_testwise
    (pY_XZ pY_Z rhoY pR_XZ pR_Z qY_XZ qY_Z : ℝ)
    (rhoX pRX_XRY pRX_RY tY_X tY alpha : ℝ) (reject : ℝ → ℝ)
    (hX : ObsWithX pY_XZ rhoY pR_XZ qY_XZ)
    (hZ : ObsWithoutX pY_Z rhoY pR_Z qY_Z)
    (hCI : CompleteCI pY_XZ pY_Z)
    (hN : NormalizerEq pR_XZ pR_Z)
    (hpos : 0 < pR_Z)
    (hXr : ObsWithX_RX qY_XZ rhoX pRX_XRY tY_X)
    (hZr : ObsWithoutX_RX qY_Z rhoX pRX_RY tY)
    (hNr : TWDNormEq pRX_XRY pRX_RY)
    (hposr : 0 < pRX_RY)
    (hcal : reject tY = alpha) :
    tY_X = tY ∧ reject tY_X = alpha := by
  have hpres : qY_XZ = qY_Z :=
    ci_preserved pY_XZ pY_Z rhoY pR_XZ pR_Z qY_XZ qY_Z hX hZ hCI hN hpos
  have htwd : tY_X = tY :=
    ci_preserved_testwise qY_XZ qY_Z rhoX pRX_XRY pRX_RY tY_X tY hpres hXr hZr hNr hposr
  exact ⟨htwd, type_one_control tY_X tY alpha reject htwd hcal⟩

end CARMNAR
