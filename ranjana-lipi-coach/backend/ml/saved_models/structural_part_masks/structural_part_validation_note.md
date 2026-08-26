# Structural Part-Mask Validation Note

The validated 5-character feedback path now uses curated required structural parts.
Each part mask is measured with:

```text
coverage = covered_required_part_pixels / total_required_part_pixels
missing_score = 1 - coverage
```

Only parts marked `required: true` in `parts.json` affect the score and problem
regions. Style-dependent/advisory parts remain visible in `structural_parts`, but
they do not lower the score or create region warnings.

Latest validation output:

```text
Final_demo_images/21_structural_part_curated_v2_validation/structural_part_validation_5_curated_v2.csv
Good samples flagged: 3/25
Flawed samples intended broad #1 or blocked: 14/15
```

The remaining flawed miss is `data/FlawedValidation/aa/aa-top.jpeg`. Debug output
shows the top-head part coverage is `0.917`, meaning the image still contains
enough top-stroke ink for the current structural mask. It should not be used as a
clean missing-top proof sample.

The remaining good false positives are `a` samples where the taught top-head stroke
is absent under the current mask definition. This is a deliberate taught-form
coaching tradeoff: the app can warn when the taught top part is missing, but some
real handwriting styles that omit that part will be flagged.
