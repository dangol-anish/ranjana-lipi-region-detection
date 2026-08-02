# Controlled Demo Accuracy Diagrams - Talk Track

## 01_controlled_demo_success_rates.png
Use this as the main demo accuracy slide.
- The final controlled flawed demo set gives 5/5 broad-band matches (100%).
- The known-good demo set gives 5/5 clean results (100%).
- The canvas demo gives 1/1 clean/high result (100%).
- What to say: "For the controlled demonstration, the selected validated samples behave exactly as expected: flawed samples point to the intended broad region, while good samples remain clean."

## 02_good_sample_false_positive_rates.png
Use this to justify threshold calibration.
- Dataset: 100 real correct processed samples, not reference-bank images and not augmented images.
- Fine-grid false positives: 6/100 (6.0%).
- Broad-band false positives: 3/100 (3.0%).
- What to say: "The z-score threshold was calibrated to reduce false warnings on genuinely correct handwriting. Broad feedback is more stable, while fine-grid feedback is intentionally more sensitive."

## 03_region_feedback_accuracy_summary.png
Use this to be honest about validation versus controlled demo.
- Full flawed validation #1 broad-band match: 5/15 (33.3%).
- Full flawed validation top-2 broad-band match: 10/15 (66.7%).
- Final controlled flawed demo #1 broad-band match: 5/5 (100%).
- What to say: "The general flawed validation shows the limitation of region localization on ambiguous samples, so the final demo uses confirmed controlled samples where the intended flaw is clear."

## 04_recognizer_accuracy_comparison.png
Use this for old-vs-new model improvement.
- Old ResNet50 baseline: 35.61%.
- New validated 5-class CNN recognizer: 100.00%.
- New general 62-class CNN recognizer: 99.52%.
- What to say: "The old model was a large RGB classifier with poor reliability. The new recognizer uses corrected normalization and a lightweight PyTorch CNN, which is better suited to this handwriting dataset."

## 05_controlled_demo_score_distribution.png
Use this to explain score behavior.
- Good samples generally score in the low-to-mid 90s.
- Flawed photo samples can score lower, while synthetic region flaws may still have high overall score but correct regional flags.
- What to say: "The percentage score is global reconstruction quality, while the region result is the important coaching output. A sample can have a high overall score but still have one localized region flagged."
