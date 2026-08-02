# Old System Proof Artifacts

Generated in the current project folder from read-only old-system files under:
`/Users/anishdangol/Desktop/ranjana-app/model-details`

## Rerun status
I attempted to rerun the old evaluator now. It failed while loading the old Keras model because the saved model contains legacy `BatchNormalization` config keys (`renorm`, `renorm_clipping`, `renorm_momentum`) that the currently installed Keras does not accept. The command log is saved as `old_resnet50_rerun.log`, and a screenshot-style summary is `06_old_rerun_environment_failure_screenshot.png`.

## Actual old-model result proof
The old project already contains generated evaluation outputs in `supervisor_results/reports/`. Those report:
- total test samples: 1477
- correct samples: 526
- incorrect samples: 951
- accuracy: 35.61%
- max confidence on a wrong prediction: 99.9988%

## Generated proof images
- `01_old_system_metrics_from_report.png`
- `02_old_high_confidence_wrong_examples.png`
- `03_old_resnet50_confusion_matrix_report.png`
- `03b_old_best_model_final_confusion_matrix_report.png`
- `03c_new_5class_confusion_matrix.png`
- `03d_new_62class_confusion_matrix.png`
- `04_old_gradcam_and_heatmap_contact_sheet.png`
- `05_old_preprocessing_contact_sheet.png`
- `06_old_rerun_environment_failure_screenshot.png`
