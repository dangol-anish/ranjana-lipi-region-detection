# Reference Image Caveat

Initial good-sample sanity testing showed that the raw reference bank images scored notably worse than real processed dataset samples. The five reference images averaged about 63.94 overall reconstruction score, while sampled real processed dataset images averaged about 93.37.

This is likely because the autoencoders were trained on normalized and augmented real dataset images, not directly on the raw reference bank images. The reference images may also differ in lighting, capture conditions, stroke thickness, scale, or background texture.

Because of this, raw reference bank images should not be used as proof that a "correct" sample scores cleanly. For validation and demo purposes, use real processed dataset samples from `backend/ml/processed/<class_name>/` as known-good ground truth instead.
