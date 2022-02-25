# TODO
- [x] load examples from multiple files (chain together into a single iterable)
- [ ] start training large model on 100k, 1m examplees
- [ ] adjust tensorboard writer global step: batch no, not epoch
- [ ] compute per-seq loss instead of per-tok loss
- [ ] generate training data for discriminator
  - sets of bitmaps: ((B_1, B_2), [01])
    - class 0 if bitmaps drawn from different programs; class 1 if bitmaps drawn from the same program 
