# ONNX Relay: Two-Stage IOBinding Pipeline

## Steps to run
1. run `python run_pipeline.py`
2. it will create two models locally and run, you will see the log `OVEP-----inputs [4 elements]: -0.25 0.25 0.75 1.25 `

## update code to reproduce iobinding issue
1. move `sess0.run_with_iobinding(sess0_io_binding)` after sess1's iobinding; before `sess1.run_with_iobinding(sess1_io_binding)`
2. run `python run_pipeline.py`
3. you will see the log `OVEP-----inputs [4 elements]: 0 0 0 0 ` which is incorrect.


