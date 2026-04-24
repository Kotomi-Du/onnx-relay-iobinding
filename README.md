# ONNX Relay: Two-Stage IOBinding Pipeline

## Steps to run
1. run `python run_pipeline.py`
2. it will create two models locally and run, you will see the log `OVEP-----inputs [4 elements]: -0.25 0.25 0.75 1.25 `

> Tip: OVEP log is present after adding the code below before https://github.com/intel-innersource/frameworks.ai.onnxruntime.openvino-plugin-ep/blob/develop/plugin_impl/ov_compute.cc#L190

```
 if(1) //input_info.name == "input_hidden_states")
      {
        auto tensor = context.GetInput(input_info.onnx_index);
        auto data_ptr = tensor.GetTensorRawData();
        size_t num_elements = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        size_t print_count = std::min(num_elements, static_cast<size_t>(20));

        std::ostringstream oss;
        oss << "input [" << num_elements << " elements]: ";
        if (input_info.type == ov::element::f32) {
          const float* fp = static_cast<const float*>(data_ptr);
          for (size_t j = 0; j < print_count; ++j) {
            oss << fp[j] << " ";
          }
        } else if (input_info.type == ov::element::f16) {
          const uint16_t* hp = static_cast<const uint16_t*>(data_ptr);
          for (size_t j = 0; j < print_count; ++j) {
            // Decode IEEE 754 half-precision to float for display
            uint16_t h = hp[j];
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            float val;
            if (exp == 0) {
              val = std::ldexp(static_cast<float>(mant), -24);
            } else if (exp == 31) {
              val = mant ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
            } else {
              val = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
            }
            if (sign) val = -val;
            oss << val << " ";
          }
        } else {
          oss << "(unsupported element type: " << input_info.type.get_type_name() << ")";
        }
        if (num_elements > print_count) oss << "...";
        std::cout << oss.str() << std::endl;
      }
```
## update code to reproduce iobinding issue
1. move `sess0.run_with_iobinding(sess0_io_binding)` after sess1's iobinding; before `sess1.run_with_iobinding(sess1_io_binding)`
2. run `python run_pipeline.py`
3. you will see the log `OVEP-----inputs [4 elements]: 0 0 0 0 ` which is incorrect.




