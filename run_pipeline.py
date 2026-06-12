"""
Three-model pipeline with ONNX Runtime IO Binding.

Flow:
1) Prepare IO specs for model 0, model 1, and model 2.
2) Run model 0 on CPU.
3) Run model 1 on the OV plugin path.
4) Run model 2 on CPU using model 1 output as input.
"""

import os

import numpy as np
import onnx
import onnxruntime as ort

from ep_utils import OVEPManager
from utils import get_model_io_shapes, bind_np_array




def ensure_models_exist(model0_path="model0.onnx", model1_path="model1.onnx", model2_path="model2.onnx"):
    if not os.path.exists(model0_path) or not os.path.exists(model1_path) or not os.path.exists(model2_path):
        print("Model files missing - regenerating models ...")
        import create_models

        create_models.create_model0(model0_path)
        create_models.create_model1(model1_path)
        create_models.create_model2(model2_path)


def prepare_io_specs(model0_path="model0.onnx", model1_path="model1.onnx", model2_path="model2.onnx"):
    m0_input_shape, m0_output_shape = get_model_io_shapes(model0_path, "X", "Y")
    m1_input_shape, m1_output_shape = get_model_io_shapes(model1_path, "A", "B")
    m2_input_shape, m2_output_shape = get_model_io_shapes(model2_path, "B", "C")

    if m0_output_shape != m1_input_shape:
        raise ValueError(
            f"Shape mismatch: model0 output {m0_output_shape} != model1 input {m1_input_shape}"
        )
    if m1_output_shape != m2_input_shape:
        raise ValueError(
            f"Shape mismatch: model1 output {m1_output_shape} != model2 input {m2_input_shape}"
        )

    x_data = np.linspace(0.0, 1.0, np.prod(m0_input_shape), dtype=np.float32).reshape(m0_input_shape)
    y_data = np.zeros(m0_output_shape, dtype=np.float32)
    b_data = np.zeros(m1_output_shape, dtype=np.float32)
    c_data = np.zeros(m2_output_shape, dtype=np.float32)

    model0_io = {
        "input_name": "X",
        "output_name": "Y",
        "input_array": x_data,
        "output_array": y_data,
    }
    model1_io = {
        "input_name": "A",
        "output_name": "B",
        "input_array": model0_io["output_array"],  # model 1 input is model 0 output
        "output_array": b_data
    }
    model2_io = {
        "input_name": "B",
        "output_name": "C",
        "input_array": model1_io["output_array"],  # model 2 input is model 1 output
        "output_array": c_data,
    }
    return model0_io, model1_io, model2_io




def create_model1_session(model1_path="model1.onnx", provider_mode="CPU"):
    if provider_mode == "CPU":
        return ort.InferenceSession(model1_path, providers=["CPUExecutionProvider"]), None, "CPUExecutionProvider"

    if provider_mode == "OV_ABI_GPU":
        ov_device = "GPU"
        ovep_manager = OVEPManager(device_type=ov_device)
        ovep_manager.register_plugin_ep()

        discovered_devices = ovep_manager.get_plugin_ep_devices(ovep_manager.ep_name_, ov_device)
        if not discovered_devices:
            raise RuntimeError(f"No ABI EP devices discovered for device_type={ov_device}")

        sess1_options = ort.SessionOptions()
        sess1_options.add_provider_for_devices(discovered_devices, {"precision": "FP32"})
        session = ort.InferenceSession(model1_path, sess_options=sess1_options)
        label = f"ABI {discovered_devices[0].ep_name} (ov_device={ov_device})"
        return session, ovep_manager, label

    raise ValueError(f"Unsupported provider_mode={provider_mode}")


if __name__ == "__main__":
    # Configurable model paths
    model0_path = "model0.onnx"
    model1_path = "model1.onnx"
    model2_path = "model2.onnx"
    
    ensure_models_exist(model0_path, model1_path, model2_path)

    print("\n=== Step 1: Prepare IO for all models ===")
    model0_io, model1_io, model2_io = prepare_io_specs(model0_path, model1_path, model2_path)

    print("\n=== Step 2: Run CPU -> OVEP -> CPU pipeline ===")
    
    ov_device = "GPU"
    ovep_manager = OVEPManager(device_type=ov_device)
    ovep_manager.register_plugin_ep()

    discovered_devices = ovep_manager.get_plugin_ep_devices(ovep_manager.ep_name_, ov_device)
    if not discovered_devices:
        raise RuntimeError(f"No ABI EP devices discovered for device_type={ov_device}")


    sess_options = ort.SessionOptions()
    sess_options.add_provider_for_devices(discovered_devices, {"precision": "FP32"})
    sess0 = ort.InferenceSession(model0_path, providers=["CPUExecutionProvider"])
    sess1 = ort.InferenceSession(model1_path, sess_options=sess_options)
    sess2 = ort.InferenceSession(model2_path, providers=["CPUExecutionProvider"])
    label = f"ABI {discovered_devices[0].ep_name} (ov_device={ov_device})"

   
    sess0_io_binding = sess0.io_binding()
    bind_np_array(sess0_io_binding, model0_io["input_name"], model0_io["input_array"], is_input=True)
    bind_np_array(sess0_io_binding, model0_io["output_name"], model0_io["output_array"], is_input=False)
    sess0.run_with_iobinding(sess0_io_binding)

    sess1_io_binding = sess1.io_binding()
    bind_np_array(sess1_io_binding, model1_io["input_name"], model1_io["input_array"], is_input=True)
    bind_np_array(sess1_io_binding, model1_io["output_name"], model1_io["output_array"], is_input=False)
    sess1.run_with_iobinding(sess1_io_binding)

    sess2_io_binding = sess2.io_binding()
    bind_np_array(sess2_io_binding, model2_io["input_name"], model2_io["input_array"], is_input=True)
    bind_np_array(sess2_io_binding, model2_io["output_name"], model2_io["output_array"], is_input=False)
    sess2.run_with_iobinding(sess2_io_binding)

    if ovep_manager is not None:
        ovep_manager.unregister_plugin_ep()

    print("\n  Summary")
    print(f"    Model 0 provider : {sess0.get_providers()}")
    print(f"    Model 1 provider : {sess1.get_providers()}")
    print(f"    Model 2 provider : {sess2.get_providers()}")
    print(f"    Model 1 setup    : {label}")
    print(f"    X -> Model 0 -> Y: {model0_io['input_array']} -> {model0_io['output_array']}")
    print(f"    Y -> Model 1 -> B: {model1_io['input_array']} -> {model1_io['output_array']}")
    print(f"    B -> Model 2 -> C: {model2_io['input_array']} -> {model2_io['output_array']}")
