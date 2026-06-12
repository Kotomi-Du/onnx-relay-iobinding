import argparse
import csv
import io
from pathlib import Path

import onnx
import onnxruntime as ort
import numpy as np

from ep_utils import OVEPManager
from utils import bind_np_array


ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load an ONNX model with the OpenVINO execution provider."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="model1.onnx",
        help="Path to the ONNX model to load.",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device type to use, for example CPU, GPU, or HETERO:NPU,CPU.",
    )
    parser.add_argument(
        "--precision",
        default="FP16",
        help="Precision option passed to OVEP when creating the session.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Path to save weights information as CSV file.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Enable OVEP context compilation mode and skip inference runs.",
    )
    return parser.parse_args()


def create_ovep_session(model_path: str, device_type: str, precision: str, compile_only: bool = False):
    ovep_manager = OVEPManager(device_type=device_type)
    ovep_manager.register_plugin_ep()

    discovered_devices = ovep_manager.get_plugin_ep_devices(ovep_manager.ep_name_, device_type)
    if not discovered_devices:
        ovep_manager.unregister_plugin_ep()
        raise RuntimeError(f"No OVEP devices discovered for device_type={device_type}")

    session_options = ort.SessionOptions()
    session_model = str(model_path)
    if compile_only:
        model_parent = Path(model_path).resolve().parent
        ctx_dir = model_parent / "ep_ctx"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        ctx_name = str(ctx_dir / f"{Path(model_path).stem}.ctx")
        session_options.add_session_config_entry("ep.context_enable", "1")
        session_options.add_session_config_entry("ep.context_file_path", ctx_name)
        session_model = str(model_path)

    session_options.add_provider_for_devices(discovered_devices, {"precision": precision})

    try:
        session = ort.InferenceSession(session_model, sess_options=session_options)
    except Exception:
        ovep_manager.unregister_plugin_ep()
        raise

    return session, ovep_manager, discovered_devices

def _resolve_tensor_shape(shape):
    resolved_shape = []
    for dim in shape:
        # Dynamic dimensions are represented as None/str/<=0 in ORT metadata.
        if isinstance(dim, int) and dim > 0:
            resolved_shape.append(dim)
        else:
            resolved_shape.append(1)
    return resolved_shape


def _build_input_data(shape, dtype):
    element_count = int(np.prod(shape, dtype=np.int64))

    if np.issubdtype(dtype, np.floating):
        return np.linspace(0.0, 1.0, element_count, dtype=dtype).reshape(shape)
    if np.issubdtype(dtype, np.integer):
        return np.arange(element_count, dtype=dtype).reshape(shape)
    if dtype == np.bool_:
        return (np.arange(element_count) % 2 == 0).reshape(shape)

    return np.zeros(shape, dtype=dtype)


def prepare_io_specs(session: ort.InferenceSession):
    input_specs = []
    output_specs = []

    for input_meta in session.get_inputs():
        dtype = ORT_TYPE_TO_NP.get(input_meta.type, np.float32)
        shape = _resolve_tensor_shape(input_meta.shape)
        input_specs.append(
            {
                "name": input_meta.name,
                "dtype": dtype,
                "shape": shape,
                "array": _build_input_data(shape, dtype),
            }
        )

    for output_meta in session.get_outputs():
        dtype = ORT_TYPE_TO_NP.get(output_meta.type, np.float32)
        shape = _resolve_tensor_shape(output_meta.shape)
        output_specs.append(
            {
                "name": output_meta.name,
                "dtype": dtype,
                "shape": shape,
                "array": np.zeros(shape, dtype=dtype),
            }
        )

    if not input_specs:
        raise RuntimeError("Model has no inputs.")
    if not output_specs:
        raise RuntimeError("Model has no outputs.")

    return {"inputs": input_specs, "outputs": output_specs}

def dump_weights_info(model_path: str, csv_output: str = None):
    """Load ONNX model and dump weights information (shape and datatype) in CSV format.
    
    Args:
        model_path: Path to the ONNX model
        csv_output: Optional path to save CSV output. If None, prints to stdout.
    """
    model = onnx.load(model_path)
    
    initializers = model.graph.initializer
    
    if not initializers:
        print("\nWeights Information:")
        print("No weights/initializers found in model.")
        return
    
    # Prepare CSV output
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Index", "Name", "Shape", "DataType"])
    
    # Write weight rows
    for i, initializer in enumerate(initializers, 1):
        name = initializer.name
        dims = list(initializer.dims)
        data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(initializer.data_type, "Unknown")
        
        # Handle dtype name extraction (works with both old and new numpy versions)
        dtype_name = getattr(data_type, 'name', str(data_type)) if data_type != "Unknown" else "Unknown"
        
        writer.writerow([i, name, str(dims), dtype_name])
    
    csv_data = output.getvalue()
    
    # Print to stdout
    print("\nWeights Information (CSV):")
    print(csv_data)
    
    # Save to file if specified
    if csv_output:
        with open(csv_output, 'w', newline='') as f:
            f.write(csv_data)
        print(f"Saved weights information to: {csv_output}")


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session, ovep_manager, discovered_devices = create_ovep_session(
        str(model_path),
        args.device,
        args.precision,
        args.compile_only,
    )

    if args.compile_only:
        print("Compile-only mode enabled. Session created with ep.context options; skipping inference.")
    else:
        sess0_io_binding = session.io_binding()
        model0_io = prepare_io_specs(session)
        for input_spec in model0_io["inputs"]:
            bind_np_array(sess0_io_binding, input_spec["name"], input_spec["array"], is_input=True)
        for output_spec in model0_io["outputs"]:
            bind_np_array(sess0_io_binding, output_spec["name"], output_spec["array"], is_input=False)
        print("Running inference with OVEP session...")
        for i in range(500):
            session.run_with_iobinding(sess0_io_binding)
        print("Inference completed.")


    try:
        print("Loaded model with OVEP")
        print(f"  model      : {model_path}")
        print(f"  device     : {args.device}")
        print(f"  precision  : {args.precision}")
        print(f"  compile_only: {args.compile_only}")
        print(f"  providers  : {session.get_providers()}")
        print(
            "  ep devices : "
            + ", ".join(
                f"{device.ep_name}[ov_device={device.ep_metadata.get('ov_device', 'N/A')}]"
                for device in discovered_devices
            )
        )

        # Dump weights information
        #dump_weights_info(str(model_path), args.csv_output)

    finally:
        ovep_manager.unregister_plugin_ep()


if __name__ == "__main__":
    main()