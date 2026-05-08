import argparse
import csv
import io
from pathlib import Path

import onnx
import onnxruntime as ort

from ep_utils import OVEPManager


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
        default="FP32",
        help="Precision option passed to OVEP when creating the session.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Path to save weights information as CSV file.",
    )
    return parser.parse_args()


def create_ovep_session(model_path: str, device_type: str, precision: str):
    ovep_manager = OVEPManager(device_type=device_type)
    ovep_manager.register_plugin_ep()

    discovered_devices = ovep_manager.get_plugin_ep_devices(ovep_manager.ep_name_, device_type)
    if not discovered_devices:
        ovep_manager.unregister_plugin_ep()
        raise RuntimeError(f"No OVEP devices discovered for device_type={device_type}")

    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices(discovered_devices, {"precision": precision})

    try:
        session = ort.InferenceSession(model_path, sess_options=session_options)
    except Exception:
        ovep_manager.unregister_plugin_ep()
        raise

    return session, ovep_manager, discovered_devices


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
    )

    try:
        print("Loaded model with OVEP")
        print(f"  model      : {model_path}")
        print(f"  device     : {args.device}")
        print(f"  precision  : {args.precision}")
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