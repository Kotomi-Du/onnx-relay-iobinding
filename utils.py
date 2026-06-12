import onnx
import onnxruntime as ort
import numpy as np

def value_info_shape(value_info):
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def get_model_io_shapes(model_path, input_name, output_name):
    model = onnx.load(model_path)
    input_shape = value_info_shape(next(value for value in model.graph.input if value.name == input_name))
    output_shape = value_info_shape(next(value for value in model.graph.output if value.name == output_name))
    return input_shape, output_shape


def bind_np_array(io_binding: ort.IOBinding, name: str, array: np.ndarray, is_input: bool=True, ep_device = None):
    assert array.flags["C_CONTIGUOUS"], f"Provided array {name} is not contigous and it cannot be used in io bindings"
    args = [name, "cpu", 0, array.dtype, array.shape, array.ctypes.data]
    if is_input:
        io_binding.bind_input(*args)
    else:
        io_binding.bind_output(*args)