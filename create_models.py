"""
Create three ONNX models for the CPU -> OVEP-GPU -> CPU pipeline test.

Model 0: Simple Add (runs on CPU)
    input: X (1, 1, 4) float32
    output: Y (1, 1, 4) float32

Model 1: Simple Add + Tanh (runs on OVEP GPU)
    input: A (1, 1, 4) float32
    output: B (1, 1, 4) float32

Model 2: Simple Mul (runs on CPU)
    input: B (1, 1, 4) float32
    output: C (1, 1, 4) float32
"""

import numpy as np
import onnx
from onnx import TensorProto, helper


MODEL_SHAPE = [1, 1, 4]


def create_model0(path: str = "model0.onnx"):
    # X + bias -> Y
    bias_val = np.linspace(-0.25, 0.25, MODEL_SHAPE[-1], dtype=np.float32).reshape(1, 1, MODEL_SHAPE[-1])

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, MODEL_SHAPE)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, MODEL_SHAPE)

    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1, 1, MODEL_SHAPE[-1]], bias_val.flatten().tolist())

    add_node = helper.make_node("Add", inputs=["X", "bias"], outputs=["Y"])

    graph = helper.make_graph(
        [add_node],
        "model0_graph",
        [X],
        [Y],
        initializer=[bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved {path}")


def create_model1(path: str = "model1.onnx"):
    # A + bias2 -> Tanh -> B
    bias_val = np.linspace(0.1, -0.1, MODEL_SHAPE[-1], dtype=np.float32).reshape(1, 1, MODEL_SHAPE[-1])

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, MODEL_SHAPE)
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, MODEL_SHAPE)

    bias2 = helper.make_tensor("bias2", TensorProto.FLOAT, [1, 1, MODEL_SHAPE[-1]], bias_val.flatten().tolist())

    add_node = helper.make_node("Add", inputs=["A", "bias2"], outputs=["add_out"])
    tanh_node = helper.make_node("Tanh", inputs=["add_out"], outputs=["B"])

    graph = helper.make_graph(
        [add_node, tanh_node],
        "model1_graph",
        [A],
        [B],
        initializer=[bias2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved {path}")


def create_model2(path: str = "model2.onnx"):
    # B * scale -> C
    scale_val = np.linspace(0.5, 1.25, MODEL_SHAPE[-1], dtype=np.float32).reshape(1, 1, MODEL_SHAPE[-1])

    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, MODEL_SHAPE)
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, MODEL_SHAPE)

    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1, 1, MODEL_SHAPE[-1]], scale_val.flatten().tolist())

    mul_node = helper.make_node("Mul", inputs=["B", "scale"], outputs=["C"])

    graph = helper.make_graph(
        [mul_node],
        "model2_graph",
        [B],
        [C],
        initializer=[scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    create_model0()
    create_model1()
    create_model2()
    print("Done creating models.")
