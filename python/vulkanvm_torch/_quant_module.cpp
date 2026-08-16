// _quant_module.cpp
// Pybind11 bindings for vulkanvm_quant.hpp

#include "vulkanvm_quant.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace vvm_torch::quant;

PYBIND11_MODULE(vulkanvm_quant, m) {
    m.doc() = "Chonk Buffer quantization: INT8 weights with fused dequant+matmul";

    py::class_<QuantizedWeight>(m, "QuantizedWeight")
        .def_readwrite("qweight", &QuantizedWeight::qweight)
        .def_readwrite("scales", &QuantizedWeight::scales)
        .def_readwrite("zeros", &QuantizedWeight::zeros)
        .def_readwrite("in_features", &QuantizedWeight::in_features)
        .def_readwrite("out_features", &QuantizedWeight::out_features)
        .def_readwrite("group_size", &QuantizedWeight::group_size)
        .def_readwrite("is_int4", &QuantizedWeight::is_int4);

    m.def("quantize_weight", &quantize_weight,
        py::arg("weight"), py::arg("group_size") = 128, py::arg("use_int4") = false);
    m.def("dequantize_weight", &dequantize_weight);

    m.def("quantized_matmul", &quantized_matmul,
        py::arg("x"), py::arg("qweight"), py::arg("scales"), py::arg("zeros"),
        py::arg("group_size") = 128, py::arg("is_int4") = false, py::arg("bias") = torch::Tensor());

    py::class_<QuantLinear, std::shared_ptr<QuantLinear>>(m, "QuantLinear")
        .def(py::init<int64_t, int64_t, bool, int64_t, bool>(),
            py::arg("in_features"), py::arg("out_features"),
            py::arg("bias") = true, py::arg("group_size") = 128, py::arg("use_int4") = false)
        .def("load_quantized_weight", &QuantLinear::load_quantized_weight,
            py::arg("qweight"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = torch::Tensor())
        .def("forward", &QuantLinear::forward)
        .def("set_lora", &QuantLinear::set_lora,
            py::arg("lora_a"), py::arg("lora_b"), py::arg("scale"))
        .def_readwrite("in_features", &QuantLinear::in_features)
        .def_readwrite("out_features", &QuantLinear::out_features)
        .def_readwrite("group_size", &QuantLinear::group_size)
        .def_readwrite("use_int4", &QuantLinear::use_int4)
        .def_readwrite("has_bias", &QuantLinear::has_bias);

    py::class_<QuantizedModelWeights>(m, "QuantizedModelWeights")
        .def_readwrite("qweights", &QuantizedModelWeights::qweights)
        .def_readwrite("scales", &QuantizedModelWeights::scales)
        .def_readwrite("zeros", &QuantizedModelWeights::zeros)
        .def_readwrite("biases", &QuantizedModelWeights::biases)
        .def_readwrite("group_size", &QuantizedModelWeights::group_size)
        .def_readwrite("is_int4", &QuantizedModelWeights::is_int4);

    m.def("quantize_weight", &quantize_weight,
        py::arg("weight"), py::arg("group_size") = 128, py::arg("use_int4") = false);
    m.def("dequantize_weight", &dequantize_weight);
    m.def("quantized_matmul", &quantized_matmul,
        py::arg("x"), py::arg("qweight"), py::arg("scales"), py::arg("zeros"),
        py::arg("group_size") = 128, py::arg("is_int4") = false, py::arg("bias") = torch::Tensor());

    py::class_<QuantizedModelWeights>(m, "QuantizedModelWeights")
        .def_readwrite("qweights", &QuantizedModelWeights::qweights)
        .def_readwrite("scales", &QuantizedModelWeights::scales)
        .def_readwrite("zeros", &QuantizedModelWeights::zeros)
        .def_readwrite("biases", &QuantizedModelWeights::biases)
        .def_readwrite("group_size", &QuantizedModelWeights::group_size)
        .def_readwrite("is_int4", &QuantizedModelWeights::is_int4);

    m.def("quantize_model_weights", &quantize_model_weights,
        py::arg("state_dict"), py::arg("target_modules"),
        py::arg("group_size") = 128, py::arg("use_int4") = false);

    // Note: allocate_quantized_model_buffer requires ChonkPool (C++ only)
    // Python side will use chonk.py to allocate in pool
}