#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> complex_scan_forward_cuda(
    torch::Tensor a_real,
    torch::Tensor a_imag,
    torch::Tensor u_real,
    torch::Tensor u_imag);

std::vector<torch::Tensor> complex_scan_backward_cuda(
    torch::Tensor a_real,
    torch::Tensor a_imag,
    torch::Tensor h_real,
    torch::Tensor h_imag,
    torch::Tensor grad_h_real,
    torch::Tensor grad_h_imag);

static void check_scan_tensor(const torch::Tensor& value) {
  TORCH_CHECK(value.is_cuda(), "complex scan expects CUDA tensors");
  TORCH_CHECK(value.is_contiguous(), "complex scan expects contiguous tensors");
  TORCH_CHECK(value.dim() == 4, "complex scan expects [B,L,D,N] tensors");
  TORCH_CHECK(
      value.scalar_type() == torch::kFloat32 ||
          value.scalar_type() == torch::kFloat64,
      "complex scan supports float32 and float64");
}

std::vector<torch::Tensor> complex_scan_forward(
    torch::Tensor a_real,
    torch::Tensor a_imag,
    torch::Tensor u_real,
    torch::Tensor u_imag) {
  check_scan_tensor(a_real);
  check_scan_tensor(a_imag);
  check_scan_tensor(u_real);
  check_scan_tensor(u_imag);
  TORCH_CHECK(a_real.sizes() == a_imag.sizes(), "a parts must match");
  TORCH_CHECK(a_real.sizes() == u_real.sizes(), "a and u must match");
  TORCH_CHECK(a_real.sizes() == u_imag.sizes(), "u parts must match");
  TORCH_CHECK(a_real.scalar_type() == a_imag.scalar_type(), "a dtypes must match");
  TORCH_CHECK(a_real.scalar_type() == u_real.scalar_type(), "a/u dtypes must match");
  TORCH_CHECK(a_real.scalar_type() == u_imag.scalar_type(), "u dtypes must match");
  return complex_scan_forward_cuda(a_real, a_imag, u_real, u_imag);
}

std::vector<torch::Tensor> complex_scan_backward(
    torch::Tensor a_real,
    torch::Tensor a_imag,
    torch::Tensor h_real,
    torch::Tensor h_imag,
    torch::Tensor grad_h_real,
    torch::Tensor grad_h_imag) {
  check_scan_tensor(a_real);
  check_scan_tensor(a_imag);
  check_scan_tensor(h_real);
  check_scan_tensor(h_imag);
  check_scan_tensor(grad_h_real);
  check_scan_tensor(grad_h_imag);
  return complex_scan_backward_cuda(
      a_real, a_imag, h_real, h_imag, grad_h_real, grad_h_imag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &complex_scan_forward, "Complex scan forward (CUDA)");
  module.def("backward", &complex_scan_backward, "Complex scan backward (CUDA)");
}
