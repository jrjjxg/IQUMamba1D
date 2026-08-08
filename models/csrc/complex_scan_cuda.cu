#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

template <typename scalar_t>
__global__ void complex_scan_forward_kernel(
    const scalar_t* __restrict__ a_real,
    const scalar_t* __restrict__ a_imag,
    const scalar_t* __restrict__ u_real,
    const scalar_t* __restrict__ u_imag,
    scalar_t* __restrict__ h_real,
    scalar_t* __restrict__ h_imag,
    int64_t length,
    int64_t states_per_step,
    int64_t total_sequence_states) {
  const int64_t sequence_state =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sequence_state >= total_sequence_states) {
    return;
  }
  const int64_t batch = sequence_state / states_per_step;
  const int64_t state = sequence_state - batch * states_per_step;
  scalar_t hr = scalar_t(0);
  scalar_t hi = scalar_t(0);
  int64_t index = batch * length * states_per_step + state;
  for (int64_t time = 0; time < length; ++time) {
    const scalar_t ar = a_real[index];
    const scalar_t ai = a_imag[index];
    const scalar_t next_real = ar * hr - ai * hi + u_real[index];
    const scalar_t next_imag = ar * hi + ai * hr + u_imag[index];
    hr = next_real;
    hi = next_imag;
    h_real[index] = hr;
    h_imag[index] = hi;
    index += states_per_step;
  }
}

template <typename scalar_t>
__global__ void complex_scan_backward_kernel(
    const scalar_t* __restrict__ a_real,
    const scalar_t* __restrict__ a_imag,
    const scalar_t* __restrict__ h_real,
    const scalar_t* __restrict__ h_imag,
    const scalar_t* __restrict__ grad_h_real,
    const scalar_t* __restrict__ grad_h_imag,
    scalar_t* __restrict__ grad_a_real,
    scalar_t* __restrict__ grad_a_imag,
    scalar_t* __restrict__ grad_u_real,
    scalar_t* __restrict__ grad_u_imag,
    int64_t length,
    int64_t states_per_step,
    int64_t total_sequence_states) {
  const int64_t sequence_state =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sequence_state >= total_sequence_states) {
    return;
  }
  const int64_t batch = sequence_state / states_per_step;
  const int64_t state = sequence_state - batch * states_per_step;
  scalar_t future_real = scalar_t(0);
  scalar_t future_imag = scalar_t(0);
  int64_t index =
      batch * length * states_per_step + (length - 1) * states_per_step + state;
  for (int64_t time = length - 1; time >= 0; --time) {
    const scalar_t total_real = grad_h_real[index] + future_real;
    const scalar_t total_imag = grad_h_imag[index] + future_imag;
    const scalar_t previous_real =
        time > 0 ? h_real[index - states_per_step] : scalar_t(0);
    const scalar_t previous_imag =
        time > 0 ? h_imag[index - states_per_step] : scalar_t(0);
    grad_u_real[index] = total_real;
    grad_u_imag[index] = total_imag;
    grad_a_real[index] =
        total_real * previous_real + total_imag * previous_imag;
    grad_a_imag[index] =
        -total_real * previous_imag + total_imag * previous_real;
    const scalar_t ar = a_real[index];
    const scalar_t ai = a_imag[index];
    future_real = total_real * ar + total_imag * ai;
    future_imag = -total_real * ai + total_imag * ar;
    index -= states_per_step;
  }
}

std::vector<torch::Tensor> complex_scan_forward_cuda(
    torch::Tensor a_real,
    torch::Tensor a_imag,
    torch::Tensor u_real,
    torch::Tensor u_imag) {
  const c10::cuda::CUDAGuard device_guard(a_real.device());
  auto h_real = torch::empty_like(u_real);
  auto h_imag = torch::empty_like(u_imag);
  const int64_t batch = a_real.size(0);
  const int64_t length = a_real.size(1);
  const int64_t states_per_step = a_real.size(2) * a_real.size(3);
  constexpr int threads = 256;
  const int64_t total = batch * states_per_step;
  const dim3 blocks((total + threads - 1) / threads, 1, 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(a_real.scalar_type(), "complex_scan_forward_cuda", [&] {
    complex_scan_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        a_real.data_ptr<scalar_t>(),
        a_imag.data_ptr<scalar_t>(),
        u_real.data_ptr<scalar_t>(),
        u_imag.data_ptr<scalar_t>(),
        h_real.data_ptr<scalar_t>(),
        h_imag.data_ptr<scalar_t>(),
        length,
        states_per_step,
        total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {h_real, h_imag};
}

std::vector<torch::Tensor> complex_scan_backward_cuda(
    torch::Tensor a_real,
    torch::Tensor a_imag,
    torch::Tensor h_real,
    torch::Tensor h_imag,
    torch::Tensor grad_h_real,
    torch::Tensor grad_h_imag) {
  const c10::cuda::CUDAGuard device_guard(a_real.device());
  auto grad_a_real = torch::empty_like(a_real);
  auto grad_a_imag = torch::empty_like(a_imag);
  auto grad_u_real = torch::empty_like(h_real);
  auto grad_u_imag = torch::empty_like(h_imag);
  const int64_t batch = a_real.size(0);
  const int64_t length = a_real.size(1);
  const int64_t states_per_step = a_real.size(2) * a_real.size(3);
  constexpr int threads = 256;
  const int64_t total = batch * states_per_step;
  const dim3 blocks((total + threads - 1) / threads, 1, 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(a_real.scalar_type(), "complex_scan_backward_cuda", [&] {
    complex_scan_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        a_real.data_ptr<scalar_t>(),
        a_imag.data_ptr<scalar_t>(),
        h_real.data_ptr<scalar_t>(),
        h_imag.data_ptr<scalar_t>(),
        grad_h_real.data_ptr<scalar_t>(),
        grad_h_imag.data_ptr<scalar_t>(),
        grad_a_real.data_ptr<scalar_t>(),
        grad_a_imag.data_ptr<scalar_t>(),
        grad_u_real.data_ptr<scalar_t>(),
        grad_u_imag.data_ptr<scalar_t>(),
        length,
        states_per_step,
        total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_a_real, grad_a_imag, grad_u_real, grad_u_imag};
}
