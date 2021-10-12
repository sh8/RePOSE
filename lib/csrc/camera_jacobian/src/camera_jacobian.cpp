#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward_cuda(
        const at::Tensor& face_index_map,
        const at::Tensor& depth_map,
        const at::Tensor& K,
        const at::Tensor& R,
        const at::Tensor& w,
        const at::Tensor& t,
        const at::Tensor& bbox);

at::Tensor forward_cpu(
        const at::Tensor& face_index_map,
        const at::Tensor& depth_map,
        const at::Tensor& K,
        const at::Tensor& R,
        const at::Tensor& w,
        const at::Tensor& t,
        const at::Tensor& bbox) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(K);
    CHECK_INPUT(R);
    CHECK_INPUT(w);
    CHECK_INPUT(t);
    CHECK_INPUT(bbox);

    return forward_cuda(face_index_map, depth_map,
            K, R, w, t, bbox);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "FORWARD (CUDA)");
}
