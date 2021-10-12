#include <torch/torch.h>

#include <tuple>

// CUDA forward declarations

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward_cuda(
        at::Tensor feature_map,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor lock_map,
        const at::Tensor& faces,
        const at::Tensor& textures,
        const int image_height,
        const int image_width,
        const float near,
        const float far);

at::Tensor backward_cuda(
        const at::Tensor& face_index_map,
        const at::Tensor& weight_map,
        at::Tensor& grad_feature_map,
        at::Tensor& grad_textures,
        int num_faces);

// C++ interface

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK 
#endif
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward_cpu(
        at::Tensor feature_map,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor lock_map,
        const at::Tensor& faces,
        const at::Tensor& textures,
        const int image_height,
        const int image_width,
        const float near,
        const float far) {

    CHECK_INPUT(feature_map);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(lock_map);
    CHECK_INPUT(faces);
    CHECK_INPUT(textures);

    return forward_cuda(feature_map, face_index_map, weight_map, depth_map, lock_map, faces, textures, image_height, image_width, near, far);
}

torch::Tensor backward_cpu(
        const at::Tensor& face_index_map,
        const at::Tensor& weight_map,
        torch::Tensor& grad_feature_map,
        torch::Tensor& grad_textures,
        int num_faces) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(grad_feature_map);
    CHECK_INPUT(grad_textures);

    return backward_cuda(face_index_map, weight_map, grad_feature_map,
                                  grad_textures, num_faces);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "FORWARD (CUDA)");
    m.def("backward", &backward_cpu, "BACKWARD (CUDA)");
}
