#include <ATen/ATen.h>

__device__ __forceinline__ void derivative_dRdw(
        float *jac_c, const float *w, const float *R,
        float xw, float yw, float zw,
        float xc, float yc, float zc, float zc_sq,
        float fx, float fy) {

  float w_sq = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  float dxdR00 = (fx * xw) / zc;
  float dxdR10 = (fx * yw) / zc;
  float dxdR20 = (fx * zw) / zc;
  float dxdR31 = (fy * xw) / zc;
  float dxdR41 = (fy * yw) / zc;
  float dxdR51 = (fy * zw) / zc;
  float dxdR60 = - (fx * xc * xw) / zc_sq;
  float dxdR70 = - (fx * xc * yw) / zc_sq;
  float dxdR80 = - (fx * xc * zw) / zc_sq;
  float dxdR61 = - (fy * yc * xw) / zc_sq;
  float dxdR71 = - (fy * yc * yw) / zc_sq;
  float dxdR81 = - (fy * yc * zw) / zc_sq;

  for (int k = 0; k < 3; k++) {
    float v01, v02, v12;
    float dRdv[9];

    if (k == 0) {
      v01 = - w[0] * w[2] + w[0] * R[3 * 1 + 0] + w[1] * (1 - R[3 * 0 + 0]);
      v02 = w[0] * w[1] + w[0] * R[3 * 2 + 0] + w[2] * (1 - R[3 * 0 + 0]);
      v12 = - w[0] * w[0] + w[1] * R[3 * 2 + 0] - w[2] * R[3 * 1 + 0];
    } else if (k == 1){
      v01 = - w[0] * (1 - R[3 * 1 + 1]) - w[1] * w[2] - w[1] * R[3 * 0 + 1];
      v02 = w[0] * R[3 * 2 + 1] + w[1] * w[1] - w[2] * R[3 * 0 + 1];
      v12 = - w[0] * w[1] + w[1] * R[3 * 2 + 1] + w[2] * (1 - R[3 * 1 + 1]);
    } else {
      v01 = w[0] * R[3 * 1 + 2] - w[1] * R[3 * 0 + 2] - w[2] * w[2];
      v02 = - w[0] * (1 - R[3 * 2 + 2]) + w[1] * w[2] - w[2] * R[3 * 0 + 2];
      v12 = - w[0] * w[2] - w[1] * (1 - R[3 * 2 + 2]) - w[2] * R[3 * 1 + 2];
    }

    v01 /= w_sq;
    v02 /= w_sq;
    v12 /= w_sq;
  
    dRdv[0] = v01 * R[3 * 1 + 0] + v02 * R[3 * 2 + 0]; 
    dRdv[1] = v01 * R[3 * 1 + 1] + v02 * R[3 * 2 + 1]; 
    dRdv[2] = v01 * R[3 * 1 + 2] + v02 * R[3 * 2 + 2]; 
    dRdv[3] = - v01 * R[3 * 0 + 0] + v12 * R[3 * 2 + 0]; 
    dRdv[4] = - v01 * R[3 * 0 + 1] + v12 * R[3 * 2 + 1]; 
    dRdv[5] = - v01 * R[3 * 0 + 2] + v12 * R[3 * 2 + 2]; 
    dRdv[6] = - v02 * R[3 * 0 + 0] - v12 * R[3 * 1 + 0]; 
    dRdv[7] = - v02 * R[3 * 0 + 1] - v12 * R[3 * 1 + 1]; 
    dRdv[8] = - v02 * R[3 * 0 + 2] - v12 * R[3 * 1 + 2]; 
    jac_c[6 * 0 + k] = dRdv[0] * dxdR00 +
                       dRdv[1] * dxdR10 +
                       dRdv[2] * dxdR20 +
                       dRdv[6] * dxdR60 +
                       dRdv[7] * dxdR70 +
                       dRdv[8] * dxdR80;
    jac_c[6 * 1 + k] = dRdv[3] * dxdR31 +
                       dRdv[4] * dxdR41 +
                       dRdv[5] * dxdR51 +
                       dRdv[6] * dxdR61 +
                       dRdv[7] * dxdR71 +
                       dRdv[8] * dxdR81;
  }
}

__global__ void forward_cuda_kernel(
        float* jacobian_c,
        int32_t* face_index_map,
        float* depth_map,
        float* K, float* R, float* w, float* t,
        int image_height, int image_width, int batch_size,
        int32_t* bbox) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_size * image_height * image_width) {
      return;
  }

  const int bn = i / (image_height * image_width);
  const int pn = i % (image_height * image_width);
  const int xi = pn % image_width + bbox[bn * 4 + 1];
  const int yi = pn / image_width + bbox[bn * 4 + 0];
  const int face_index = face_index_map[i];

  if (face_index >= 0) {
      const float* K_ = &K[bn * 9];
      const float* R_ = &R[bn * 9];
      const float* w_ = &w[bn * 3];
      const float* t_ = &t[bn * 3];
      const float fx = K_[0];
      const float fy = K_[4];
      const float px = K_[2];
      const float py = K_[5];

      float* jac_c = &jacobian_c[i * 2 * 6];

      const float zc = depth_map[i];
      const float xc = zc * ((xi - px) / fx);
      const float yc = zc * ((yi - py) / fy);
      const float zc_sq = zc * zc;

      const float xct = xc - t_[0];
      const float yct = yc - t_[1];
      const float zct = zc - t_[2];
      const float xw = R_[0] * xct + R_[3] * yct + R_[6] * zct;
      const float yw = R_[1] * xct + R_[4] * yct + R_[7] * zct;
      const float zw = R_[2] * xct + R_[5] * yct + R_[8] * zct;

      /*
      printf("xw: %f, face[0]: %f, face[3]: %f, face[6]: %f \n", xw, face[0], face[3], face[6]);
      printf("yw: %f, face[1]: %f, face[4]: %f, face[7]: %f  \n", yw, face[1], face[4], face[7]);
      printf("zw: %f, face[2]: %f, face[5]: %f, face[8]: %f  \n", zw, face[2], face[5], face[8]);
      */

      derivative_dRdw(jac_c, w_, R_, xw, yw, zw, xc, yc, zc, zc_sq, fx, fy);
      jac_c[3] = fx / zc;
      jac_c[10] = fy / zc;
      jac_c[5] = -((fx * xc) / zc_sq);
      jac_c[11] = -((fy * yc) / zc_sq);

      /* printf("Grad X: %d, Grad Y: %d \n", grad[0], grad[1]); */
  }
}

at::Tensor forward_cuda(
        const at::Tensor& face_index_map,
        const at::Tensor& depth_map,
        const at::Tensor& K,
        const at::Tensor& R,
        const at::Tensor& w,
        const at::Tensor& t,
        const at::Tensor& bbox) {

    const int batch_size = face_index_map.size(0);
    const int image_height = face_index_map.size(1);
    const int image_width = face_index_map.size(2);
    const int threads = 512;
    const dim3 blocks ((batch_size * image_height * image_width - 1) / threads + 1);

    auto float_opts = face_index_map.options().dtype(at::kFloat);
    at::Tensor jacobian_c = at::full(
        {batch_size, image_height, image_width, 2, 6}, 0.0f, float_opts);

    forward_cuda_kernel<<<blocks, threads>>>(
        jacobian_c.data_ptr<float>(),
        face_index_map.data_ptr<int32_t>(),
        depth_map.data_ptr<float>(),
        K.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(), t.data_ptr<float>(),
        image_height, image_width, batch_size,
        bbox.data_ptr<int32_t>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in forward: %s\n", cudaGetErrorString(err));
    }

    return jacobian_c;
}
