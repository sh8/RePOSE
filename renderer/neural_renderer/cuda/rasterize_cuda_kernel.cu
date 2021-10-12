#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>


__global__ void forward_face_index_map_cuda_kernel(
        const float* faces,
        const int batch_size,
        const int num_faces,
        const int image_height,
        const int image_width,
        const float near,
        const float far,
        int32_t* face_index_map,
        float* weight_map,
        float* depth_map,
        int32_t* lock_map) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int ih = image_height;
    const int iw = image_width;
    const int bn = i / num_faces;
    const int fn = i % num_faces;

    const float* face = &faces[i * 9];

    /* pi[0], pi[1], pi[2] = leftmost, middle, rightmost points */
    int pi[3];
    if (face[0] < face[3]) {
        if (face[6] < face[0]) pi[0] = 2; else pi[0] = 0;
        if (face[3] < face[6]) pi[2] = 2; else pi[2] = 1;
    } else {
        if (face[6] < face[3]) pi[0] = 2; else pi[0] = 1;
        if (face[0] < face[6]) pi[2] = 2; else pi[2] = 0;
    }
    for (int k = 0; k < 3; k++) {
      if (pi[0] != k && pi[2] != k) {
          pi[1] = k;
      }
    }

    /* p[num][xyz]: x, y is normalized from [-1, 1] to [0, ih or iw - 1]. */
    float p[3][3];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 3; dim++) {
            if (dim == 0) {
                p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * iw + iw - 1);
            } else if (dim == 1) {
                p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * ih + ih - 1);
            } else {
                p[num][dim] = face[3 * pi[num] + dim];
            }
        }
    }
    if (p[0][0] == p[2][0]) return; // line, not triangle

    /* compute face_inv */
    float face_inv[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};

    float face_inv_denominator = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));

    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
    }

    const int xi_min = max(ceil(p[0][0]), 0.);
    const int xi_max = min(p[2][0], iw - 1.0);
    for (int xi = xi_min; xi <= xi_max; xi++) {
        /* compute yi_min and yi_max */
        float yi1, yi2;
        if (xi <= p[1][0]) {
            if (p[1][0] - p[0][0] != 0) {
                yi1 = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];
            } else {
                yi1 = p[1][1];
            }
        } else {
            if (p[2][0] - p[1][0] != 0) {
                yi1 = (p[2][1] - p[1][1]) / (p[2][0] - p[1][0]) * (xi - p[1][0]) + p[1][1];
            } else {
                yi1 = p[1][1];
            }
        }
        yi2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];

        const int yi_min = max(0., ceil(min(yi1, yi2)));
        const int yi_max = min(max(yi1, yi2), ih - 1.0);

        for (int yi = yi_min; yi <= yi_max; yi++) {
            /* index in output buffers */
            int index = bn * ih * iw + yi * iw + xi;
            /* compute w = face_inv * p */
            float w[3];
            for (int k = 0; k < 3; k++) {
                w[k] = face_inv[3 * k + 0] * xi + face_inv[3 * k + 1] * yi + face_inv[3 * k + 2];
            }
            /* sum(w) -> 1, 0 < w < 1 */
            float w_sum = 0;
            for (int k = 0; k < 3; k++) {
                w[k] = min(max(w[k], 0.0), 1.0);
                w_sum += w[k];
            }
            for (int k = 0; k < 3; k++) w[k] /= w_sum;
            /* compute 1 / zp = sum(w / z) */
            const float zp = 1.0 / (w[0] / p[0][2] + w[1] / p[1][2] + w[2] / p[2][2]);
            if (zp <= near || far <= zp) continue;

            /* lock and update */
            bool locked = false;
            do {
                if (locked = atomicCAS(&lock_map[index], 0, 1) == 0) {
                    if (zp < atomicAdd(&depth_map[index], 0)) {
                        float record = 0;
                        atomicExch(&depth_map[index], zp);
                        atomicExch(&face_index_map[index], fn);
                        for (int k = 0; k < 3; k++) {
                          atomicExch(&weight_map[3 * index + pi[k]], w[k]);
                        }
                        record += atomicAdd(&depth_map[index], 0.);
                        record += atomicAdd(&face_index_map[index], 0.);
                        if (record > 0) atomicExch(&lock_map[index], 0);
                    } else {
                      atomicExch(&lock_map[index], 0);
                    }
                }
            } while (!locked);

        }
    }
}

__global__ void forward_texture_sampling_cuda_kernel(
		const float* faces,
		const float* textures,
	  const int32_t* face_index_map,
	  const float* weight_map,
    const size_t batch_size,
    const int num_faces,
    const int image_height,
    const int image_width,
    const int texture_size,
		float* feature_map) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height * image_width) {
        return;
    }

    const int ts = texture_size;
    const int face_index = face_index_map[i];
    float* pixel = &feature_map[i * (ts + 1)];
    
    if (face_index >= 0) {
        /*
            from global variables:
            batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
            texture[ts][RGB];
        */
        const int bn = i / (image_height * image_width);
        const int nf = num_faces;

        const float* texture = &textures[(bn * nf + face_index) * ts * 3];
        const float* weight = &weight_map[i * 3];
    
        /* blend */
        for (int k = 0; k < ts; k++) {
            for (int j = 0; j < 3; j++) {
                pixel[k] += weight[j] * texture[ts * j + k];
            }
        }
        pixel[ts] = 1.0f;
    }
}

__global__ void backward_cuda_kernel(
        const int32_t* face_index_map,
        const float* weight_map,
        const float* grad_feature_map,
        float* grad_textures,
        size_t batch_size,
        size_t num_faces,
        int image_height,
        int image_width,
        size_t texture_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_height * image_width) {
        return;
    }
    const int face_index = face_index_map[i];
    if (face_index >= 0) {
        int bn = i / (image_width * image_height);    // batch number [0 -> bs]
        int nf = num_faces;
        int ts = texture_size;
    
        const float* weight = &weight_map[i * 3];
        float* grad_texture = &grad_textures[(bn * nf + face_index) * ts * 3];

        for (int k = 0; k < ts; k++) {
            const float grad_feature = grad_feature_map[i * (ts + 1) + k];
            for (int j = 0; j < 3; j++) {
                atomicAdd(&grad_texture[ts * j + k], weight[j] * grad_feature);
            }
        }
    }
}

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
        const float far) {

    const int batch_size = faces.size(0);
    const int num_faces = faces.size(1);
    const int texture_size = textures.size(3);
    const int threads = 512;

    const dim3 blocks1 ((batch_size * num_faces - 1) / threads +1);

    forward_face_index_map_cuda_kernel<<<blocks1, threads>>>(
        faces.data_ptr<float>(),
        batch_size,
        num_faces,
        image_height,
        image_width,
        near,
        far,
        face_index_map.data_ptr<int32_t>(),
        weight_map.data_ptr<float>(),
        depth_map.data_ptr<float>(),
        lock_map.data_ptr<int32_t>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)  {
        printf("Error in forward_face_index_map: %s\n", cudaGetErrorString(err));
    }

    const dim3 blocks2 ((batch_size * image_height * image_width - 1) / threads + 1);

    forward_texture_sampling_cuda_kernel<<<blocks2, threads>>>(
        faces.data_ptr<float>(),
        textures.data_ptr<float>(),
        face_index_map.data_ptr<int32_t>(),
        weight_map.data_ptr<float>(),
        batch_size,
        num_faces,
        image_height,
        image_width,
        texture_size,
        feature_map.data_ptr<float>());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in forward_texture_sampling: %s\n", cudaGetErrorString(err));
    }

    return std::make_tuple(face_index_map, weight_map, depth_map, feature_map);
}

at::Tensor backward_cuda(
        const at::Tensor& face_index_map,
        const at::Tensor& weight_map,
        at::Tensor& grad_feature_map,
        at::Tensor& grad_textures,
        int num_faces) {

    const int batch_size = face_index_map.size(0);
    const int image_height = face_index_map.size(1);
    const int image_width = face_index_map.size(2);
    const int texture_size = grad_textures.size(3);
    const int threads = 512;
    const dim3 blocks ((batch_size * image_height * image_width - 1) / threads + 1);

    backward_cuda_kernel<<<blocks, threads>>>(
        face_index_map.data_ptr<int32_t>(),
        weight_map.data_ptr<float>(),
        grad_feature_map.data_ptr<float>(),
        grad_textures.data_ptr<float>(),
        batch_size,
        num_faces,
        image_height,
        image_width,
        texture_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in backward: %s\n", cudaGetErrorString(err));
    }

    return grad_textures;
}
