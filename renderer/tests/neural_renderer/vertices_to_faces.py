import torch


@torch.jit.script
def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32, device=device) *
                     nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[2]))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
