import os
import numpy as np
def write_obj(file_path: str, vertices: np.ndarray, faces: np.ndarray):
	with open(file_path, "w", encoding="utf-8") as f:
		for v in vertices:
			f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
		for tri in faces:
			f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def get_sample_tet_grid(origin=(-0.2, 0.3, -0.2), size=(0.4, 0.4, 0.8), resolution=(10, 10, 20)):
    """Return (vertices [N,3], tets [M,4]) for a structured 6-tet-per-cell grid."""

    rx, ry, rz = resolution
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    origin = np.asarray(origin, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    cell = size / np.asarray([rx, ry, rz], dtype=np.float64)

    def vid(ix, iy, iz):
        # Same indexing order as the C++ grid_vertex_index style loops.
        return ((ix * (ry + 1) + iy) * (rz + 1) + iz)

    verts = np.empty(((rx + 1) * (ry + 1) * (rz + 1), 3), dtype=np.float64)
    for ix in range(rx + 1):
        for iy in range(ry + 1):
            for iz in range(rz + 1):
                idx = vid(ix, iy, iz)
                verts[idx] = origin + cell * np.array([ix, iy, iz], dtype=np.float64)

    def orient_positive(a, b, c, d):
        vol6 = np.linalg.det(np.column_stack((verts[b] - verts[a], verts[c] - verts[a], verts[d] - verts[a])))
        if vol6 > 0.0:
            return [a, b, c, d]
        return [a, c, b, d]

    tets = []
    for ix in range(rx):
        for iy in range(ry):
            for iz in range(rz):
                v000 = vid(ix, iy, iz)
                v100 = vid(ix + 1, iy, iz)
                v010 = vid(ix, iy + 1, iz)
                v110 = vid(ix + 1, iy + 1, iz)
                v001 = vid(ix, iy, iz + 1)
                v101 = vid(ix + 1, iy, iz + 1)
                v011 = vid(ix, iy + 1, iz + 1)
                v111 = vid(ix + 1, iy + 1, iz + 1)

                local = [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ]
                for tet in local:
                    tets.append(orient_positive(*tet))

    tets = np.asarray(tets, dtype=np.int32)
    return verts, tets
