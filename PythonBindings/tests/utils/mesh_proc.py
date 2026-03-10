import os
import numpy as np
def write_obj(file_path: str, vertices: np.ndarray, faces: np.ndarray):
	with open(file_path, "w", encoding="utf-8") as f:
		for v in vertices:
			f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
		for tri in faces:
			f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")