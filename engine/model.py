
from .config import *
from io import BytesIO


# Map gltf binary component type IDs to numpy data types
COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32
}

# Map gltf data types to their dimensions
NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16
}

# Fallback configuration for materials if the model lacks PBR data.
DEFAULT_MATERIAL = {
    "ambient": (0.1, 0.1, 0.1),
    "diffuse": (1.0, 1.0, 1.0),
    "specular": (0.5, 0.5, 0.5),
    "shininess": 32.0,
    "alpha": 1.0,
    "alpha_mode": "OPAQUE"
}

class GLBModel:
    def __init__(self, name: str, path: str):
        self.name = name
        self.gltf = GLTF2().load(path)
        self._blob = self.gltf.binary_blob()
        self.meshes = {}
        self.materials = {}
        self._loaded_textures: set[str] = set()
        self._build_parent_map()
        self._load_meshes()
        del self.gltf, self._blob

    def _build_parent_map(self):
        """Maps each child node index to its parent node index"""
        self.parent_map = {
            child: i
            for i, node in enumerate(self.gltf.nodes)
            if node.children
            for child in node.children
        }

    def _node_matrix(self, node) -> np.ndarray:
        """Returns the local transformation matrix for a single node"""
        if node.matrix is not None:
            return np.array(node.matrix, dtype=np.float32).reshape(4, 4)
        
        # If no transformation matrix defined explicitly, create one
        T, R, S = (np.eye(4, dtype=np.float32) for _ in range(3))
        if node.translation:
            T[:3, 3] = node.translation
        if node.rotation:
            x, y, z, w = node.rotation
            # Quaternion to rotation matrix conversion
            R[:3, :3] = [
                [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)    ],
                [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
                [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
            ]
        if node.scale:
            np.fill_diagonal(S, [*node.scale, 1.0])
        return T @ R @ S

    def _world_matrix(self, node_index: int) -> np.ndarray:
        """Recursively accumulates transforms up the node hierarchy"""
        mat = self._node_matrix(self.gltf.nodes[node_index])
        parent = self.parent_map.get(node_index)
        if parent is not None:
            mat = self._world_matrix(parent) @ mat
        return mat

    def _apply_matrix(self, vertices, matrix) -> np.ndarray:
        """Transforms an (N, 3) position array by a 4x4 matrix"""
        homogeneous = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float32)])
        return (homogeneous @ matrix.T)[:, :3]

    def _load_meshes(self):
        for node_index, node in enumerate(self.gltf.nodes):
            if node.mesh is None:
                continue

            world_mat  = self._world_matrix(node_index)
            normal_mat = np.linalg.inv(world_mat[:3, :3]).T
            is_flipped = np.linalg.det(world_mat) < 0
            base_name  = node.name or f"node_{node_index}"

            for prim_index, prim in enumerate(self.gltf.meshes[node.mesh].primitives):
                mode = prim.mode if prim.mode is not None else 4
                if mode != 4:
                    print(f"Warning: Mesh {base_name} uses unsupported mode {mode}. Expected 4 (TRIANGLES).")

                attr    = prim.attributes
                verts   = self._get_accessor(attr.POSITION)
                normals = self._get_accessor(attr.NORMAL) if attr.NORMAL is not None else np.zeros((len(verts), 3), dtype='f4')
                uvs     = self._get_accessor(attr.TEXCOORD_0) if attr.TEXCOORD_0 is not None else np.zeros((len(verts), 2), dtype='f4')

                if prim.indices is not None:
                    idx = self._get_accessor(prim.indices).flatten().astype(np.int32)
                    if is_flipped:
                        idx = idx.reshape(-1, 3)
                        idx[:, [1, 2]] = idx[:, [2, 1]]
                        idx = idx.flatten()
                    verts, normals, uvs = verts[idx], normals[idx], uvs[idx]

                verts   = self._apply_matrix(verts, world_mat)
                normals = (normals @ normal_mat.T).astype(np.float32)
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normals /= norms

                name        = f"{base_name}_prim_{prim_index}"
                image_index = self._get_texture_index(prim)
                tex_label   = f"{self.name}.img_{image_index}" if image_index is not None else None
                mat         = self._get_material(prim)

                self.meshes[name] = {
                    "vertices": verts,
                    "normals": normals,
                    "uvs": uvs,
                    "texture_name": tex_label,
                    "alpha_mode": mat["alpha_mode"]
                }
                self.materials[name] = mat
                if tex_label is not None and tex_label not in self._loaded_textures:
                    tex_data = self._get_texture(prim)
                    if tex_data:
                        from .data import Data
                        Data.load_bin_tex(tex_label, tex_data["size"], tex_data["bytes"])
                        self._loaded_textures.add(tex_label)

    def _get_accessor(self, accessor_index: int) -> np.ndarray:
        # Gather metadata for the accessor
        accessor = self.gltf.accessors[accessor_index]
        dtype = COMPONENT_DTYPE[accessor.componentType]
        n_comp = NUM_COMPONENTS[accessor.type]
        element_bytes = np.dtype(dtype).itemsize * n_comp

        # Handle empty data by defaulting to zeros
        if accessor.bufferView is None:
            return np.zeros((accessor.count, n_comp), dtype=dtype)

        # Data boundaries are located using offset and stride
        bv = self.gltf.bufferViews[accessor.bufferView]
        offset = (bv.byteOffset or 0) + (accessor.byteOffset or 0)
        stride = bv.byteStride or element_bytes

        # Open a memoryview for fast look up and slice the data for accessor
        blob_view = memoryview(self._blob)[offset: offset + accessor.count * stride]

        if stride == element_bytes:
            # Tightly packed data: read normally
            out = np.frombuffer(blob_view, dtype=dtype).reshape(accessor.count, n_comp).copy()
        else:
            # Strided / interleaved data: read as raw bytes first, then reinterpret
            raw = np.frombuffer(blob_view, dtype=np.uint8)
            out = np.lib.stride_tricks.as_strided(raw, shape=(accessor.count, element_bytes), strides=(stride, 1))
            out = np.frombuffer(out.copy(), dtype=dtype).reshape(accessor.count, n_comp).copy()

        # Patch sparse data
        if accessor.sparse is not None:
            sp = accessor.sparse
            si_bv    = self.gltf.bufferViews[sp.indices.bufferView]
            si_dtype = COMPONENT_DTYPE[sp.indices.componentType]
            si_off   = (si_bv.byteOffset or 0) + (sp.indices.byteOffset or 0)
            indices  = np.frombuffer(self._blob[si_off: si_off + sp.count * np.dtype(si_dtype).itemsize], dtype=si_dtype).copy()
            sv_bv   = self.gltf.bufferViews[sp.values.bufferView]
            sv_off  = (sv_bv.byteOffset or 0) + (sp.values.byteOffset or 0)
            values  = np.frombuffer(self._blob[sv_off: sv_off + sp.count * element_bytes], dtype=dtype).reshape(sp.count, n_comp).copy()
            out[indices] = values
        return out

    def _get_material(self, prim) -> dict:
        # Handle empty material values
        if prim.material is None:
            return DEFAULT_MATERIAL.copy()

        # Load PBR material info if defined
        pbr = self.gltf.materials[prim.material].pbrMetallicRoughness
        if not pbr:
            return DEFAULT_MATERIAL.copy()

        # Extract material data
        base  = pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0]
        diff  = tuple(base[:3])
        metal = pbr.metallicFactor  if pbr.metallicFactor  is not None else 1.0
        rough = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
        
        # Translate PBR material data into the Blinn-Phong lighting model
        return {
            "ambient":    tuple(c * 0.2 for c in diff),
            "diffuse":    diff,
            "specular":   diff if metal > 0.5 else (0.5, 0.5, 0.5),
            "shininess":  max(1.0, (1.0 - rough) * 128),
            "alpha":      base[3],
            "alpha_mode": self.gltf.materials[prim.material].alphaMode or "OPAQUE"
        }

    def _get_texture_index(self, prim) -> int | None:
        """Returns the gltf image index for this primitive's base-colour texture or None"""
        if prim.material is None:
            return None
        pbr = self.gltf.materials[prim.material].pbrMetallicRoughness
        if not pbr or not pbr.baseColorTexture:
            return None
        return self.gltf.textures[pbr.baseColorTexture.index].source

    def _get_texture(self, prim):
        if prim.material is None:
            return None
        pbr = self.gltf.materials[prim.material].pbrMetallicRoughness
        if not pbr or not pbr.baseColorTexture:
            return None
        
        image = self.gltf.images[self.gltf.textures[pbr.baseColorTexture.index].source]
        if image.bufferView is not None:
            bv = self.gltf.bufferViews[image.bufferView]
            # No memoryview: slicing raw bytes directly is fine here as Pillow forces a 
            #                full decompression copy next anyway
            img_bytes = self._blob[(bv.byteOffset or 0): (bv.byteOffset or 0) + bv.byteLength]
        elif image.uri:
            if image.uri.startswith("data:"):
                img_bytes = base64.b64decode(image.uri.split(",", 1)[1])
            else:
                with open(image.uri, "rb") as f:
                    img_bytes = f.read()
        else:
            return None

        # Decrypt image data
        img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        return {"bytes": img.tobytes(), "size": img.size}