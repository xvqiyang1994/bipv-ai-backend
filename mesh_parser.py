"""
Mesh file parser for OBJ and GLB/GLTF formats.
Converts uploaded files into numpy arrays (vertices, faces)
that the solar engine can process.
"""

import numpy as np
import struct
import json
import base64
from io import BytesIO
from typing import Tuple


def parse_obj(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a Wavefront OBJ file into vertices and faces.

    Args:
        text: OBJ file content as string

    Returns:
        vertices: (V, 3) float64 array
        faces: (F, 3) int32 array (triangulated)
    """
    positions = []
    face_indices = []

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith('v '):
            parts = line.split()
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

        elif line.startswith('f '):
            tokens = line[2:].strip().split()
            # Parse vertex indices (handles v, v/vt, v/vt/vn, v//vn)
            indices = []
            for tok in tokens:
                vi = int(tok.split('/')[0])
                # OBJ indices are 1-based
                indices.append(vi - 1 if vi > 0 else vi + len(positions))

            # Fan triangulation for polygons with > 3 vertices
            for i in range(1, len(indices) - 1):
                face_indices.append([indices[0], indices[i], indices[i + 1]])

    if not positions:
        raise ValueError("No vertices found in OBJ file")
    if not face_indices:
        raise ValueError("No faces found in OBJ file")

    vertices = np.array(positions, dtype=np.float64)
    faces = np.array(face_indices, dtype=np.int32)

    return vertices, faces


def parse_glb(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a binary GLB (glTF 2.0) file into vertices and faces.

    Args:
        data: GLB file content as bytes

    Returns:
        vertices: (V, 3) float64 array
        faces: (F, 3) int32 array
    """
    # GLB header: magic(4) + version(4) + length(4) = 12 bytes
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != 0x46546C67:  # 'glTF'
        raise ValueError("Not a valid GLB file")

    version = struct.unpack_from('<I', data, 4)[0]
    if version != 2:
        raise ValueError(f"Unsupported glTF version: {version}")

    # First chunk: JSON
    chunk0_length = struct.unpack_from('<I', data, 12)[0]
    chunk0_type = struct.unpack_from('<I', data, 16)[0]
    if chunk0_type != 0x4E4F534A:  # 'JSON'
        raise ValueError("First GLB chunk is not JSON")

    json_data = data[20:20 + chunk0_length].decode('utf-8')
    gltf = json.loads(json_data)

    # Second chunk: binary buffer
    bin_offset = 20 + chunk0_length
    bin_length = struct.unpack_from('<I', data, bin_offset)[0]
    bin_data = data[bin_offset + 8: bin_offset + 8 + bin_length]

    # Extract mesh data
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh in gltf.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            # Get position accessor
            pos_idx = primitive.get('attributes', {}).get('POSITION')
            if pos_idx is None:
                continue

            pos_accessor = gltf['accessors'][pos_idx]
            pos_bv = gltf['bufferViews'][pos_accessor['bufferView']]
            pos_offset = pos_bv.get('byteOffset', 0) + pos_accessor.get('byteOffset', 0)
            pos_count = pos_accessor['count']

            # Read position data
            pos_data = np.frombuffer(
                bin_data[pos_offset: pos_offset + pos_count * 12],
                dtype=np.float32
            ).reshape(-1, 3).astype(np.float64)

            all_vertices.append(pos_data)

            # Get index accessor
            idx_field = primitive.get('indices')
            if idx_field is not None:
                idx_accessor = gltf['accessors'][idx_field]
                idx_bv = gltf['bufferViews'][idx_accessor['bufferView']]
                idx_offset = idx_bv.get('byteOffset', 0) + idx_accessor.get('byteOffset', 0)
                idx_count = idx_accessor['count']

                # Determine index type
                comp_type = idx_accessor['componentType']
                if comp_type == 5123:  # UNSIGNED_SHORT
                    indices = np.frombuffer(
                        bin_data[idx_offset: idx_offset + idx_count * 2],
                        dtype=np.uint16
                    ).astype(np.int32)
                elif comp_type == 5125:  # UNSIGNED_INT
                    indices = np.frombuffer(
                        bin_data[idx_offset: idx_offset + idx_count * 4],
                        dtype=np.uint32
                    ).astype(np.int32)
                else:  # UNSIGNED_BYTE
                    indices = np.frombuffer(
                        bin_data[idx_offset: idx_offset + idx_count],
                        dtype=np.uint8
                    ).astype(np.int32)

                # Offset indices for merged mesh
                faces = indices.reshape(-1, 3) + vertex_offset
                all_faces.append(faces)
            else:
                # No indices — sequential triangles
                n_tris = pos_count // 3
                faces = np.arange(n_tris * 3, dtype=np.int32).reshape(-1, 3) + vertex_offset
                all_faces.append(faces)

            vertex_offset += pos_count

    if not all_vertices:
        raise ValueError("No mesh data found in GLB file")

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    return vertices, faces


def center_and_normalize(vertices: np.ndarray) -> np.ndarray:
    """
    Center mesh on XZ plane and place bottom at Y=0.
    """
    # Center X and Z
    center_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    center_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    y_min = vertices[:, 1].min()

    vertices = vertices.copy()
    vertices[:, 0] -= center_x
    vertices[:, 1] -= y_min
    vertices[:, 2] -= center_z

    return vertices
