from .states import *


class Renderer:

    """
    Fixes Required:
    - Glitchy Transparent 2D texture rendering (rapid flickering)
    """

    BUFF_GROWTH_FACTOR = 2

    # Byte sizes for GPU buffer layout
    VERTEX_SIZE   = 8   # 2f uv + 3f pos + 3f normal
    INSTANCE_SIZE = 96  # mat4 + tex_id + pad[3] + tint + alpha
    MATERIAL_SIZE = 48  # ambient + diffuse + specular + shininess

    MAX_POINTS    = 500
    MAX_LINES     = 500
    MAX_INSTANCES = 500
    MAX_DRAWS     = 256

    core = None

    # Points
    points     = []
    point_prog = None
    point_vbo  = None
    point_vao  = None

    # Lines
    lines     = []
    line_prog = None
    line_vbo  = None
    line_vao  = None

    # Models — geometry is packed into one big VBO shared across all meshes
    global_ibo      = None
    global_vbo      = None
    global_vbo_data = np.array([], 'f4')
    model_prog      = None
    model_vao       = None
    model_tex_handles = []
    model_tex_lookup  = {}

    # Per-frame draw buckets, keyed by mesh label
    meshes               = {}
    opaque_instances     = {}
    cutout_instances     = {}   # alpha_mode == "MASK"
    transparent_instances = {}
    persistent_draw_entries  = {}   # handle -> {mesh_label, tint, alpha, base_instance, draw_cmd_index}
    persistent_instance_data = {}   # mesh_label -> packed bytes already in the SSBO
    persistent_draw_cmds     = {}   # mesh_label -> {count, instance_count, firstIndex, baseVertex, base_instance}
    persistent_material_data = {}   # mesh_label -> packed floats already in material SSBO
    _persistent_handle_counter = 0
    _persistent_instance_count = 0  # how many instance slots are occupied at the front of the SSBO
    _persistent_draw_count     = 0  # how many draw commands are occupied at the front of the draw cmd buffer


    # SSBOs
    instance_ssbo              = None
    base_instance_ssbo         = None
    draw_cmd_buffer            = None
    material_ssbo              = None
    model_texture_handle_ssbo  = None
    tex_texture_handle_ssbo    = None

    # Texture sprites (procedural quad geometry generated in the vertex shader)
    tex_prog          = None
    tex_vao           = None
    tex_instance_ssbo = None
    tex_opaque_queue      = []  # alpha >= 0.9
    tex_transparent_queue = []  # alpha <  0.9
    tex_tex_handles = []
    tex_tex_lookup  = {}
    MAX_TEX_INSTANCES = 500
    TEX_INSTANCE_SIZE = 96

    # Lights
    lights        = []
    ambient_light = [0.5, 0.5, 0.5]

    skybox_vertices = np.array([
        -1,  1, -1,  -1, -1, -1,   1, -1, -1,
         1, -1, -1,   1,  1, -1,  -1,  1, -1,

        -1, -1,  1,  -1, -1, -1,  -1,  1, -1,
        -1,  1, -1,  -1,  1,  1,  -1, -1,  1,

         1, -1, -1,   1, -1,  1,   1,  1,  1,
         1,  1,  1,   1,  1, -1,   1, -1, -1,

        -1, -1,  1,  -1,  1,  1,   1,  1,  1,
         1,  1,  1,   1, -1,  1,  -1, -1,  1,

        -1,  1, -1,   1,  1, -1,   1,  1,  1,
         1,  1,  1,  -1,  1,  1,  -1,  1, -1,

        -1, -1, -1,  -1, -1,  1,   1, -1, -1,
         1, -1, -1,  -1, -1,  1,   1, -1,  1,
    ], dtype='f4')
    skybox_vbo     = None
    skybox_prog    = None
    skybox_vao     = None
    skybox_cubemap = None
    skybox_enabled = False

    _enabled = True

    global_vbo_offset = 0

    # -------------------------------------------------------------------------
    # Enable / disable
    # -------------------------------------------------------------------------

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    @classmethod
    def init(cls, core):
        cls.core = core
        ctx = cls.core.ctx

        ctx.line_width  = 1
        ctx.depth_func  = '<='
        ctx.blend_func  = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.enable(mgl.PROGRAM_POINT_SIZE)
        ctx.enable(mgl.BLEND)
        ctx.disable(mgl.CULL_FACE)

        # Points — 32 bytes per vertex (1f size + 3f pos + 4f color)
        cls.point_vbo  = ctx.buffer(reserve=32 * cls.MAX_POINTS, dynamic=True)
        cls.point_prog = ctx.program(
            load_engine_shader("point_vs.glsl"),
            load_engine_shader("point_fs.glsl"),
        )
        cls.point_prog['model'].write(np.eye(4, dtype='f4').tobytes())
        cls.point_prog['projection'].write(np.array(Camera.projection.to_list(), dtype='f4').tobytes())
        cls.point_vao = ctx.vertex_array(cls.point_prog, cls.point_vbo, 'in_size', 'in_pos', 'in_col')

        # Lines — 56 bytes per vertex (3f pos + 4f color) × 2 endpoints
        cls.line_vbo  = ctx.buffer(reserve=56 * cls.MAX_LINES, dynamic=True)
        cls.line_prog = ctx.program(
            load_engine_shader("line_vs.glsl"),
            load_engine_shader("line_fs.glsl"),
        )
        cls.line_prog['model'].write(np.eye(4, dtype='f4').tobytes())
        cls.line_prog['projection'].write(np.array(Camera.projection.to_list(), dtype='f4').tobytes())
        cls.line_vao = ctx.vertex_array(cls.line_prog, cls.line_vbo, 'in_pos', 'in_col')

        # Models
        cls.model_prog = ctx.program(
            load_engine_shader("model_vs.glsl"),
            load_engine_shader("model_fs.glsl"),
        )
        cls.model_prog['projection'].write(np.array(Camera.projection.to_list(), dtype='f4').tobytes())
        cls.set_ambient_light(cls.ambient_light)
        cls.enable_specular()

        cls.global_ibo        = ctx.buffer(np.arange(400000, dtype='u4').tobytes())
        cls.instance_ssbo     = ctx.buffer(reserve=cls.MAX_INSTANCES * cls.INSTANCE_SIZE)
        cls.draw_cmd_buffer   = ctx.buffer(reserve=cls.MAX_DRAWS * 20)
        cls.base_instance_ssbo = ctx.buffer(reserve=cls.MAX_DRAWS * 4)
        cls.material_ssbo     = ctx.buffer(reserve=cls.MAX_DRAWS * cls.MATERIAL_SIZE)
        cls.model_texture_handle_ssbo = ctx.buffer(reserve=500 * 64)
        cls.tex_texture_handle_ssbo   = ctx.buffer(reserve=500 * 64)

        cls.instance_ssbo.bind_to_storage_buffer(0)
        cls.base_instance_ssbo.bind_to_storage_buffer(1)
        cls.material_ssbo.bind_to_storage_buffer(2)
        cls.model_texture_handle_ssbo.bind_to_storage_buffer(4)
        cls.tex_texture_handle_ssbo.bind_to_storage_buffer(5)

        # Texture sprite MDI
        cls.tex_prog = ctx.program(
            load_engine_shader("texture_vs.glsl"),
            load_engine_shader("texture_fs.glsl"),
        )
        cls.tex_prog['projection'].write(np.array(Camera.projection.to_list(), dtype='f4').tobytes())
        cls.tex_vao           = ctx.vertex_array(cls.tex_prog, [])
        cls.tex_instance_ssbo = ctx.buffer(reserve=cls.MAX_TEX_INSTANCES * cls.TEX_INSTANCE_SIZE, dynamic=True)
        cls.tex_instance_ssbo.bind_to_storage_buffer(3)

        # Skybox
        cls.skybox_vbo  = ctx.buffer(cls.skybox_vertices.tobytes())
        cls.skybox_prog = ctx.program(
            load_engine_shader("skybox_vs.glsl"),
            load_engine_shader("skybox_fs.glsl"),
        )
        cls.skybox_prog['projection'].write(np.array(Camera.projection.to_list(), dtype='f4').tobytes())
        cls.skybox_vao = ctx.vertex_array(
            cls.skybox_prog,
            [(cls.skybox_vbo, '3f', 'in_position')],
        )

        # Default 1×1 white texture occupies slot 0; used for untextured meshes
        white_pixel = np.array([255, 255, 255, 255], dtype='u1')
        cls.default_tex = ctx.texture((1, 1), 4, white_pixel.tobytes())
        cls.model_tex_lookup["default_white"] = 0
        cls.model_tex_handles.append(cls.default_tex.get_handle())
        handles64 = np.array(cls.model_tex_handles, dtype=np.uint64)
        handles32 = np.zeros((len(handles64), 2), dtype=np.uint32)
        handles32[:, 0] = handles64 & 4294967295
        handles32[:, 1] = handles64 >> 32
        cls.model_texture_handle_ssbo.write(handles32.tobytes())

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    @classmethod
    def set_line_width(cls, line_width: float):
        cls.core.ctx.line_width = line_width

    @classmethod
    def set_ambient_light(cls, light: list):
        cls.ambient_light = light
        cls.model_prog['ambient_light'].value = tuple(light)

    @classmethod
    def enable_specular(cls):
        cls.model_prog['use_spec'].value = True

    @classmethod
    def disable_specular(cls):
        cls.model_prog['use_spec'].value = False

    @classmethod
    def enable_skybox(cls):
        """Skybox is only rendered in 3D perspective."""
        cls.skybox_enabled = True

    @classmethod
    def disable_skybox(cls):
        cls.skybox_enabled = False

    @classmethod
    def set_skybox(cls, label: str):
        cls.skybox_cubemap = Data.get_cubemap(label)

    # -------------------------------------------------------------------------
    # Internal GPU uploads
    # -------------------------------------------------------------------------

    @classmethod
    def _upload_lights(cls):
        cls.model_prog['num_lights'].value = len(cls.lights)
        for i, light in enumerate(cls.lights):
            cls.model_prog[f'lights[{i}].position'].value  = tuple(light.pos)
            cls.model_prog[f'lights[{i}].intensity'].value = float(light.intensity)
            cls.model_prog[f'lights[{i}].color'].value     = tuple(light.color)

    @classmethod
    def _upload_view(cls):
        view_bytes = np.array(Camera.view.to_list(), dtype='f4').tobytes()
        cls.model_prog['view'].write(view_bytes)
        cls.model_prog['view_pos'].write(Camera.pos.to_bytes())
        cls.tex_prog['view'].write(view_bytes)

    @classmethod
    def _update_projection(cls):
        proj_bytes = np.array(Camera.projection.to_list(), dtype='f4').tobytes()
        for prog in (cls.skybox_prog, cls.point_prog, cls.line_prog, cls.model_prog, cls.tex_prog):
            prog['projection'].write(proj_bytes)

    @classmethod
    def _grow_point_buffer(cls):
        cls.MAX_POINTS = int(cls.MAX_POINTS * cls.BUFF_GROWTH_FACTOR)
        cls.point_vbo.release()
        cls.point_vbo = cls.core.ctx.buffer(reserve=32 * cls.MAX_POINTS, dynamic=True)
        cls.point_vao = cls.core.ctx.vertex_array(cls.point_prog, cls.point_vbo, 'in_size', 'in_pos', 'in_col')

    @classmethod
    def _grow_line_buffer(cls):
        cls.MAX_LINES = int(cls.MAX_LINES * cls.BUFF_GROWTH_FACTOR)
        cls.line_vbo.release()
        cls.line_vbo = cls.core.ctx.buffer(reserve=56 * cls.MAX_LINES, dynamic=True)
        cls.line_vao = cls.core.ctx.vertex_array(cls.line_prog, cls.line_vbo, 'in_pos', 'in_col')

    @classmethod
    def _grow_model_ssbos(cls):
        """Double all model-pipeline SSBOs when either instance or draw capacity is exceeded."""
        ctx = cls.core.ctx

        cls.MAX_INSTANCES = int(cls.MAX_INSTANCES * cls.BUFF_GROWTH_FACTOR)
        cls.MAX_DRAWS     = int(cls.MAX_DRAWS     * cls.BUFF_GROWTH_FACTOR)

        # Preserve live data so the resize is transparent to the caller
        instance_data      = cls.instance_ssbo.read()
        base_instance_data = cls.base_instance_ssbo.read()
        draw_cmd_data      = cls.draw_cmd_buffer.read()
        material_data      = cls.material_ssbo.read()
        tex_handle_data    = cls.model_texture_handle_ssbo.read()

        cls.instance_ssbo.release()
        cls.base_instance_ssbo.release()
        cls.draw_cmd_buffer.release()
        cls.material_ssbo.release()
        cls.model_texture_handle_ssbo.release()

        cls.instance_ssbo     = ctx.buffer(reserve=cls.MAX_INSTANCES * cls.INSTANCE_SIZE)
        cls.base_instance_ssbo = ctx.buffer(reserve=cls.MAX_DRAWS * 4)
        cls.draw_cmd_buffer   = ctx.buffer(reserve=cls.MAX_DRAWS * 20)
        cls.material_ssbo     = ctx.buffer(reserve=cls.MAX_DRAWS * cls.MATERIAL_SIZE)
        cls.model_texture_handle_ssbo = ctx.buffer(reserve=cls.MAX_INSTANCES * 64)

        # Re-bind to the same slots they were assigned during init()
        cls.instance_ssbo.bind_to_storage_buffer(0)
        cls.base_instance_ssbo.bind_to_storage_buffer(1)
        cls.material_ssbo.bind_to_storage_buffer(2)
        cls.model_texture_handle_ssbo.bind_to_storage_buffer(4)

        # Restore the data that was already staged for this frame
        cls.instance_ssbo.write(instance_data)
        cls.base_instance_ssbo.write(base_instance_data)
        cls.draw_cmd_buffer.write(draw_cmd_data)
        cls.material_ssbo.write(material_data)
        cls.model_texture_handle_ssbo.write(tex_handle_data)    

    @classmethod
    def _rebuild_model_vao(cls):
        """Releases and recreates the model VAO against the current global_vbo."""
        if cls.model_vao:
            cls.model_vao.release()
        cls.model_vao = cls.core.ctx.vertex_array(
            cls.model_prog,
            [(cls.global_vbo, "2f 3f 3f", "in_uv", "in_pos", "in_norm")],
            index_buffer=cls.global_ibo
        )

    @classmethod
    def add_model(cls, label: str, filename: str):
        if label in Data.models:
            return
        Data.load_model(label, filename)
        glb_model = Data.get_model(label)
        new_data  = np.array([], 'f4')

        for mesh, data in glb_model.meshes.items():
            final_label = glb_model.name + '#' + mesh
            mesh_data   = np.ascontiguousarray(np.hstack([data["uvs"], data["vertices"], data["normals"]]).ravel())
            cls.meshes[final_label] = {
                "offset":     cls.global_vbo_offset // cls.VERTEX_SIZE,
                "count":      len(mesh_data) // cls.VERTEX_SIZE,
                "alpha_mode": data.get("alpha_mode", "OPAQUE"),
            }
            new_data = np.concatenate((new_data, mesh_data))
            cls.global_vbo_offset += len(mesh_data)

            # Register the texture handle only once per unique image
            tex_label = data["texture_name"]
            if tex_label is not None:
                if tex_label not in cls.model_tex_lookup:
                    cls.model_tex_lookup[tex_label] = len(cls.model_tex_handles)
                    cls.model_tex_handles.append(Data.get_tex(tex_label).get_handle())
                cls.meshes[final_label]["tex_index"] = cls.model_tex_lookup[tex_label]
            else:
                cls.meshes[final_label]["tex_index"] = 0

            del data["uvs"], data["vertices"], data["normals"]

        cls.model_texture_handle_ssbo.clear()
        handles64 = np.array(cls.model_tex_handles, dtype=np.uint64)
        handles32 = np.zeros((len(handles64), 2), dtype=np.uint32)
        handles32[:, 0] = handles64 & 4294967295
        handles32[:, 1] = handles64 >> 32
        cls.model_texture_handle_ssbo.write(handles32.tobytes())

        # Append new geometry to the global VBO (re-allocate to grow it)
        existing = cls.global_vbo.read() if cls.global_vbo else b''
        if cls.global_vbo:
            cls.global_vbo.release()
        cls.global_vbo = cls.core.ctx.buffer(existing + new_data.tobytes())

        cls._rebuild_model_vao()

    @classmethod
    def remove_model(cls, label: str):
        if label not in Data.models:
            return

        glb_model      = Data.get_model(label)
        mesh_labels    = [glb_model.name + '#' + mesh for mesh in glb_model.meshes]
        mesh_label_set = set(mesh_labels)

        vbo_data = np.frombuffer(cls.global_vbo.read(), dtype='f4').copy()

        # Remove blocks from highest offset downward so earlier offsets remain valid
        for ml in sorted(mesh_labels, key=lambda ml: cls.meshes[ml]["offset"], reverse=True):
            mesh    = cls.meshes[ml]
            f_start = mesh["offset"] * cls.VERTEX_SIZE
            f_end   = f_start + mesh["count"] * cls.VERTEX_SIZE

            new_data = np.empty(len(vbo_data) - (f_end - f_start), dtype='f4')
            new_data[:f_start] = vbo_data[:f_start]
            new_data[f_start:] = vbo_data[f_end:]
            del vbo_data
            vbo_data = new_data

            # Shift offsets of all meshes that came after the removed block
            for other_label, other_mesh in cls.meshes.items():
                if other_label not in mesh_label_set and other_mesh["offset"] > mesh["offset"]:
                    other_mesh["offset"] -= mesh["count"]

        for ml in mesh_labels:
            cls.meshes.pop(ml, None)
            cls.opaque_instances.pop(ml, None)
            cls.cutout_instances.pop(ml, None)
            cls.transparent_instances.pop(ml, None)

        if cls.meshes:
            last = max(cls.meshes.values(), key=lambda m: m["offset"])
            cls.global_vbo_offset = (last["offset"] + last["count"]) * cls.VERTEX_SIZE
        else:
            cls.global_vbo_offset = 0

        cls.global_vbo.release()
        cls.global_vbo = (
            cls.core.ctx.buffer(vbo_data.tobytes()) if len(vbo_data)
            else cls.core.ctx.buffer(reserve=cls.VERTEX_SIZE * 4)
        )
        del vbo_data

        cls._rebuild_model_vao()

        Data.remove_model(label)

        import gc
        gc.collect()

    @classmethod
    def load_textures(cls, texture_labels: list):
        for label in texture_labels:
            cls.tex_tex_lookup[label] = len(cls.tex_tex_handles)
            cls.tex_tex_handles.append(Data.get_tex(label).get_handle())
        cls.tex_texture_handle_ssbo.clear()
        cls.tex_texture_handle_ssbo.write(np.array(cls.tex_tex_handles, dtype=np.uint64).tobytes())

    # -------------------------------------------------------------------------
    # Draw calls (queued per frame, flushed in render())
    # -------------------------------------------------------------------------

    @classmethod
    def draw_point(cls, radius: float = 1.0,
                   x: float = 0.0, y: float = 0.0, z: float = 0.0,
                   r: float = 1.0, g: float = 1.0, b: float = 1.0, a: float = 1.0):
        """Queue a filled circle (mainly for debugging)."""
        size = radius * Camera.zoom if Camera.perspective == "2d" else radius
        cls.points.extend([size, x, y, z, r, g, b, a])

    @classmethod
    def draw_line(cls, point1: list, point2: list):
        """Queue a line segment. Each point: [x, y, z, r, g, b, a]."""
        cls.lines.extend([*point1, *point2])

    @classmethod
    def _draw_mesh(cls, mesh_label: str, transform: glm.mat4, tint=(1, 1, 1), alpha=1.0):
        """Append one mesh instance to the appropriate alpha bucket."""
        alpha_mode = cls.meshes[mesh_label].get("alpha_mode", "OPAQUE")
        if alpha_mode == "MASK":
            bucket = cls.cutout_instances
        elif alpha < 0.9:
            bucket = cls.transparent_instances
        else:
            bucket = cls.opaque_instances

        m         = np.array(glm.transpose(transform), dtype='f4').flatten()
        tex_data  = np.array([float(cls.meshes[mesh_label].get("tex_index", 0)), 0.0, 0.0, 0.0], dtype='f4')
        tint_data = np.array([*tint, alpha], dtype='f4')
        inst      = np.concatenate([m, tex_data, tint_data]).tobytes()

        if mesh_label not in bucket:
            bucket[mesh_label] = {"transforms": [], "count": 0}
        bucket[mesh_label]["transforms"].append(inst)
        bucket[mesh_label]["count"] += 1

    @classmethod
    def _draw_mesh_persistent(cls, mesh_label: str, transform: glm.mat4,
                            tint=(1, 1, 1), alpha=1.0) -> int:
        """
        Write a mesh instance permanently into the front of the instance SSBO
        and draw command buffer. It will be drawn every frame at zero CPU cost —
        no loop, no re-pack, no re-write.

        Returns a handle for later removal via _unset_persistent().
        """

        mesh      = cls.meshes[mesh_label]
        mat       = Data.get_mesh_material(mesh_label)
        final_alpha = mat["alpha"] * alpha

        # Build the 96-byte instance payload
        m         = np.array(glm.transpose(transform), dtype='f4').flatten()
        tex_data  = np.array([float(mesh.get("tex_index", 0)), 0.0, 0.0, 0.0], dtype='f4')
        tint_data = np.array([*tint, final_alpha], dtype='f4')
        inst_bytes = np.concatenate([m, tex_data, tint_data]).tobytes()

        base_instance = cls._persistent_instance_count
        draw_index    = cls._persistent_draw_count

        # Grow buffers if needed before writing
        while (cls._persistent_instance_count + 1) > cls.MAX_INSTANCES or \
            (cls._persistent_draw_count + 1)    > cls.MAX_DRAWS:
            cls._grow_model_ssbos()

        # Write instance into the persistent slot in the SSBO
        cls.instance_ssbo.write(inst_bytes, offset=base_instance * cls.INSTANCE_SIZE)

        # Write material into its persistent slot
        mat_array = np.array(
            [*mat["ambient"], 0.0, *mat["diffuse"], 0.0, *mat["specular"], mat["shininess"]],
            dtype='f4'
        )
        cls.material_ssbo.write(mat_array.tobytes(), offset=draw_index * cls.MATERIAL_SIZE)

        # Write base_instance into its persistent slot
        cls.base_instance_ssbo.write(
            np.array([base_instance], dtype='u4').tobytes(),
            offset=draw_index * 4
        )

        # Write draw command into its persistent slot
        # [count, instanceCount, firstIndex, baseVertex, baseInstance]
        cmd = np.array([mesh["count"], 1, 0, mesh["offset"], 0], dtype='u4')
        cls.draw_cmd_buffer.write(cmd.tobytes(), offset=draw_index * 20)

        cls._persistent_instance_count += 1
        cls._persistent_draw_count     += 1

        handle = cls._persistent_handle_counter
        cls._persistent_handle_counter += 1

        cls.persistent_draw_entries[handle] = {
            "mesh_label":    mesh_label,
            "base_instance": base_instance,
            "draw_index":    draw_index,
            "inst_bytes":    inst_bytes,   # kept for potential _update_persistent
        }

        return handle
    
    @classmethod
    def _update_persistent(cls, handle: int, transform: list = None,
                            tint=None, alpha: float = None):
        """
        Patch a persistent instance's payload in-place — only rewrites the
        bytes that changed. No bucket rebuild, no full SSBO rewrite.
        """
        entry     = cls.persistent_draw_entries[handle]
        mesh      = cls.meshes[entry["mesh_label"]]
        mat       = Data.get_mesh_material(entry["mesh_label"])

        # Unpack the stored bytes so we can patch selectively
        old = np.frombuffer(entry["inst_bytes"], dtype='f4').copy()

        if transform is not None:
            old[:16] = np.array(glm.transpose(glm.mat4(transform)), dtype='f4').flatten()
        if tint is not None or alpha is not None:
            cur_tint  = tuple(old[20:23]) if tint  is None else tint
            cur_alpha = float(old[23])    if alpha is None else (mat["alpha"] * alpha)
            old[20:24] = [*cur_tint, cur_alpha]

        new_bytes = old.tobytes()
        entry["inst_bytes"] = new_bytes
        cls.instance_ssbo.write(new_bytes, offset=entry["base_instance"] * cls.INSTANCE_SIZE)

    @classmethod
    def _unset_persistent(cls, handle: int):
        """
        Remove a persistent instance. Because the SSBO is a flat packed array,
        removal shifts all entries that came after it down by one slot — their
        SSBO offsets, draw command offsets, and base_instance values are all
        patched in one contiguous rewrite of the affected region.
        """
        if handle not in cls.persistent_draw_entries:
            return

        removed = cls.persistent_draw_entries.pop(handle)
        bi      = removed["base_instance"]
        di      = removed["draw_index"]

        # Read the full persistent regions
        inst_bytes = bytearray(cls.instance_ssbo.read(
            size=cls._persistent_instance_count * cls.INSTANCE_SIZE
        ))
        cmd_bytes  = bytearray(cls.draw_cmd_buffer.read(
            size=cls._persistent_draw_count * 20
        ))
        mat_bytes  = bytearray(cls.material_ssbo.read(
            size=cls._persistent_draw_count * cls.MATERIAL_SIZE
        ))
        base_bytes = bytearray(cls.base_instance_ssbo.read(
            size=cls._persistent_draw_count * 4
        ))

        # Excise the removed slot from each buffer
        IS = cls.INSTANCE_SIZE
        inst_bytes = inst_bytes[:bi * IS] + inst_bytes[(bi + 1) * IS:]

        MS = cls.MATERIAL_SIZE
        mat_bytes  = mat_bytes[:di * MS]  + mat_bytes[(di + 1) * MS:]
        base_bytes = base_bytes[:di * 4]  + base_bytes[(di + 1) * 4:]
        cmd_bytes  = cmd_bytes[:di * 20]  + cmd_bytes[(di + 1) * 20:]

        cls._persistent_instance_count -= 1
        cls._persistent_draw_count     -= 1

        # Patch every entry that shifted due to the removal
        for entry in cls.persistent_draw_entries.values():
            if entry["base_instance"] > bi:
                entry["base_instance"] -= 1
            if entry["draw_index"] > di:
                entry["draw_index"] -= 1

        # Rebuild base_instance values in the buffer to reflect the shift
        base_arr = np.frombuffer(base_bytes, dtype='u4').copy()
        for i in range(len(base_arr)):
            if base_arr[i] > bi:
                base_arr[i] -= 1
        base_bytes = base_arr.tobytes()

        # Write everything back
        cls.instance_ssbo.write(bytes(inst_bytes))
        cls.draw_cmd_buffer.write(bytes(cmd_bytes))
        cls.material_ssbo.write(bytes(mat_bytes))
        cls.base_instance_ssbo.write(base_bytes)

    @classmethod
    def draw_model(cls, label: str, transform: glm.mat4, tint=(1, 1, 1), alpha=1.0, persistent: bool = False):
        glb_model = Data.get_model(label)
        persistent_handles = []
        for mesh in glb_model.meshes:
            final_label  = glb_model.name + '#' + mesh
            final_alpha  = Data.get_mesh_material(final_label)["alpha"] * alpha
            if persistent:
                handle = cls._draw_mesh_persistent(final_label, transform, tint, final_alpha)
                persistent_handles.append(handle)
            else:
                cls._draw_mesh(final_label, transform, tint, final_alpha)
        if persistent:
            return list(persistent_handles)


    @classmethod
    def draw_tex(cls, transform, label: int = 0, tint=(1, 1, 1), alpha: float = 1.0):
        """Queue a texture sprite."""
        m    = np.array(glm.transpose(glm.mat4(transform)), dtype='f4').flatten()
        inst = np.concatenate([
            m,
            np.array([float(cls.tex_tex_lookup[label]), 0.0, 0.0, 0.0], dtype='f4'),
            np.array([*tint, alpha], dtype='f4'),
        ]).tobytes()
        if alpha >= 0.9:
            cls.tex_opaque_queue.append(inst)
        else:
            cls.tex_transparent_queue.append(inst)

    @classmethod
    def draw_axes(cls):
        """Draw the debug coordinate axes (X=red, Y=green, Z=blue)."""
        cls.core.ctx.line_width = 3
        cx, cy, cz = Camera.pos.x, Camera.pos.y, Camera.pos.z
        far = Camera.far
        cls.draw_line([cx - far, 0, 0, 1, 0, 0, 0], [cx,       0, 0, 1, 0, 0, 1])
        cls.draw_line([cx,       0, 0, 1, 0, 0, 1], [cx + far, 0, 0, 1, 0, 0, 0])
        cls.draw_line([0, cy - far, 0, 0, 1, 0, 0], [0, cy,       0, 0, 1, 0, 1])
        cls.draw_line([0, cy,       0, 0, 1, 0, 1], [0, cy + far, 0, 0, 1, 0, 0])
        cls.draw_line([0, 0, cz - far, 0, 0, 1, 0], [0, 0, cz,       0, 0, 1, 1])
        cls.draw_line([0, 0, cz,       0, 0, 1, 1], [0, 0, cz + far, 0, 0, 1, 0])

    @classmethod
    def render(cls):
        if not cls._enabled:
            return

        ctx = cls.core.ctx

        if Camera._updated_projection:
            cls._update_projection()
            Camera._updated_projection = False

        # Skybox — drawn with depth writes off so it never occludes geometry
        if cls.skybox_enabled and cls.skybox_cubemap is not None and Camera.perspective == "3d":
            cls.skybox_prog['view'].write(np.array(Camera.view.to_list(), dtype='f4').tobytes())
            cls.skybox_cubemap.use(location=1)
            cls.skybox_prog['skybox'].value = 1
            ctx.depth_mask = False
            cls.skybox_vao.render()
            ctx.depth_mask = True

        num_opaque  = len(cls.opaque_instances)
        num_cutout  = len(cls.cutout_instances)

        # Pack all draw buckets in order: opaque → cutout → transparent.
        # Solid geometry writes depth first, so alpha-tested meshes never
        # incorrectly occlude opaque ones.
        combined_buckets = (
            list(cls.opaque_instances.items()) +
            list(cls.cutout_instances.items()) +
            list(cls.transparent_instances.items())
        )

        material_data  = []
        base_instances = []
        commands       = []
        all_instances  = []
        current_base   = cls._persistent_instance_count

        for label, info in combined_buckets:
            mat  = Data.get_mesh_material(label)
            mesh = cls.meshes[label]
            material_data.extend([*mat["ambient"], 0.0, *mat["diffuse"], 0.0, *mat["specular"], mat["shininess"]])
            commands.extend([mesh["count"], info["count"], 0, mesh["offset"], 0])
            base_instances.append(current_base)
            all_instances.append(b''.join(info["transforms"]))
            current_base += info["count"]

        num_transparent = len(cls.transparent_instances)

        total_instances = sum(info["count"] for _, info in combined_buckets)
        total_draws     = len(combined_buckets)

        while total_instances > cls.MAX_INSTANCES or total_draws > cls.MAX_DRAWS:
            cls._grow_model_ssbos()

        # Write dynamic data into the region immediately after the persistent block
        dyn_inst_offset = cls._persistent_instance_count * cls.INSTANCE_SIZE
        dyn_draw_offset = cls._persistent_draw_count

        if all_instances:
            cls.instance_ssbo.write(b''.join(all_instances), offset=dyn_inst_offset)

        if combined_buckets:
            cls.material_ssbo.write(
                np.array(material_data, dtype='f4').tobytes(),
                offset=dyn_draw_offset * cls.MATERIAL_SIZE
            )
            cls.base_instance_ssbo.write(
                np.array(base_instances, dtype='u4').tobytes(),
                offset=dyn_draw_offset * 4
            )
            cls.draw_cmd_buffer.write(
                np.array(commands, dtype='u4').tobytes(),
                offset=dyn_draw_offset * 20
            )

        cls._upload_view()
        cls._upload_lights()

        p = cls._persistent_draw_count

        # Opaque pass — persistent draws + dynamic opaque draws, depth writes on
        if p + num_opaque > 0:
            ctx.depth_mask = True
            cls.model_prog['u_draw_id_offset'].value = 0
            cls.model_vao.render_indirect(cls.draw_cmd_buffer, count=p + num_opaque)

        # Cutout pass — starts immediately after opaque block
        if num_cutout > 0:
            ctx.depth_mask = True
            cls.model_prog['u_draw_id_offset'].value = p + num_opaque
            cls.model_vao.render_indirect(
                cls.draw_cmd_buffer,
                count=num_cutout,
                first=p + num_opaque
            )

        # Opaque sprites
        if cls.tex_opaque_queue:
            n = len(cls.tex_opaque_queue)
            cls.tex_instance_ssbo.write(b''.join(cls.tex_opaque_queue))
            ctx.depth_mask = True
            cls.tex_vao.render(mgl.TRIANGLES, vertices=n * 6)
            cls.tex_opaque_queue.clear()

        # Primitives — always on top, depth writes off
        ctx.depth_mask = False

        if cls.lines:
            n_lines = len(cls.lines) // 14
            while n_lines > cls.MAX_LINES:
                cls._grow_line_buffer()
            cls.line_vbo.write(np.array(cls.lines, dtype='f4').tobytes())
            cls.line_prog['view'].write(np.array(Camera.view.to_list(), dtype='f4').tobytes())
            cls.line_vao.render(mgl.LINES, vertices=n_lines * 2)
            cls.lines.clear()

        if cls.points:
            n_points = len(cls.points) // 8
            while n_points > cls.MAX_POINTS:
                cls._grow_point_buffer()
            cls.point_vbo.write(np.array(cls.points, dtype='f4').tobytes())
            cls.point_prog['view'].write(np.array(Camera.view.to_list(), dtype='f4').tobytes())
            cls.point_vao.render(mgl.POINTS, vertices=n_points)
            cls.points.clear()

        # Transparent models — two-pass (back faces then front faces)
        if num_transparent > 0:
            ctx.depth_mask = False
            ctx.enable(mgl.CULL_FACE)
            first_t = p + num_opaque + num_cutout

            ctx.cull_face = 'front'
            cls.model_prog['u_draw_id_offset'].value = first_t
            cls.model_vao.render_indirect(cls.draw_cmd_buffer, count=num_transparent, first=first_t)

            ctx.cull_face = 'back'
            cls.model_vao.render_indirect(cls.draw_cmd_buffer, count=num_transparent, first=first_t)

            ctx.disable(mgl.CULL_FACE)
            ctx.depth_mask = True

        # Transparent sprites
        if cls.tex_transparent_queue:
            n = len(cls.tex_transparent_queue)
            cls.tex_instance_ssbo.write(b''.join(cls.tex_transparent_queue))
            ctx.depth_mask = False
            cls.tex_vao.render(mgl.TRIANGLES, vertices=n * 6)
            ctx.depth_mask = True
            cls.tex_transparent_queue.clear()

        cls.opaque_instances.clear()
        cls.cutout_instances.clear()
        cls.transparent_instances.clear()


class LightSource:

    def __init__(self, position: glm.vec3, color: glm.vec3, intensity: float = 0.5):
        self.pos       = position
        self.color     = color
        self.intensity = intensity
        if len(Renderer.lights) < 8:
            Renderer.lights.append(self)

    def destroy(self):
        if self in Renderer.lights:
            Renderer.lights.remove(self)