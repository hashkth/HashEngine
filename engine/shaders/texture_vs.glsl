#version 330 core
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_bindless_texture : require

uniform mat4 projection;
uniform mat4 view;

struct TexInstance {
    mat4  transform;   // model matrix (encodes pos/size/rotation)
    vec4  tex_data;    // x = tex_id (float-cast int), yzw = pad
    vec4  tint_alpha;  // rgb = tint, w = alpha
};

layout(std430, binding = 3) buffer TexInstanceBuffer { TexInstance tex_instances[]; };

out vec2      v_uv;
flat out int  v_tex_id;
out vec4      v_tint_alpha;

const vec2 POSITIONS[6] = vec2[6](
    vec2(-0.5,  0.5),   // TL
    vec2(-0.5, -0.5),   // BL
    vec2( 0.5, -0.5),   // BR

    vec2(-0.5,  0.5),   // TL
    vec2( 0.5, -0.5),   // BR
    vec2( 0.5,  0.5)    // TR
);

const vec2 UVS[6] = vec2[6](
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),

    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);

void main() {
    int inst_id = gl_VertexID / 6;
    int vert_id = gl_VertexID % 6;

    TexInstance inst = tex_instances[inst_id];

    vec4 world_pos = inst.transform * vec4(POSITIONS[vert_id], 0.0, 1.0);
    gl_Position    = projection * view * world_pos;

    v_uv         = UVS[vert_id];
    v_tex_id     = int(inst.tex_data.x);
    v_tint_alpha = inst.tint_alpha;
}
