#version 330 core
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_draw_parameters : require
#extension GL_ARB_bindless_texture : require

in vec2 in_uv;
in vec3 in_pos;
in vec3 in_norm;

uniform mat4 view;
uniform mat4 projection;
uniform int u_draw_id_offset; // NEW: Crucial for multi-pass indexing

struct Instance {
    mat4 transform;
    vec4 tex_data;
    vec4 tint_alpha;
};

layout(std430, binding = 0) buffer InstanceBuffer { Instance instances[]; };
layout(std430, binding = 1) buffer BaseInstanceBuffer { uint base_instances[]; };

out vec2 uv;
out vec3 v_pos;
out vec3 v_normal;
flat out int v_tex_id;
out vec4 v_tint_alpha;
flat out int v_draw_id;

void main() {
    // gl_DrawIDARB starts at 0 for every render_indirect call.
    // We add the offset to find the correct data in our combined SSBOs.
    uint draw_id = uint(gl_DrawIDARB + u_draw_id_offset);
    
    uint base = base_instances[draw_id];
    Instance inst = instances[base + uint(gl_InstanceID)];

    vec4 world_pos = inst.transform * vec4(in_pos, 1.0);
    gl_Position = projection * view * world_pos;

    v_pos = world_pos.xyz;
    uv = in_uv;
    mat3 normal_mat = transpose(inverse(mat3(inst.transform)));
    v_normal = normal_mat * in_norm;
    v_tex_id = int(inst.tex_data.x);
    v_tint_alpha = inst.tint_alpha;
    v_draw_id = int(draw_id); 
}