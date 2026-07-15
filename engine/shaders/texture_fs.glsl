#version 330 core
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_bindless_texture : require

layout(std430, binding = 5) buffer TextureHandles { uvec2 handles[]; };

in vec2      v_uv;
flat in int  v_tex_id;
in vec4      v_tint_alpha;

out vec4 fragColor;

vec4 sampleTexture(int id, vec2 uv) {
    uvec2 handle = handles[id];
    sampler2D tex = sampler2D(handle);
    return texture(tex, uv);
}

void main() {
    vec4 tex_color = sampleTexture(v_tex_id, v_uv);
    fragColor = vec4(tex_color.rgb * v_tint_alpha.rgb, tex_color.a * v_tint_alpha.w);
}
