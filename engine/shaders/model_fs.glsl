#version 330 core
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_draw_parameters : require
#extension GL_ARB_bindless_texture : require


struct Light {
    vec3 position;
    float intensity;  // packed with position's padding slot
    vec3 color;
    float _pad;       // explicit padding to keep alignment clean
};

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

#define MAX_LIGHTS 8
uniform Light lights[MAX_LIGHTS];
uniform int num_lights;
uniform vec3 view_pos;
uniform vec3 ambient_light;
uniform bool use_spec;

layout(std430, binding = 2) buffer MaterialBuffer { Material materials[]; };
layout(std430, binding = 4) buffer TextureHandles { uvec2 handles[]; };

in vec2 uv;
in vec3 v_pos;
in vec3 v_normal;
flat in int v_tex_id;
in vec4 v_tint_alpha;
flat in int v_draw_id;

out vec4 fragColor;

// Returns full RGBA sample from a bindless texture handle
vec4 sampleTexture(int id, vec2 uv) {
    uvec2 handle = handles[id];
    sampler2D tex = sampler2D(handle);
    return texture(tex, uv);
}

void main() {
    Material mat = materials[v_draw_id];
    vec3 norm = normalize(v_normal);
    if (!gl_FrontFacing) norm = -norm;
    vec3 view_dir = normalize(view_pos - v_pos);

    vec4 texSample = sampleTexture(v_tex_id, uv);
    vec3 albedo = mat.diffuse * texSample.rgb;
    vec3 result = ambient_light * mat.ambient * albedo * 10;

    for (int i = 0; i < num_lights; i++) {
        vec3 light_dir = normalize(lights[i].position - v_pos);
        
        // Attenuation
        float dist = length(lights[i].position - v_pos);
        float attenuation = 1.0 / max(dist * dist, 0.01);
        vec3 radiance = lights[i].color * lights[i].intensity * attenuation;

        // Diffuse
        float diff = max(dot(norm, light_dir), 0.0);
        result += diff * albedo * radiance;

        // Blinn-Phong Specular
        if (use_spec) {
            vec3 halfway = normalize(light_dir + view_dir);
            float spec = pow(max(dot(norm, halfway), 0.0), mat.shininess);
            result += spec * mat.specular * radiance;
        }
    }

    // Combine texture alpha with tint alpha
    float alpha = texSample.a * v_tint_alpha.w;
    if (alpha < 0.02) discard;
    fragColor = vec4(result * v_tint_alpha.rgb, alpha);
}
