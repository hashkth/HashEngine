#version 330

in vec3 in_position;
out vec3 v_texcoord;

uniform mat4 projection;
uniform mat4 view;

void main() {
    mat4 rot_view = mat4(mat3(view));
    vec4 pos = projection * rot_view * vec4(in_position, 1.0);
    gl_Position = pos.xyww;  // force depth = 1.0
    v_texcoord = in_position;
}
