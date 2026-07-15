#version 330

in vec3 v_texcoord;
out vec4 fragColor;

uniform samplerCube skybox;

void main() {
    fragColor = texture(skybox, v_texcoord);
}
