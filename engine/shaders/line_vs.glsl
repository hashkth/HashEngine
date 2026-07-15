#version 330 core

in vec3 in_pos;
in vec4 in_col;

out vec4 out_col;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(in_pos, 1.0);
    out_col = in_col;
}