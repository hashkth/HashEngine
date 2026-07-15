#version 330 core

in vec4 out_col;

out vec4 fragColor;

void main()
{
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) {
        discard;
    }
    fragColor = out_col;
}