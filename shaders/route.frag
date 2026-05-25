#version 450

// Solid colour for now — bright magenta reads well against grass/rock/snow.
// Push-constant the colour later if we want gradient-by-sun-hours.
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.20, 0.55, 1.0);
}
