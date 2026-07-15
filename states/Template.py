
from entities import *

class Template(State):

    # Template

    def __init__(self):
        super().__init__("Template")

    def enter(self):
        Core.imgui_io.font_global_scale = 2
        Camera.set_perspective("3d")
        Camera.pos.z = 5
        Camera.smoothing = 4

    def exit(self):
        Core.imgui_io.font_global_scale = 1

    def events(self):
        if glux.keyboard.held(glux.keys.K_ESCAPE):
            Core.win.close()

    def process(self):
        pass
    
    def render(self):
        Renderer.draw_axes()

    def render_ui(self):
        pass

Core.add_state(Template)