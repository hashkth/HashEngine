
from .ecs import *


class Core:

    win = None
    ctx = None
    imgui_io = None
    render_targets = {}

    states = {}
    active_s = None
    queued_s = None

    w, h = 0, 0
    hw, hh = 0, 0 
    qw, qh = 0, 0
    console_open = False
    logs = []

    @classmethod
    def init(cls, width: int, height: int, title: str):
        cls.w, cls.h = width, height
        cls.hw, cls.hh = cls.w // 2, cls.h // 2
        cls.qw, cls.qh = cls.w // 4, cls.h // 4

        cls.win = glux.Window(width, height, title, gl_major=3, gl_minor=3)
        
        cls.win.set_events_callback(cls.events)
        cls.win.set_process_callback(cls.process)
        cls.win.set_render_callback(cls.render)
        cls.win.set_render_ui_callback(cls.render_ui)

        cls.ctx = mgl.create_context()
        cls.imgui_io = imgui.get_io()

        Clock.init(cls)
        Data.init(cls)
        try:
            Audio.init(cls)
        except:
            print("Audio Device not found! Plug in an audio device")
        Camera.init(cls)
        Renderer.init(cls)

    @classmethod
    def add_render_target(cls, label: str, width: int, height: int, channels: int=4, filter_flags: tuple=(mgl.NEAREST, mgl.NEAREST)):
        cls.render_targets[label] = [cls.ctx.texture((width, height), channels), None]
        cls.render_targets[label][0].filter = filter_flags
        cls.render_targets[label][1] = cls.ctx.framebuffer(color_attachments=[cls.render_targets[label][0]])

    @classmethod
    def delete_render_target(cls, label: str):
        if label in cls.render_targets.keys():
            cls.render_targets[label][0].release()
            cls.render_targets[label][1].release()
            del cls.render_targets[label]

    @classmethod
    def set_render_target(cls, label: str):
        cls.render_targets[label][1].use()

    @classmethod
    def reset_render_target(cls):
        cls.ctx.screen.use()

    @classmethod
    def get_render_target_tex(cls, label: str):
        return cls.render_targets[label][0]

    @classmethod
    def add_state(cls, state: State):
        for s in cls.states.values():
            if isinstance(s, state):
                print("State already added")
                return
        new_state = state()
        cls.states[new_state.id] = new_state

    @classmethod
    def remove_state(cls, state_id: any):
        if state_id in cls.states.keys():
            if cls.active_s:
                if cls.active_s.id == state_id:
                    cls.active_s.exit()
                    cls.active_s = None
            cls.states[state_id].cleanup()
            del cls.states[state_id]

    @classmethod
    def activate_state(cls, state_id):
        cls.queued_s = cls.states[state_id]

    @classmethod
    def log(cls, message: str):
        cls.logs.insert(0, str(message))
        if len(cls.logs) > 100: cls.logs.pop()

    @classmethod
    def run(cls):
        cls.win.run()
        cls.win.close()
        al.shutdown()

    @classmethod
    def events(cls):
        if cls.queued_s:
            if cls.active_s:
                cls.active_s.exit()
            cls.active_s = cls.queued_s
            cls.active_s.enter()
            cls.queued_s = None

        if glux.keyboard.action == glux.actions.PRESS:
            if glux.keyboard.key == glux.keys.K_GRAVE_ACCENT:
                cls.console_open = not cls.console_open
        if cls.active_s:
            cls.active_s.events()

    @classmethod
    def process(cls):
        Clock.process()
        Audio.process()
        Camera.process()

        cls.w, cls.h = cls.win.get_size()
        cls.hw, cls.hh = math.ceil(cls.w) // 2, math.ceil(cls.h) // 2
        cls.qw, cls.qh = math.ceil(cls.w) // 4, math.ceil(cls.h) // 4
        
        if cls.active_s:
            cls.active_s.process()
            TK_ROOT.update()

    @classmethod
    def render(cls):
        cls.ctx.clear(0.0, 0.0, 0.0, 1.0)
        if cls.active_s:
            cls.active_s.render()
        # ModelRenderer.render()
        Renderer.render()

    @classmethod
    def render_ui(cls):
        if cls.active_s:
            cls.active_s.render_ui()
        if cls.console_open:
            imgui.push_style_var_float(imgui.StyleVar.Alpha, 0.75)
            imgui.set_next_window_pos(imgui.Vec2(0, cls.h - cls.qh))
            imgui.set_next_window_size(imgui.Vec2(cls.w, cls.qh))
            _, cls.console_open = imgui.begin("Console", cls.console_open, flags=imgui.WindowFlags.NoCollapse | imgui.WindowFlags.NoResize)
            for log in cls.logs:
                imgui.text(log)
            imgui.end()
            imgui.pop_style_var()