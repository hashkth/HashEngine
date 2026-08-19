from .render import *

# Units:
# Time   - seconds
# Length - metres
# Mass   - kilograms
# Angle  - degree

class Entity:

    def __init__(self, state):
        self.state = state
        
        self.visible = True
        self._dirty = False
        
        self.pos   = glm.vec3(0, 0, 0)
        self.scale = glm.vec3(1, 1, 1)
        self.tint  = glm.vec3(1, 1, 1)
        self.alpha = 1.0

        self.vel = glm.vec3(0, 0, 0)
        self.acc = glm.vec3(0, 0, 0)

    def physics(self):
        """Compute physics behavior"""
        if self.acc.x or self.acc.y or self.acc.z:
            self.vel += self.acc * Clock.dt
        if self.vel.x or self.vel.y or self.vel.z:
            self.pos  += self.vel * Clock.dt
            self._dirty = True

    def process(self):
        pass

    def render(self):
        pass


class Entity2D(Entity):

    def __init__(self, state):
        super().__init__(state)
        self.tex_label = None
        self.layer = 0  # greater layers rendered on top
        self.angle = 0  # positive is anti-clockwise
        self._transform = glm.mat4(1.0)

    def _rebuild_transform(self):
        mat = glm.mat4(1.0)
        mat = glm.translate(mat, self.pos)
        mat = glm.rotate(mat, glm.radians(self.angle), glm.vec3(0, 0, 1))
        mat = glm.scale(mat, self.scale)
        self._transform = mat

    def process(self):
        if self._dirty:
            self._rebuild_transform()
            self._dirty = False

    def render(self):
        if not self.visible or self.tex_label is None:
            return
        Renderer.draw_tex(
                            self.tex_label,
                            self._transform,
                            self.layer,
                            self.tint,
                            self.alpha
                        )

    def set_tex(self, tex_label: str):
        self.tex_label = tex_label
        self.scale.x, self.scale.y = Data.get_tex(self.tex_label).size
        self._dirty = True

    def set_pos(self, x: float, y: float):
        self.pos.x, self.pos.y = x, y
        self._dirty = True

    def set_vel(self, x: float, y: float):
        self.vel.x, self.vel.y = x, y

    def set_acc(self, x: float, y: float):
        self.acc.x, self.acc.y = x, y

    def set_scale(self, x: float, y: float):
        self.scale.x, self.scale.y = x, y
        self._dirty = True

    def set_angle(self, angle: float):
        self.angle = angle
        self._dirty = True


class Entity3D(Entity):

    def __init__(self, state):
        super().__init__(state)
        self.model_label = None
        self.persistent = False
        self.pitch = 0.0   # rotation around X (degrees)
        self.yaw   = 0.0   # rotation around Y (degrees)
        self.roll  = 0.0   # rotation around Z (degrees)
        self._transform = glm.mat4(1.0)

    def _rebuild_transform(self):
        mat = glm.mat4(1.0)
        mat = glm.translate(mat, self.pos)
        mat = glm.rotate(mat, glm.radians(self.yaw),   glm.vec3(0, 1, 0))
        mat = glm.rotate(mat, glm.radians(self.pitch), glm.vec3(0, 0, 1))
        mat = glm.rotate(mat, glm.radians(self.roll),  glm.vec3(1, 0, 0))
        mat = glm.scale(mat, self.scale)
        self._transform = mat

    def set_model(self, model_label: str):
        self.model_label = model_label

    def process(self):
        if self._dirty:
            self._rebuild_transform()
            self._dirty = False

    def render(self):
        if not self.visible or self.model_label is None:
            return
        Renderer.draw_model(
                            self.model_label,
                            self._transform,
                            (self.tint.x, self.tint.y, self.tint.z),
                            self.alpha,
                            self.persistent
                        )

    def set_pos(self, x: float, y: float, z: float):
        self.pos.x, self.pos.y, self.pos.z = x, y, z
        self._dirty = True

    def set_vel(self, x: float, y: float, z: float):
        self.vel.x, self.vel.y, self.vel.z = x, y, z

    def set_acc(self, x: float, y: float, z: float):
        self.acc.x, self.acc.y, self.acc.z = x, y, z

    def set_scale(self, x: float, y: float, z: float):
        self.scale.x, self.scale.y, self.scale.z = x, y, z
        self._dirty = True

    def set_rotation(self, pitch: float = None, yaw: float = None, roll: float = None):
        if pitch is not None: self.pitch = pitch
        if yaw   is not None: self.yaw   = yaw
        if roll  is not None: self.roll  = roll
        self._dirty = True