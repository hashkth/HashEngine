from .render import *

# Units:
# Time   - seconds
# Length - metres
# Mass   - kilograms

class Entity:

    def __init__(self, state):
        self.state = state

        self.model_name = None
        self.visible    = True
        self.persistent = False

        self.pos   = glm.vec3(0, 0, 0)
        self.scale = glm.vec3(1, 1, 1)
        self.tint  = glm.vec3(1, 1, 1)
        self.alpha = 1.0

        self.pitch = 0.0   # rotation around X (degrees)
        self.yaw   = 0.0   # rotation around Y (degrees)
        self.roll  = 0.0   # rotation around Z (degrees)

        self.vel = glm.vec3(0, 0, 0)
        self.acc = glm.vec3(0, 0, 0)

        # Pre-compute the model matrix and mark it clean.
        self._dirty = False
        self._model_mat = glm.mat4(1.0)

    def physics(self):
        """Integrate velocity and position.  Call once per frame before render."""
        if self.acc.x or self.acc.y or self.acc.z:
            self.vel += self.acc * Clock.dt
        if self.vel.x or self.vel.y or self.vel.z:
            self.pos  += self.vel * Clock.dt
            self._dirty = True

    def _rebuild_matrix(self):
        mat = glm.mat4(1.0)
        mat = glm.translate(mat, self.pos)
        mat = glm.rotate(mat, glm.radians(self.yaw),   glm.vec3(0, 1, 0))
        mat = glm.rotate(mat, glm.radians(self.pitch), glm.vec3(0, 0, 1))
        mat = glm.rotate(mat, glm.radians(self.roll),  glm.vec3(1, 0, 0))
        mat = glm.scale(mat, self.scale)
        self._model_mat = mat

    def process(self):
        pass

    def render(self):
        if not self.visible or self.model_name is None:
            return
        if self._dirty:
            self._rebuild_matrix()
            self._dirty = False
        Renderer.draw_model(
                        self.model_name,
                        self._model_mat,
                        (self.tint.x, self.tint.y, self.tint.z),
                        self.alpha,
                        self.persistent
                    )

    def set_pos(self, x: float, y: float, z: float):
        self.pos.x, self.pos.y, self.pos.z = x, y, z
        self._dirty = True

    def set_scale(self, x: float, y: float, z: float):
        self.scale.x, self.scale.y, self.scale.z = x, y, z
        self._dirty = True

    def set_rotation(self, pitch: float = None, yaw: float = None, roll: float = None):
        if pitch is not None: self.pitch = pitch
        if yaw   is not None: self.yaw   = yaw
        if roll  is not None: self.roll  = roll
        self._dirty = True