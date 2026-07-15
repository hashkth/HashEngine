
from .audio import *


class Camera:

    """
    Zooming works and makes sense only in 2D perspective
    """

    core = None

    pos = glm.vec3(0, 0, 0)
    front = glm.vec3(0, 0, -1)
    up = glm.vec3(0, 1, 0)
    right = glm.normalize(glm.cross(front, up))

    target = glm.vec3(0, 0, 0)

    view = glm.lookAt(pos, pos + front, up)
    projection = None

    far = 1000
    _far = 1000
    fov = 45
    _fov = 45
    yaw = -90
    pitch = 0
    final_yaw = -90
    final_pitch = 0
    sensitivity = 0.05
    smoothing = 1
    speed = 5

    mode = "free"
    mouse_enabled = False

    _mlastx, _mlasty, _moffx, _moffy = 0, 0, 0, 0
    _first_mouse_use = True

    # Used for smoothing camera movement
    # front is the real raw value based on which _front is a smoothed variation
    _front = glm.vec3()

    zoom = 1.0
    _zoom = 1.0
    perspective = "3d"
    _updated_projection = False

    @classmethod
    def init(cls, core):
        cls.core = core
        cls.update_projection()

    @classmethod
    def update_projection(cls):
        cls._updated_projection = True
        if cls.perspective == "2d":
            half_w = (cls.core.w / 2) / cls.zoom
            half_h = (cls.core.h / 2) / cls.zoom
            cls.projection = glm.ortho(
                -half_w, half_w,
                -half_h, half_h,
                -100, 100
            )
        elif cls.perspective == "3d":
            cls.projection = glm.perspective(glm.radians(cls.fov), cls.core.w/cls.core.h, 0.1, cls.far)

    @classmethod
    def set_mode(cls, mode: str):
        cls.mode = mode
        if mode == "free":
            cls._first_mouse_use = True

    @classmethod
    def set_perspective(cls, perspective: str):
        cls.perspective = perspective
        cls.update_projection()
    
    @classmethod
    def enable_mouse(cls):
        cls.mouse_enabled = True

    @classmethod
    def disable_mouse(cls):
        cls.mouse_enabled = False

    @classmethod
    def set_target(cls, target: glm.vec3):
        cls.target = target

    @classmethod
    def generic_movement(cls):
        if glux.keyboard.held(glux.keys.K_W):
            cls.pos +=  cls.speed * cls.front * Clock.dt
        if glux.keyboard.held(glux.keys.K_S):
            cls.pos += -cls.speed * cls.front * Clock.dt
        if glux.keyboard.held(glux.keys.K_D):
            cls.pos +=  cls.right   * cls.speed * Clock.dt
        if glux.keyboard.held(glux.keys.K_A):
            cls.pos -=  cls.right   * cls.speed * Clock.dt

    @classmethod
    def process(cls):
        cls.zoom = max(cls.zoom, 0.1)
        if cls.far != cls._far or cls.fov != cls._fov or cls.zoom != cls._zoom:
            cls.update_projection()
            cls._far = cls.far
            cls._fov = cls.fov
            cls._zoom = cls.zoom

        if cls.perspective == "2d":
            cls.yaw = -90
            cls.pitch = 0
            cls.view = glm.translate(glm.mat4(1.0), -cls.pos)

        elif cls.perspective == "3d":
            smooth = max(cls.smoothing, 1.0)
            cls.final_yaw += ((cls.yaw - cls.final_yaw + 180) % 360 - 180) / smooth
            cls.final_pitch += (cls.pitch - cls.final_pitch) / smooth

            cls.right = glm.normalize(glm.cross(cls.front, cls.up))
            
            if cls.mode == "free":
                if cls.mouse_enabled:
                    if cls._first_mouse_use:
                        glux.cursor.set_pos(0, 0)
                        cls._first_mouse_use = False
                        cls._mlastx = glux.cursor.x
                        cls._mlasty = glux.cursor.y
                    cls._moffx = cls.sensitivity * (glux.cursor.x - cls._mlastx)
                    cls._moffy = cls.sensitivity * (glux.cursor.y - cls._mlasty)
                    cls._mlastx = glux.cursor.x
                    cls._mlasty = glux.cursor.y
                else:
                    cls._moffx = 0
                    cls._moffy = 0
                    cls._first_mouse_use = True

                cls.yaw += cls._moffx
                cls.pitch += cls._moffy
                cls.pitch = max(min(cls.pitch, 89), -89)

                cls.front.x = math.cos(glm.radians(cls.final_yaw)) * math.cos(glm.radians(cls.final_pitch))
                cls.front.y = math.sin(glm.radians(cls.final_pitch))
                cls.front.z = math.sin(glm.radians(cls.final_yaw)) * math.cos(glm.radians(cls.final_pitch))
                cls.front = glm.normalize(cls.front)
                cls.view = glm.lookAt(cls.pos, cls.pos + cls.front, cls.up)
            
            elif cls.mode == "follow":
                direction = cls.target - cls.pos
                cls.front = glm.normalize(direction)
                cls.pitch = glm.degrees(math.asin(glm.clamp(cls.front.y, -1.0, 1.0)))
                cls.yaw = glm.degrees(math.atan2(cls.front.z, cls.front.x))
                cls._front.x = math.cos(glm.radians(cls.final_yaw)) * math.cos(glm.radians(cls.final_pitch))
                cls._front.y = math.sin(glm.radians(cls.final_pitch))
                cls._front.z = math.sin(glm.radians(cls.final_yaw)) * math.cos(glm.radians(cls.final_pitch))
                cls._front = glm.normalize(cls._front)
                cls.view = glm.lookAt(cls.pos, cls.pos + cls._front, cls.up)