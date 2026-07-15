from .model import *


class Data:

    """
    A unified Global Resource Manager\n
    Tightly integrated with the internal Engine components\n
    Loads up stuff for every other component to avoid manual handling
    """

    core          = None
    cubemaps      = {}
    textures      = {}
    models        = {}
    imgui_fonts   = {}
    imgui_font_configs = {}

    @classmethod
    def init(cls, core):
        cls.core = core
        # The font to default to if no imgui fonts have been loaded
        cls.imgui_fonts["--default--"] = cls.core.imgui_io.fonts.add_font_default()

    @classmethod
    def load_cubemap(cls, label: str, right: str, left: str, up: str, down: str, front: str, back: str):
        """
        Load up a cubemap
        label: identifier for the cubemap
        right, left, up, down, front, back are the names of respective \
        images in CWD/data/textures
        """
        faces  = [right, left, up, down, back, front]
        images = [Image.open(TEX_DIR / f) for f in faces]
        images[2] = images[2].rotate(90)
        data = b"".join(img.tobytes() for img in images)
        cls.cubemaps[label] = cls.core.ctx.texture_cube(
            size=images[0].size, components=3, data=data
        )

    @classmethod
    def get_cubemap(cls, label: str):
        """
        Returns a moderngl.TextureCube object associated with label
        """
        return cls.cubemaps[label]

    @classmethod
    def remove_cubemap(cls, label: str):
        """
        Remove the cubemap associated with label
        """
        cls.cubemaps[label].release()
        del cls.cubemaps[label]

    @classmethod
    def load_tex(cls, label: str, filename: str):
        """
        Load up a texture
        label: identifier for texture
        filename: name of image in CWD/engine/textures
        """
        if label not in cls.textures:
            img = Image.open(TEX_DIR / filename).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
            cls.textures[label] = cls.core.ctx.texture(size=img.size, components=3, data=img.tobytes())
            cls.textures[label].build_mipmaps()

    @classmethod
    def load_bin_tex(cls, label: str, size: tuple, bytes: str):
        if label not in cls.textures:
            cls.textures[label] = cls.core.ctx.texture(size=size, components=4, data=bytes)
            cls.textures[label].build_mipmaps()

    @classmethod
    def get_tex(cls, label: str):
        """
        Returns a moderngl.Texture object associated with label
        """
        return cls.textures[label]

    @classmethod
    def remove_tex(cls, label: str):
        """
        Remove the cubemap associated with label
        """
        cls.textures[label].release()
        del cls.textures[label]

    @classmethod
    def use_tex(cls, label: str):
        """
        Use the texture identified by label for use in shaders
        """
        # Unit 0 is reserved for ImGui; use unit 1.
        cls.textures[label].use(location=1)

    @classmethod
    def load_model(cls, label: str, filename: str):
        cls.models[label] = GLBModel(label, MODEL_DIR.joinpath(filename))

    @classmethod
    def remove_model(cls, label: str):
        """
        Removes the model and all textures prefixed with the model's label
        """
        if label in cls.models:
            prefix = f"{label}#"
            targets = [t for t in cls.textures if t.startswith(prefix)]
            for t in targets:
                cls.remove_tex(t)
            del cls.models[label]
    
    @classmethod
    def get_mesh_material(cls, final_label: str):
        """
        final_label = "model_name#mesh_name"
        """
        model_name, mesh_name = final_label.split('#')
        return cls.models[model_name].materials[mesh_name]

    @classmethod
    def get_model(cls, label: str):
        return cls.models[label]

    @classmethod
    def add_imgui_font(cls, label: str, filename: str, size: int):
        cfg = imgui.FontConfig()
        cfg.name         = label
        cfg.oversample_h = 2
        cfg.oversample_v = 2
        cls.imgui_font_configs[label] = cfg
        cls.imgui_fonts[label] = cls.core.imgui_io.fonts.add_font_from_file_ttf(
            str(TEX_DIR / filename), size, cfg
        )

    @classmethod
    def remove_imgui_font(cls, label: str):
        cls.imgui_font_configs.pop(label, None)
        del cls.imgui_fonts[label]

    @classmethod
    def get_imgui_font(cls, label: str):
        return cls.imgui_fonts[label]