# HashEngine - Script-Based Python Application Creation Engine
# Copyright (c) 2026 hashkth

# System libraries
import os
import sys
import math
import time
import random
import pathlib
import logging
import warnings
import base64
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass

# Hidden tk instance, tk is used for file dialogs
root = tk.Tk()
root.withdraw()

# Setup paths
CWD = pathlib.Path().cwd()
EXPORT_DIR = CWD.joinpath("exports")
DATA_DIR = CWD.joinpath("data")
SHADER_DIR = CWD.joinpath("shaders")
ENGINE_SHADER_DIR = CWD.joinpath("engine/shaders")
TEX_DIR = DATA_DIR.joinpath("textures")
AUDIO_DIR = DATA_DIR.joinpath("audio")
MODEL_DIR = DATA_DIR.joinpath("models")

# Disable hrtf to prevent muddy filter
os.environ["ALSOFT_CONF"] = str(CWD.joinpath("alsoft.ini"))

# Other libraries
import glux
from glux import imgui
import pyopenalsoft as al
import moderngl as mgl
import numpy as np
from pyglm import glm
from PIL import Image
from pygltflib import GLTF2

# To silence warning: Unimplemented OBJ format statement 's' on line 's off'
logging.getLogger('pywavefront').setLevel(logging.ERROR)

# To prevent transparency warning for RGBA images
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Palette images with Transparency expressed in bytes*"
)

# Shader loading utility --> to be moved into a utils module
def load_shader(filename: str):
    """Loads shader code from CWD/shaders"""
    data = None
    with open(SHADER_DIR.joinpath(filename), 'r') as f:
        data = f.read()
    return data

def load_engine_shader(filename: str):
    """Loads shader code from CWD/engine/shaders"""
    data = None
    with open(ENGINE_SHADER_DIR.joinpath(filename), 'r') as f:
        data = f.read()
    return data

@dataclass
class Timer:
    """
    Signals 'interval' seconds after 'time' based on 'looping' and 'active' \n
    Looping timers: Send single rings after each interval if active \n
    Non-looping timers: Continuously ring after interval since start until stopped
    """
    time: float
    interval: float
    looping: bool
    active: bool
    ringed: bool
    pause_buffer: float
    paused: bool

class Clock:

    """
    The global clock for all time-based calculations, timers etc.
    """
    _init_time = time.perf_counter()
    _last_time = 0

    core = None
    time = 0
    dt = 0.016
    dt_sq = dt * dt

    # Holds looping, non-looping timers
    timers = {}

    @classmethod
    def init(cls, core: any):
        cls.core = core

    @classmethod
    def create_timer(cls, label: str, interval: float, looping: bool):
        """
        label: accessor for timer object\n
        interval: loop interval if looping\n
        looping: to loop or not\n
        Countable loops NOT implemented
        """
        cls.timers[label] = Timer(cls.time, interval, looping, False, False, 0.0, False)

    @classmethod
    def remove_timer(cls, label: str):
        """
        Remove timer identified by given label
        """
        cls.timers.pop(label, None)

    @classmethod
    def start_timer(cls, label: str):
        """
        Start timer identified by given label
        """
        if cls.timers[label].paused:
            # See pause_timer() for explanation
            cls.timers[label].time = cls.time - cls.timers[label].pause_buffer
            cls.timers[label].paused = False
        else:
            cls.timers[label].time = cls.time
        cls.timers[label].active = True

    @classmethod
    def set_interval(cls, label: str, interval: float):
        cls.timers[label].interval = interval

    @classmethod
    def pause_timer(cls, label: str):
        """
        Pause timer identified by given label
        """
        # Pause buffer stores time elapsed since last ring
        # We will subtract this from cls.time while resuming so that 
        # interval - already_elapsed time will be left until the next ring
        cls.timers[label].pause_buffer = (cls.time - cls.timers[label].time)
        cls.timers[label].active = False
        cls.timers[label].paused = True

    @classmethod
    def stop_timer(cls, label: str):
        """
        Stop timer identified by given label
        """
        cls.timers[label].active = False

    @classmethod
    def ringed(cls, label: str):
        """
        Check if the timer identified by label ringed
        """
        return cls.timers[label].ringed

    @classmethod
    def process(cls):
        cls.time = time.perf_counter() - cls._init_time
        cls.dt = cls.time - cls._last_time
        cls.dt_sq = cls.dt * cls.dt
        cls._last_time = cls.time

        for timer in cls.timers.values():
            timer.ringed = False
            if timer.active:
                if Clock.time - timer.time > timer.interval:
                    if timer.looping: timer.time = Clock.time
                    timer.ringed = True