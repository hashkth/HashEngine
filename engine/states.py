
from .camera import *


class State:

    def __init__(self, id):
        self.id = id

    def enter(self):
        pass

    def exit(self):
        # Called on exiting / deleting the instance of this state
        pass

    def cleanup(self):
        # Called only on deleting the instance of this state
        pass

    def events(self):
        pass

    def process(self):
        pass

    def render(self):
        pass

    def render_ui(self):
        pass