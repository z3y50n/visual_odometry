import logging
import sdl2
import sdl2.ext

logger = logging.getLogger(__name__)


class Display:
    def __init__(self, W, H, position=(0, 0)):
        sdl2.ext.init()
        self.window = sdl2.ext.Window(
            "Calibration Challenge", size=(W, H), position=position
        )
        self.window.show()

    def draw(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        windowArray = sdl2.ext.pixels3d(self.window.get_surface())
        windowArray[:, :, 0:3] = img.swapaxes(0, 1)
        self.window.refresh()
