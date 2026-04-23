from pynput.keyboard import Key, Controller
import time

keyboard = Controller()


class SendCommand(object):
    def __init__(self):
        pass

    def sand_command(self, param1, param2):
        if param1 is not None:
            if param1 > 0:
                for i in range(15):
                    keyboard.press(Key.left)
                    time.sleep(0.01)
                keyboard.release(Key.left)

            elif param1 < 0:
                for i in range(15):
                    keyboard.press(Key.right)
                    time.sleep(0.01)
                keyboard.release(Key.right)
