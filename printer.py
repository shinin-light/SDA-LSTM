class Printer:

    def __init__(self, filename=None, on_screen=True):
        self.filename = filename
        self.on_screen = on_screen
    
    def open(self, mode='a'):
        if (self.filename is not None):
            self.file_object = open(self.filename, mode)
    
    def close(self):
        if (self.filename is not None):
            self.file_object.close()

    def print(self, string_value):
        if(self.on_screen):
            print(string_value)
        if(self.filename is not None):
            self.file_object.write(string_value + '\n')
    