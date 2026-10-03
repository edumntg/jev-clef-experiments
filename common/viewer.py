"""Live window for a stream of PIL frames. tkinter is stdlib, so no extra dependency.
Closing the window just stops updates; the run keeps going."""
import tkinter as tk

from PIL import Image, ImageTk


class Viewer:
    def __init__(self, title, scale=2):
        self.scale, self.open = scale, True
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.caption = tk.Label(self.root, font=("Menlo", 11), anchor="w", justify="left")
        self.caption.pack(fill="x")

    def show(self, img, text=""):
        if not self.open:
            return
        if self.scale != 1:
            img = img.resize((img.width * self.scale, img.height * self.scale), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)   # keep a ref or Tk drops it
        self.label.configure(image=self.photo)
        self.caption.configure(text=text)
        self.root.update()

    def close(self):
        if self.open:
            self.open = False
            self.root.destroy()
