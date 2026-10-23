
### IMPORTS
import tkinter as tk
import torch as tor
import PIL as pil


#print("DEBUG: Imports completed")

### WINDOW PROPERTIES

mainWindow = tk.Tk()
mainWindow.title("shape-finder")
mainWindow.minsize(512, 540)
mainWindow.maxsize(512, 540)

icon = tk.PhotoImage(file="icon.png")
mainWindow.iconphoto(True, icon)
#print("DEBUG: Window properties set")

tk.Label(mainWindow, text="draw a shape!").pack()
tk.Label(mainWindow, text="predictions: circle-X% , square-X% , triangle-X%").pack()

drawingBox = tk.Canvas(mainWindow, width=440, height=440, bg="gray")
drawingBox.pack()

mainWindow.mainloop()
#print("DEBUG: Main window initialized")
