import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb, filedialog
import os
from PIL import Image, ImageTk
import magic



def CreateWidgets():
    image = Image.open("yolo.jpg")
    photo = ImageTk.PhotoImage(image)

    label = Label(root, image=photo)
    label.image = photo
    label.grid(row=3, column=3, pady=15, padx=15)




    destinationLabel = Label(root, text=" Select Image or Video    :", bg="Yellow")
    destinationLabel.grid(row=15, column=2, pady=15, padx=15)

    root.destinationText = Entry(root, width=50, textvariable=downloadPath)
    root.destinationText.grid(row=15, column=3, pady=5, padx=5)

    browseButton = Button(root, text="BROWSE", command=Browse, width=15,bg='orange')
    browseButton.grid(row=15, column=4, pady=5, padx=5)

    dwldButton = Button(root, text="Detect", command=detect, width=15, bg='#00cc00',borderwidth = 0)
    dwldButton.grid(row=16, column=3, pady=5, padx=5)
    dwldButton2 = Button(root, text="Realtime", command=realtime, width=15, bg='#ff4d4d', borderwidth=0)
    dwldButton2.grid(row=17, column=3, pady=5, padx=5)

def detect():
    print(ddt)
    if ddt=='':
        mb.showwarning('Warning', 'No File Found')
    inl=magic.from_file(ddt, mime=True)
    print(inl)
    if str(inl) =="video/mp4":
        os.system("python yolo.py --video-path="+ddt)
        os.startfile("output.avi")
    if "image/" in str(inl):
        os.system("python yolo.py --image-path="+ddt)
    else:
        mb.showwarning('Warning',"Please Check the Format of the File ")
def realtime():
    os.system('python yolo.py')

def Browse():
    dwldDirectory = filedialog.askopenfilename(initialdir="")
    downloadPath.set(dwldDirectory)
    global ddt
    ddt=dwldDirectory
    print(dwldDirectory)


root = tk.Tk()


root.geometry("650x400")
root.resizable(False, False)
root.title("YOLOv3-Object-Detection-with-OpenCV")
root.config(background="white")


videoLink = StringVar()
downloadPath = StringVar()
ddt=""
CreateWidgets()
root.mainloop()
