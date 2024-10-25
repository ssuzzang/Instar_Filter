from tkinter import *
from PIL import Image, ImageTk
import torch
import hlab_fast_neural_style


def cartoonify(img, pretrained, size):
    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained
    )
    face2paint = torch.hub.load(
        "bryandlee/animegan2-pytorch:main", "face2paint", size=size
    )
    return face2paint(model, img)


def read_square_image(filename, size):
    """이미지를 읽어서 가운데를 1:1 aspect ratio로 잘라내고 크기 변경"""

    img = Image.open(filename).convert("RGB")

    w, h = img.size
    if w > h:
        d = (w - h) // 2
        img = img.crop((d, 0, h + d, h))
    else:
        d = (h - w) // 2
        img = img.crop((0, d, w, w + d))

    return img.resize((size, size), Image.Resampling.LANCZOS)


# 주의: ImageTk 생성은 window = Tk() 이후에 호출
def tk_images(img):
    preview = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img), ImageTk.PhotoImage(preview)


def normal_click():
    # display["image"] = normal_img
    display.create_image(0, 0, anchor=NW, image=normal_tk)


def paprika_click():
    # display["image"] = paprika_img
    display.create_image(0, 0, anchor=NW, image=paprika_tk)


def mosaic_click():
    # display["image"] = mosaic_img
    display.create_image(0, 0, anchor=NW, image=mosaic_tk)


def candy_click():
    # display["image"] = candy_img
    display.create_image(0, 0, anchor=NW, image=candy_tk)


DISPLAY_SIZE = 640
PREVIEW_SIZE = 154

window = Tk()

img = read_square_image("jinsu.png", DISPLAY_SIZE)
normal_tk, normal_preview = tk_images(img)

paprika_img = cartoonify(img, "paprika", DISPLAY_SIZE)
paprika_tk, paprika_preview = tk_images(paprika_img)

mosaic_img = hlab_fast_neural_style.stylize(img, "saved_models/mosaic.pth")
mosaic_tk, mosaic_preview = tk_images(mosaic_img)

candy_img = hlab_fast_neural_style.stylize(img, "saved_models/candy.pth")
candy_tk, candy_preview = tk_images(candy_img)

# display = Label(window, image=normal_img)
display = Canvas(window, width=DISPLAY_SIZE, height=DISPLAY_SIZE, bg="white")
display.create_image(0, 0, anchor=NW, image=normal_tk)


frame_previews = Frame(window)

frame_normal = Frame(frame_previews)
frame_paprika = Frame(frame_previews)
frame_mosaic = Frame(frame_previews)
frame_candy = Frame(frame_previews)

label_normal = Label(frame_normal, text="Normal", bg="white")
label_paprika = Label(frame_normal, text="paprika", bg="white")
label_mosaic = Label(frame_normal, text="Mosaic", bg="white")
label_candy = Label(frame_normal, text="Candy", bg="white")

button_normal = Button(
    frame_normal, text="Normal", command=normal_click, image=normal_preview
)

button_paprika = Button(
    frame_paprika, text="Paprika", command=paprika_click, image=paprika_preview
)

button_mosaic = Button(
    frame_mosaic, text="Mosaic", command=mosaic_click, image=mosaic_preview
)

button_candy = Button(
    frame_candy, text="Candy", command=candy_click, image=candy_preview
)


label_normal.pack(side=TOP, fill=BOTH, expand=False)
button_normal.pack(side=TOP, fill=BOTH, expand=False)

label_paprika.pack(side=TOP, fill=BOTH, expand=False)
button_paprika.pack(side=TOP, fill=BOTH, expand=False)

label_mosaic.pack(side=TOP, fill=BOTH, expand=False)
button_mosaic.pack(side=TOP, fill=BOTH, expand=False)

label_candy.pack(side=TOP, fill=BOTH, expand=False)
button_candy.pack(side=TOP, fill=BOTH, expand=False)

frame_normal.pack(side=LEFT, fill=BOTH, expand=False)
frame_paprika.pack(side=LEFT, fill=BOTH, expand=False)
frame_mosaic.pack(side=LEFT, fill=BOTH, expand=False)
frame_candy.pack(side=LEFT, fill=BOTH, expand=False)

display.pack(side=TOP, fill=BOTH, expand=True)
frame_previews.pack(side=TOP, fill=BOTH, expand=True)

window.mainloop()
