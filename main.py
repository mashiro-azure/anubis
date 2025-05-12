import ctypes
import threading
import time
import re

from PIL import Image
import numpy

import cv2
import imgui

# import Jetson.GPIO as GPIO
import OpenGL.GL as gl
import sdl2 as sdl
import torch
from imgui.integrations.sdl2 import SDL2Renderer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, WritePrecision
from influxdb_client.client.exceptions import InfluxDBError

VideoDevice = 1
webcam_frame_width = 2560
webcam_frame_height = 720
# GPIOLEDPin = 7


class CameraThread:
    def __init__(self, src, width, height, model):
        self.src = src
        self.model = model
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.texture = gl.glGenTextures(1)
        if not (self.cap.isOpened()):
            print("VideoCapture error.")
            return

        self.ret, self.image = self.cap.read()
        if not (self.ret):
            print("No more frames.")
            return
        self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.output = self.model(self.img_rgb)

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.started = False

    def start(self):
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            self.ret, self.image = self.cap.read()

    def read(self):
        return self.image, self.output

    def bind(self, image):
        # opengl prepare textures
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            webcam_frame_width,
            webcam_frame_height,
            0,
            gl.GL_BGR,
            gl.GL_UNSIGNED_BYTE,
            image,
        )

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()


def impl_pysdl2_init():
    width, height = 1920, 1080
    window_name = "minimal ImGui/SDL2 example"

    if sdl.SDL_Init(sdl.SDL_INIT_VIDEO) < 0:
        print("Error: SDL could not initialize! SDL Error: " + sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_DOUBLEBUFFER, 1)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_DEPTH_SIZE, 24)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_STENCIL_SIZE, 8)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_ACCELERATED_VISUAL, 1)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_MULTISAMPLEBUFFERS, 1)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_MULTISAMPLESAMPLES, 16)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_FLAGS, sdl.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MINOR_VERSION, 6)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_PROFILE_MASK, sdl.SDL_GL_CONTEXT_PROFILE_COMPATIBILITY)

    # sdl.SDL_SetHint(sdl.SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, b"1")
    # sdl.SDL_SetHint(sdl.SDL_HINT_VIDEO_HIGHDPI_DISABLED, b"1")

    window = sdl.SDL_CreateWindow(
        window_name.encode("utf-8"),
        sdl.SDL_WINDOWPOS_CENTERED,
        sdl.SDL_WINDOWPOS_CENTERED,
        width,
        height,
        sdl.SDL_WINDOW_OPENGL | sdl.SDL_WINDOW_RESIZABLE,
    )

    if window is None:
        print("Error: Window could not be created! SDL Error: " + sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    gl_context = sdl.SDL_GL_CreateContext(window)
    if gl_context is None:
        print("Error: Cannot create OpenGL Context! SDL Error: " + sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    sdl.SDL_GL_MakeCurrent(window, gl_context)
    if sdl.SDL_GL_SetSwapInterval(1) < 0:
        print("Warning: Unable to set VSync! SDL Error: " + sdl.SDL_GetError().decode("utf-8"))
        exit(1)
    sdl.SDL_GL_SetSwapInterval(0)

    return window, gl_context


# 3 FPS impact (TODO: switch to threading?)
def loggingToInfluxDB(noMaskCount):
    bucket = "maskAI"
    with InfluxDBClient.from_config_file("influxdb.ini") as client:
        try:
            DBHealth = client.health()
            if DBHealth.status == "pass":
                p = Point("no_mask").field("amount", noMaskCount)
                client.write_api(write_options=SYNCHRONOUS).write(
                    bucket=bucket, record=p, write_precision=WritePrecision.S
                )
        except InfluxDBError as e:
            pass
    return str(DBHealth.message)


def showSplash(SDLwindow):
    splashImage = Image.open("SplashScreen-2.png").convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    splashImageData = numpy.array(list(splashImage.getdata()), numpy.uint8)

    splashTexture = gl.glGenTextures(1)  # it doesn't want to bind to array texture, so separate textures creation.
    gl.glBindTexture(gl.GL_TEXTURE_2D, splashTexture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        splashImage.width,
        splashImage.height,
        0,
        gl.GL_RGB,
        gl.GL_UNSIGNED_BYTE,
        splashImageData,
    )

    splashImage.close()

    gl.glColor3f(1.0, 1.0, 1.0)  # reset texture color, as GL_TEXTURE_ENV_MODE = GL_MODULATE, refer to glTexEnv
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2f(0, 0)
    gl.glVertex2f(-1, -1)
    gl.glTexCoord2f(1, 0)
    gl.glVertex2f(1, -1)
    gl.glTexCoord2f(1, 1)
    gl.glVertex2f(1, 1)
    gl.glTexCoord2f(0, 1)
    gl.glVertex2f(-1, 1)
    gl.glEnd()
    gl.glDisable(gl.GL_TEXTURE_2D)

    sdl.SDL_GL_SwapWindow(SDLwindow)


def main():
    SDLwindow, glContext = impl_pysdl2_init()
    showSplash(SDLwindow)
    # Setup YOLOv5
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(
        "yolov5",
        "custom",
        path="models/20230122-mixedDataset-300epoch.pt",
        source="local",
    )
    #   Path to yolov5, 'custom', path to weight, source='local'
    model.to(devices)
    model.eval()

    # Setup GPIO
    # GPIO.setmode(GPIO.BOARD)
    # GPIO.setup(GPIOLEDPin, GPIO.OUT, initial=GPIO.LOW)

    # Setup Image Capture
    video = CameraThread(
        src=VideoDevice,
        width=webcam_frame_width,
        height=webcam_frame_height,
        model=model,
    )
    video.start()
    frame_height = webcam_frame_height
    frame_width = webcam_frame_width

    # Setup logging
    timeRetain = ""
    DBhealth = loggingToInfluxDB(0)
    timeEpoch = time.time()

    # Setup imgui
    imgui.create_context()  # type: ignore
    impl = SDL2Renderer(SDLwindow)
    sdlEvent = sdl.SDL_Event()

    io = imgui.get_io()  # type: ignore
    clearColorRGB = 1.0, 1.0, 1.0
    newFont = io.fonts.add_font_from_file_ttf("fonts/NotoSansMono-Regular.ttf", 36)
    impl.refresh_font_texture()

    # States and variables
    running = True
    showCustomWindow = True
    cBoxBoxClass = True
    boxThreshold = 0.4
    cBoxWithMaskClass = True
    maskThreshold = 0.4
    cBoxWithoutMaskClass = True
    cBoxWrongMaskClass = True
    showloggingWindow = True
    cBoxLogToInfluxDB = False
    maxHeadCount = 0
    maxBoxCount = 0
    showImageTexture = True

    while running:
        # read frame
        image, output = video.read()
        # print(output)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = model(img_rgb)

        # TODO: move this to the imageProcessing thread.
        # print custom bounding box
        for box in output.xyxy[0]:
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # boxes
            if cBoxBoxClass == True and box[5] == 0 and box[4] > boxThreshold:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (97, 105, 255), 6)
                image = cv2.rectangle(image, (xmin, ymin), (xmin + 150, ymin - 30), (97, 105, 255), -1)
                image = cv2.putText(
                    image,
                    f"Box {box[4]:.2f}",
                    (int(xmin), int(ymin) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            # With Masks
            if cBoxWithMaskClass == True and box[5] == 1 and box[4] > maskThreshold:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (119, 221, 119), 6)
                image = cv2.rectangle(image, (xmin, ymin), (xmin + 250, ymin - 30), (119, 221, 119), -1)
                image = cv2.putText(
                    image,
                    f"With Mask {box[4]:.2f}",
                    (int(xmin), int(ymin) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            # Without Masks
            if cBoxWithoutMaskClass == True and box[5] == 2:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (97, 105, 255), 6)
                image = cv2.rectangle(image, (xmin, ymin), (xmin + 210, ymin - 30), (97, 105, 255), -1)
                image = cv2.putText(
                    image,
                    "Without Mask",
                    (int(xmin), int(ymin) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Masks Worn Incorrectly
            if cBoxWrongMaskClass == True and box[5] == 3:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (152, 200, 250), 2)

        video.bind(image=image)

        # SDL & imgui event polling
        while sdl.SDL_PollEvent(ctypes.byref(sdlEvent)) != 0:
            if sdlEvent.type == sdl.SDL_QUIT:
                running = False
                break
            impl.process_event(sdlEvent)
        impl.process_inputs()

        imgui.new_frame()  # type: ignore

        if showCustomWindow:
            preprocess_time, inference_time, NMS_time = re.findall(r"[\d\.\d]+(?=ms)", str(output))
            expandCustomWindow, showCustomWindow = imgui.begin("sdlWindow", True)
            imgui.text(f"FPS: {io.framerate:.2f}")
            _, clearColorRGB = imgui.color_edit3("Background Color", *clearColorRGB)
            imgui.new_line()
            imgui.text(f"Total Threads: {threading.active_count()}")
            imgui.new_line()
            imgui.text(f"Pre: {preprocess_time}ms Inf: {inference_time}ms NMS: {NMS_time}ms")
            _, cBoxLogToInfluxDB = imgui.checkbox("Log to InfluxDB (experimental feature)", cBoxLogToInfluxDB)
            imgui.new_line()
            imgui.text("Settings:")
            _, cBoxBoxClass = imgui.checkbox("Box", cBoxBoxClass)
            _, boxThreshold = imgui.slider_float(
                "Box Threshold",
                boxThreshold,
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
            _, cBoxWithMaskClass = imgui.checkbox("With Mask", cBoxWithMaskClass)
            _, maskThreshold = imgui.slider_float(
                "Mask Threshold",
                maskThreshold,
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
            _, cBoxWithoutMaskClass = imgui.checkbox("Without Mask", cBoxWithoutMaskClass)
            _, cBoxWrongMaskClass = imgui.checkbox("Masks Worn Incorrectly.", cBoxWrongMaskClass)
            imgui.end()

        if showImageTexture:
            expandImageTexture, showImageTexture = imgui.begin("ImageTexture", False)
            imgui.image(video.texture, frame_width, frame_height)
            imgui.end()

        # Logging stuff
        timeNow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Separate box and head count. 0 = box, 1 = mask, 2 = wo_mask, 3 = wrong_mask
        # There should be a better way of doing this...
        output_df = output.pandas().xyxy[0]["class"].tolist()
        boxCount = output_df.count(0)
        withMaskCount = output_df.count(1)
        noMaskCount = output_df.count(2)
        wrongMaskCount = output_df.count(3)

        nowHeadCount = withMaskCount + noMaskCount + wrongMaskCount

        if showloggingWindow:
            expandloggingWindow, showloggingWindow = imgui.begin("logging", True)
            if noMaskCount > 0:
                timeRetain = timeNow
            if nowHeadCount > maxHeadCount:
                maxHeadCount = nowHeadCount
            if boxCount > maxBoxCount:
                maxBoxCount = boxCount
            if noMaskCount >= 1 and cBoxLogToInfluxDB == True and time.time() > timeEpoch + 1:
                DBhealth = loggingToInfluxDB(noMaskCount)
            with imgui.font(newFont):
                imgui.text(f"Person in view: {nowHeadCount}\tRecorded max person in view: {maxHeadCount}")
                imgui.text(f"Box in view: {boxCount}\t\tRecorded max boxes in view: {maxBoxCount}")
                imgui.new_line()
                imgui.text(f"{timeRetain}: Person with No Mask detected.")
                imgui.new_line()
                imgui.text_wrapped(f"InfluxDB Health: {DBhealth}")
            imgui.end()

        gl.glClearColor(clearColorRGB[0], clearColorRGB[1], clearColorRGB[2], 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        sdl.SDL_GL_SwapWindow(SDLwindow)

    video.stop()
    # loggingThread.join()
    impl.shutdown()
    sdl.SDL_GL_DeleteContext(glContext)
    sdl.SDL_DestroyWindow(SDLwindow)
    sdl.SDL_Quit()


if __name__ == "__main__":
    main()
