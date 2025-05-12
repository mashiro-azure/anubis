import ctypes
import queue
import threading
import time

import cv2
import imgui
import numpy

# import Jetson.GPIO as GPIO
import OpenGL.GL as gl
import sdl2 as sdl
import torch
from imgui.integrations.sdl2 import SDL2Renderer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS, WritePrecision
from PIL import Image
from ultralytics import YOLO

VideoDevice = 0
webcam_frame_width = 640
webcam_frame_height = 480
# GPIOLEDPin = 7

OPENVINO_MODEL_PATH = "./yolo11s_int8_openvino_model/"


class CameraThread:
    def __init__(self, src, width, height):
        self.src = src
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
        return self.image

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

    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_DOUBLEBUFFER, 1)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_ACCELERATED_VISUAL, 1)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_FLAGS, sdl.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MINOR_VERSION, 1)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_PROFILE_MASK, sdl.SDL_GL_CONTEXT_PROFILE_COMPATIBILITY)

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
    sdl.SDL_GL_SetSwapInterval(1)

    return window, gl_context


class YOLOPredict(threading.Thread):
    def __init__(self, model):
        super().__init__()
        self.name = "yolo inference"
        self.daemon = True
        self.model = model
        self.frame_queue = queue.Queue(maxsize=1)
        self.result = None
        self._stop_event = threading.Event()

    def update_frame(self, frame):
        # If the queue has an old frame, remove it.
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def run(self):
        while not self._stop_event.is_set():
            try:
                # Try to get the latest frame. Wait up to 1 second.
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            self.result = self.model.predict(source=frame, classes=[0])[0]

    def stop(self):
        self._stop_event.set()


# 3 FPS impact (TODO: switch to threading?)
# def loggingToInfluxDB(noMaskCount):
#     bucket = "maskAI"
#     with InfluxDBClient.from_config_file("influxdb.ini") as client:
#         try:
#             DBHealth = client.health()
#             if DBHealth.status == "pass":
#                 p = Point("no_mask").field("amount", noMaskCount)
#                 client.write_api(write_options=SYNCHRONOUS).write(
#                     bucket=bucket, record=p, write_precision=WritePrecision.S
#                 )
#         except InfluxDBError as e:
#             pass
#     return str(DBHealth.message)


def showSplash(SDLwindow):
    splashImage = Image.open("SplashScreen-2.png").convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    splashImageData = numpy.array(splashImage, numpy.uint8)

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

    # gl.glColor3f(1.0, 1.0, 1.0)  # reset texture color, as GL_TEXTURE_ENV_MODE = GL_MODULATE, refer to glTexEnv
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

    # Setup GPIO
    # GPIO.setmode(GPIO.BOARD)
    # GPIO.setup(GPIOLEDPin, GPIO.OUT, initial=GPIO.LOW)

    # Setup Image Capture
    video = CameraThread(
        src=VideoDevice,
        width=webcam_frame_width,
        height=webcam_frame_height,
    )
    video.start()
    frame_height = webcam_frame_height
    frame_width = webcam_frame_width

    # YOLO
    model = YOLO(OPENVINO_MODEL_PATH, task="detect")
    yolo_prediction = YOLOPredict(model)
    yolo_prediction.start()

    # Setup logging
    timeRetain = ""
    # DBhealth = loggingToInfluxDB(0)
    DBhealth = ""
    timeEpoch = time.time()

    # Setup imgui
    imgui.create_context()  # type: ignore
    impl = SDL2Renderer(SDLwindow)
    sdlEvent = sdl.SDL_Event()

    io = imgui.get_io()  # type: ignore
    clearColorRGB = 1.0, 1.0, 1.0
    # newFont = io.fonts.add_font_from_file_ttf("fonts/NotoSansMono-Regular.ttf", 36)
    impl.refresh_font_texture()

    # States and variables
    running = True
    showCustomWindow = True
    cBoxBoxClass = True
    boxThreshold = 0.4
    showloggingWindow = True
    cBoxLogToInfluxDB = False
    maxHeadCount = 0
    maxBoxCount = 0
    showImageTexture = True

    while running:
        # read frame
        image = video.read()
        # print(output)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_prediction.update_frame(img_rgb)
        output = yolo_prediction.result

        # TODO: move this to the imageProcessing thread.
        # print custom bounding box
        if (output is not None) and (output.boxes.xyxy.size()[0] != 0):  # how many object detected
            for box in output.boxes:
                xmin, ymin, xmax, ymax, conf, _ = box.data.tolist()[0]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                # boxes
                if cBoxBoxClass is True and conf > boxThreshold:
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (97, 105, 255), 6)
                    image = cv2.rectangle(image, (xmin, ymin), (xmin + 150, ymin - 30), (97, 105, 255), -1)
                    image = cv2.putText(
                        image,
                        f"Person {conf:.2f}",
                        (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
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
            preprocess_time, inference_time, post_time = 0, 1, 0
            if output is not None:
                preprocess_time, inference_time, post_time = (
                    output.speed["preprocess"],
                    output.speed["inference"],
                    output.speed["postprocess"],
                )
            expandCustomWindow, showCustomWindow = imgui.begin("sdlWindow", True)
            imgui.text(f"FPS: {io.framerate:.2f}")
            _, clearColorRGB = imgui.color_edit3("Background Color", *clearColorRGB)
            imgui.new_line()
            imgui.text(f"Total Threads: {threading.active_count()}")
            imgui.new_line()
            imgui.text(
                f"Pre: {preprocess_time:.2f}ms Inf: {inference_time:.2f}ms Post: {post_time:.2f}ms       FPS: {1000/inference_time:.2f}"
            )
            _, cBoxLogToInfluxDB = imgui.checkbox("Log to InfluxDB (experimental feature)", cBoxLogToInfluxDB)
            imgui.new_line()
            imgui.text("Settings:")
            _, cBoxBoxClass = imgui.checkbox("Person", cBoxBoxClass)
            _, boxThreshold = imgui.slider_float(
                "Person Threshold",
                boxThreshold,
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
            imgui.end()

        if showImageTexture:
            expandImageTexture, showImageTexture = imgui.begin("ImageTexture", False)
            imgui.image(video.texture, frame_width, frame_height)
            imgui.end()

        # Logging stuff
        timeNow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Separate box and head count. 0 = box, 1 = mask, 2 = wo_mask, 3 = wrong_mask
        # There should be a better way of doing this...
        if output is not None:
            output_df = output.to_df()
            boxCount = output_df.count(0)

        if showloggingWindow:
            expandloggingWindow, showloggingWindow = imgui.begin("logging", True)
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
