import ctypes
import queue
import threading
import time

import cv2
import imgui
import numpy

# import Jetson.GPIO as GPIO
import OpenGL.GL as gl
import paho.mqtt.client as mqtt
import sdl2 as sdl
import torch
from imgui.integrations.sdl2 import SDL2Renderer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS, WritePrecision
from PIL import Image
from ultralytics import YOLO

# import RPi.GPIO as GPIO

VideoDevice = 2
webcam_frame_width = 640
webcam_frame_height = 480
# GPIOLEDPin = 7
# GPIO.setmode(GPIO.BCM)
fan_pin = 18
# GPIO.setup(fan_pin, GPIO.OUT)

OPENVINO_MODEL_PATH = "./yolo11s_int8_openvino_model/"

mqtt_data = {"anubis/data": (0.0, 0), "anubis/audio_score": 0.0}


class CameraThread:
    def __init__(self, src, width, height):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "P", "E", "G"))
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
    width, height = 1280, 720
    window_name = "minimal ImGui/SDL2 example"

    if sdl.SDL_Init(sdl.SDL_INIT_VIDEO) < 0:
        print("Error: SDL could not initialize! SDL Error: " + sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_DOUBLEBUFFER, 1)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_ACCELERATED_VISUAL, 1)
    # sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_FLAGS, sdl.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MAJOR_VERSION, 1)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MINOR_VERSION, 2)
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
def loggingToInfluxDB(triggerActive):
    bucket = "anubis"
    with InfluxDBClient.from_config_file("influxdb.ini") as client:
        try:
            DBHealth = client.health()
            if DBHealth.status == "pass":
                p = Point("person").field("active", int(triggerActive))
                client.write_api(write_options=SYNCHRONOUS).write(
                    bucket=bucket, record=p, write_precision=WritePrecision.S
                )
        except InfluxDBError as e:
            pass
    return str(DBHealth.message)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe("anubis/data")
    client.subscribe("anubis/audio_score")


def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
    payload = msg.payload.decode()
    match msg.topic:
        case "anubis/data":
            parts = payload.split(",")
            if len(parts) == 2:
                temp = float(parts[0])
                pressure = int(parts[1])
                mqtt_data["anubis/data"] = (temp, pressure)

        case "anubis/audio_score":
            audio_score = float(payload)
            mqtt_data["anubis/audio_score"] = audio_score


def triggerActivation(client: mqtt.Client, fanSpeed: int):
    client.publish("anubis/fan_control", str(fanSpeed))
    return


def triggerDeactivation(client: mqtt.Client, fanSpeed: int):
    client.publish("anubis/fan_control", "0")
    return


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
    DBhealth = loggingToInfluxDB(0)
    # DBhealth = ""
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
    showloggingWindow = True
    cBoxLogToInfluxDB = True
    nowHeadCount = 0
    showImageTexture = True
    # TODO:
    triggerActive = False
    activationStartTime = None  # when to activate signal
    deactivationStartTime = None  # when to deactivate the signal
    activationThreshold = 3.0  # how long to wait before activating the signal
    deactivationThreshold = 3.0  # how long to wait before deactivating the signal
    triggerActiveColor = 1.0, 0.0, 0.0
    fanSpeed, temperaure, pressure, audio_score = 0, 0, 0, 0.0
    temperatureThreshold = 25
    pressureThreshold = 3000
    audioThreshold = 0.7

    # mqtt
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("localhost", 1883, keepalive=60)
    client.loop_start()

    while running:
        # read frame
        image = video.read()
        # print(output)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_prediction.update_frame(img_rgb)
        output = yolo_prediction.result

        # mqtt process
        if mqtt_data["anubis/data"] is not None and mqtt_data["anubis/audio_score"] is not None:
            temperaure, pressure = mqtt_data["anubis/data"]
            audio_score = mqtt_data["anubis/audio_score"]
            fanSpeed = min(100, max(0, (int(float(temperaure)) - 25) * 5))

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

        # effect activation
        current_time = time.time()
        if output is not None:
            output_df = output.to_df()
            nowHeadCount = output_df.shape[0]

        # activation rules
        rule_temperature = temperaure >= temperatureThreshold
        rule_pressure = pressure >= pressureThreshold
        rule_vision = nowHeadCount > 0
        rule_audio = audio_score >= audioThreshold
        if rule_temperature and rule_pressure and (rule_vision or rule_audio):
            if activationStartTime is None:
                activationStartTime = current_time
            deactivationStartTime = None

            if (current_time - activationStartTime) >= activationThreshold:
                if not triggerActive:
                    triggerActive = True
                    loggingToInfluxDB(triggerActive)
        else:
            if deactivationStartTime is None:
                deactivationStartTime = current_time
            activationStartTime = None

            if (current_time - deactivationStartTime) >= deactivationThreshold:
                if triggerActive:
                    triggerActive = False
                    loggingToInfluxDB(triggerActive)

        if triggerActive:
            triggerActiveColor = 0.0, 1.0, 0.0
            triggerActivation(client, fanSpeed)
        else:
            triggerActiveColor = 1.0, 0.0, 0.0
            triggerDeactivation(client, 0)

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

            imgui.new_line()
            imgui.text("Trigger: ")
            _, activationThreshold = imgui.slider_int(
                "Activation Threshold (s)", activationThreshold, min_value=0, max_value=10
            )
            _, deactivationThreshold = imgui.slider_int(
                "Deactivation Threshold (s)", deactivationThreshold, min_value=0, max_value=10
            )
            imgui.color_button(
                "triggerActive", *triggerActiveColor, imgui.COLOR_EDIT_NO_PICKER | imgui.COLOR_EDIT_NO_OPTIONS
            )
            imgui.new_line()
            imgui.text(
                f"Temperature: {temperaure} / Pressure: {pressure} / Audio: {audio_score} / Fan Speed: {fanSpeed}"
            )
            imgui.new_line()
            overrideOnClicked = imgui.button("on")
            overrideOffClicked = imgui.button("off")
            imgui.end()

            if overrideOnClicked:
                client.publish("anubis/fan_control", "100")
            if overrideOffClicked:
                client.publish("anubis/fan_control", "0")

        if showImageTexture:
            expandImageTexture, showImageTexture = imgui.begin("ImageTexture", False)
            imgui.image(video.texture, frame_width, frame_height)
            imgui.end()

        # Logging stuff
        timeNow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if showloggingWindow:
            expandloggingWindow, showloggingWindow = imgui.begin("logging", True)
            with imgui.font(newFont):
                imgui.text(f"Person in view: {nowHeadCount}")
                imgui.new_line()
                imgui.text_wrapped(f"InfluxDB Health: {DBhealth}")
            imgui.end()

        gl.glClearColor(clearColorRGB[0], clearColorRGB[1], clearColorRGB[2], 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        sdl.SDL_GL_SwapWindow(SDLwindow)

    client.loop_stop()
    client.disconnect()
    video.stop()
    # loggingThread.join()
    impl.shutdown()
    sdl.SDL_GL_DeleteContext(glContext)
    sdl.SDL_DestroyWindow(SDLwindow)
    sdl.SDL_Quit()


if __name__ == "__main__":
    main()
