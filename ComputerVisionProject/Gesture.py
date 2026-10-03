import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
import numpy as np
import speech_recognition as sr
 
# ── Setup ──
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()
 
cursor_history = []
HISTORY_SIZE = 5
px, py = 0, 0
 
last_click_time = 0
last_scroll_time = 0
last_screenshot_time = 0
last_ppt_time = 0
last_sign_time = 0
last_rclick_time = 0
CLICK_COOLDOWN = 0.4
SCROLL_COOLDOWN = 0.05
SCREENSHOT_COOLDOWN = 2.0
PPT_COOLDOWN = 0.6
SIGN_COOLDOWN = 2.0
RCLICK_COOLDOWN = 0.5
 
voice_label = ""
voice_label_time = 0
VOICE_LABEL_DURATION = 2.0
 
prev_hand_x = None
wave_directions = []
 
system_state = "locked"
password_hold_start = None
PASSWORD_HOLD_TIME = 2.0
spider_label = ""
spider_label_time = 0
mode_select_start = None
MODE_SELECT_HOLD = 1.5
 
typing_mode = False
 
# ── Sign typing mode ──
last_letter_time = 0
LETTER_COOLDOWN = 1.2
last_typed_letter = ""
sign_typed_text = ""
 
model = mp.tasks.vision.HandLandmarker
base_options = mp.tasks.BaseOptions(model_asset_path=r"C:\My code\hand_landmarker.task")
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = model.create_from_options(options)
 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
 
def get_distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)
 
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    up = []
    up.append(hand[4].x < hand[3].x)
    for i in range(1, 5):
        up.append(hand[tips[i]].y < hand[tips[i] - 2].y)
    return up
 
def get_thumb_direction(hand):
    thumb_tip_x = hand[4].x
    thumb_base_x = hand[2].x
    index_down  = hand[8].y  > hand[6].y
    middle_down = hand[12].y > hand[10].y
    ring_down   = hand[16].y > hand[14].y
    pinky_down  = hand[20].y > hand[18].y
    all_closed  = index_down and middle_down and ring_down and pinky_down
    if not all_closed:
        return None
    diff = thumb_tip_x - thumb_base_x
    if diff > 0.08:
        return "right"
    elif diff < -0.08:
        return "left"
    return None
 
def smooth_cursor(new_x, new_y):
    global cursor_history
    cursor_history.append((new_x, new_y))
    if len(cursor_history) > HISTORY_SIZE:
        cursor_history.pop(0)
    avg_x = int(np.mean([p[0] for p in cursor_history]))
    avg_y = int(np.mean([p[1] for p in cursor_history]))
    return avg_x, avg_y
 
def is_spider_sign(hand):
    up = fingers_up(hand)
    return (up[0] and up[1] and not up[2] and not up[3] and up[4])
 
def is_middle_ring_up(hand):
    up = fingers_up(hand)
    return (not up[1] and up[2] and up[3] and not up[4])
 
def detect_thankyou(hand):
    tips  = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    all_up = all(hand[tips[i]].y < hand[bases[i]].y for i in range(4))
    index_middle_dist = abs(hand[8].x - hand[12].x)
    middle_ring_dist  = abs(hand[12].x - hand[16].x)
    ring_pinky_dist   = abs(hand[16].x - hand[20].x)
    fingers_together  = (index_middle_dist < 0.06 and
                         middle_ring_dist  < 0.06 and
                         ring_pinky_dist   < 0.06)
    return all_up and fingers_together and hand[9].y < 0.55
 
def move_cursor(hand):
    up = fingers_up(hand)
    if up[1] and not up[2]:
        raw_x = int(hand[8].x * screen_w * 1.5 - screen_w * 0.25)
        raw_y = int(hand[8].y * screen_h * 1.5 - screen_h * 0.25)
        raw_x = max(0, min(screen_w - 1, raw_x))
        raw_y = max(0, min(screen_h - 1, raw_y))
        cx, cy = smooth_cursor(raw_x, raw_y)
        pyautogui.moveTo(cx, cy)
        return True
    else:
        cursor_history.clear()
        return False
 
def detect_asl_letter(hand):
    up = fingers_up(hand)  # [thumb, index, middle, ring, pinky]
    thumb, index, middle, ring, pinky = up
 
    thumb_index_dist = get_distance(hand[4], hand[8])
 
    # A — fist with thumb to the side
    if not index and not middle and not ring and not pinky and not thumb:
        return "A"
 
    # B — flat hand, four fingers up together, thumb tucked across palm
    if index and middle and ring and pinky and not thumb:
        return "B"
 
    # L — thumb + index up, forming an L shape, others down
    if thumb and index and not middle and not ring and not pinky:
        return "L"
 
    # Y — thumb + pinky out, others down (like "call me")
    if thumb and not index and not middle and not ring and pinky:
        return "Y"
 
    # I — only pinky up
    if not thumb and not index and not middle and not ring and pinky:
        return "I"
 
    # D — only index up, others down
    if not thumb and index and not middle and not ring and not pinky:
        return "D"
 
    # V or U — index + middle up
    if not thumb and index and middle and not ring and not pinky:
        idx_mid_dist = get_distance(hand[8], hand[12])
        if idx_mid_dist > 0.06:
            return "V"
        else:
            return "U"
 
    # W — index + middle + ring up
    if not thumb and index and middle and ring and not pinky:
        return "W"
 
    # F — index + thumb pinch, other 3 fingers up
    if middle and ring and pinky and thumb_index_dist < 0.05:
        return "F"
 
    # O — fingertips curled near thumb tip, forming a circle
    if (not index and not middle and not ring and not pinky and
            0.02 < thumb_index_dist < 0.07):
        return "O"
 
    return None
 
def handle_voice_command(command):
    global voice_label, voice_label_time, system_state, typing_mode
 
    if system_state != "voice":
        return
 
    command = command.lower().strip()
    print(f"Voice: {command}")
 
    import subprocess
 
    # ── TYPING MODE ──
    if typing_mode:
        if "stop typing" in command:
            typing_mode = False
            voice_label = "TYPING MODE OFF"
            voice_label_time = time.time()
            return
        elif "new line" in command:
            pyautogui.press('enter')
            voice_label = "NEW LINE"
        elif "delete word" in command:
            pyautogui.hotkey('ctrl', 'backspace')
            voice_label = "WORD DELETED"
        elif "delete" in command or "backspace" in command:
            pyautogui.press('backspace')
            voice_label = "DELETED"
        elif "comma" in command:
            pyautogui.write(",", interval=0.05)
            voice_label = "COMMA"
        elif "question mark" in command:
            pyautogui.write("?", interval=0.05)
            voice_label = "QUESTION MARK"
        elif "period" in command or "full stop" in command:
            pyautogui.write(".", interval=0.05)
            voice_label = "PERIOD"
        else:
            pyautogui.write(command + " ", interval=0.05)
            voice_label = f"TYPED: {command}"
        voice_label_time = time.time()
        return
 
    voice_label = f'"{command}"'
    voice_label_time = time.time()
 
    if "open chrome" in command or "open browser" in command:
        subprocess.Popen("start chrome", shell=True)
    elif "open notepad" in command:
        subprocess.Popen("notepad.exe")
    elif "open word" in command or "open microsoft word" in command:
        subprocess.Popen("start winword", shell=True)
    elif "open powerpoint" in command or "open ppt" in command:
        subprocess.Popen("start powerpnt", shell=True)
    elif "open excel" in command or "open microsoft excel" in command:
        subprocess.Popen("start excel", shell=True)
    elif "open settings" in command:
        subprocess.Popen("start ms-settings:", shell=True)
    elif "open file explorer" in command or "open explorer" in command:
        subprocess.Popen("explorer.exe")
    elif "open task manager" in command:
        subprocess.Popen("taskmgr.exe")
    elif "open sticky notes" in command:
        subprocess.Popen("StikyNot.exe")
    elif "open codes" in command or "open commands" in command:
        subprocess.Popen(['notepad.exe', r'C:\Users\j\Desktop\asep\voice_commands.txt'])
        voice_label = "OPENING COMMAND LIST"
        voice_label_time = time.time()
    elif "start presentation" in command or "start slideshow" in command:
        subprocess.Popen("start powerpnt", shell=True)
        time.sleep(3)
        pyautogui.press('f5')
        voice_label = "PRESENTATION STARTED!"
        voice_label_time = time.time()
    elif "stop presentation" in command or "end presentation" in command or "exit slideshow" in command:
        pyautogui.press('escape')
        voice_label = "PRESENTATION STOPPED!"
        voice_label_time = time.time()
    elif "next slide" in command or "slide forward" in command or "forward slide" in command:
        pyautogui.press('right')
        voice_label = "NEXT SLIDE"
        voice_label_time = time.time()
    elif "previous slide" in command or "prev slide" in command or "slide back" in command or "go back slide" in command:
        pyautogui.press('left')
        voice_label = "PREV SLIDE"
        voice_label_time = time.time()
    elif "first slide" in command or "go to first" in command:
        pyautogui.hotkey('ctrl', 'home')
        voice_label = "FIRST SLIDE"
        voice_label_time = time.time()
    elif "last slide" in command or "go to last" in command:
        pyautogui.hotkey('ctrl', 'end')
        voice_label = "LAST SLIDE"
        voice_label_time = time.time()
    elif "black screen" in command or "blank screen" in command:
        pyautogui.press('b')
        voice_label = "BLACK SCREEN"
        voice_label_time = time.time()
    elif "white screen" in command:
        pyautogui.press('w')
        voice_label = "WHITE SCREEN"
        voice_label_time = time.time()
    elif "duplicate slide" in command:
        pyautogui.hotkey('ctrl', 'd')
        voice_label = "SLIDE DUPLICATED"
        voice_label_time = time.time()
    elif "new slide" in command:
        pyautogui.hotkey('ctrl', 'm')
        voice_label = "NEW SLIDE"
        voice_label_time = time.time()
 
    elif "end page" in command or "end of page" in command:
        pyautogui.hotkey('ctrl', 'end')
        voice_label = "END OF PAGE"
        voice_label_time = time.time()
    elif "top of page" in command or "go to top" in command or "beginning of page" in command:
        pyautogui.hotkey('ctrl', 'home')
        voice_label = "TOP OF PAGE"
        voice_label_time = time.time()
    elif "scroll up" in command:
        pyautogui.scroll(20)
        voice_label = "SCROLL UP"
        voice_label_time = time.time()
    elif "scroll down" in command:
        pyautogui.scroll(-20)
        voice_label = "SCROLL DOWN"
        voice_label_time = time.time()
    elif "page up" in command:
        pyautogui.press('pageup')
        voice_label = "PAGE UP"
        voice_label_time = time.time()
    elif "page down" in command:
        pyautogui.press('pagedown')
        voice_label = "PAGE DOWN"
        voice_label_time = time.time()
    elif "go back" in command:
        pyautogui.hotkey('alt', 'left')
        voice_label = "GO BACK"
        voice_label_time = time.time()
    elif "go forward" in command:
        pyautogui.hotkey('alt', 'right')
        voice_label = "GO FORWARD"
        voice_label_time = time.time()
    elif "refresh" in command:
        pyautogui.press('f5')
        voice_label = "REFRESHED"
        voice_label_time = time.time()
    elif "find" in command:
        pyautogui.hotkey('ctrl', 'f')
        voice_label = "FIND OPENED"
        voice_label_time = time.time()
    elif "incognito" in command:
        pyautogui.hotkey('ctrl', 'shift', 'n')
        voice_label = "INCOGNITO"
        voice_label_time = time.time()
    elif "new window" in command:
        pyautogui.hotkey('ctrl', 'n')
        voice_label = "NEW WINDOW"
        voice_label_time = time.time()
 
    elif "double click" in command:
        pyautogui.doubleClick()
        voice_label = "DOUBLE CLICK"
        voice_label_time = time.time()
    elif "right click" in command:
        pyautogui.rightClick()
        voice_label = "RIGHT CLICK"
        voice_label_time = time.time()
    elif "click" in command:
        pyautogui.click()
        voice_label = "CLICK"
        voice_label_time = time.time()
 
    elif "zoom in" in command:
        pyautogui.hotkey('ctrl', '+')
        voice_label = "ZOOM IN"
        voice_label_time = time.time()
    elif "zoom out" in command:
        pyautogui.hotkey('ctrl', '-')
        voice_label = "ZOOM OUT"
        voice_label_time = time.time()
    elif "reset zoom" in command:
        pyautogui.hotkey('ctrl', '0')
        voice_label = "ZOOM RESET"
        voice_label_time = time.time()
 
    elif "screenshot" in command:
        filename = f'C:/Users/j/Desktop/asep/screenshot_{int(time.time())}.png'
        pyautogui.screenshot().save(filename)
        voice_label = "SCREENSHOT SAVED!"
        voice_label_time = time.time()
    elif "copy" in command:
        pyautogui.hotkey('ctrl', 'c')
        voice_label = "COPIED"
        voice_label_time = time.time()
    elif "paste" in command:
        pyautogui.hotkey('ctrl', 'v')
        voice_label = "PASTED"
        voice_label_time = time.time()
    elif "redo" in command:
        pyautogui.hotkey('ctrl', 'y')
        voice_label = "REDO"
        voice_label_time = time.time()
    elif "undo" in command:
        pyautogui.hotkey('ctrl', 'z')
        voice_label = "UNDO"
        voice_label_time = time.time()
    elif "select all" in command:
        pyautogui.hotkey('ctrl', 'a')
        voice_label = "SELECT ALL"
        voice_label_time = time.time()
    elif "save" in command:
        pyautogui.hotkey('ctrl', 's')
        voice_label = "SAVED"
        voice_label_time = time.time()
    elif "bold" in command:
        pyautogui.hotkey('ctrl', 'b')
        voice_label = "BOLD"
        voice_label_time = time.time()
    elif "italic" in command:
        pyautogui.hotkey('ctrl', 'i')
        voice_label = "ITALIC"
        voice_label_time = time.time()
 
    elif "close tab" in command or "close window" in command:
        pyautogui.hotkey('ctrl', 'w')
        voice_label = "CLOSED TAB"
        voice_label_time = time.time()
    elif "new tab" in command:
        pyautogui.hotkey('ctrl', 't')
        voice_label = "NEW TAB"
        voice_label_time = time.time()
    elif "minimize" in command:
        pyautogui.hotkey('win', 'down')
        voice_label = "MINIMIZED"
        voice_label_time = time.time()
    elif "maximize" in command:
        pyautogui.hotkey('win', 'up')
        voice_label = "MAXIMIZED"
        voice_label_time = time.time()
    elif "switch window" in command or "alt tab" in command:
        pyautogui.hotkey('alt', 'tab')
        voice_label = "SWITCH WINDOW"
        voice_label_time = time.time()
    elif "show desktop" in command:
        pyautogui.hotkey('win', 'd')
        voice_label = "SHOW DESKTOP"
        voice_label_time = time.time()
    elif "lock screen" in command or "lock pc" in command:
        pyautogui.hotkey('win', 'l')
        voice_label = "LOCKING PC"
        voice_label_time = time.time()
 
    elif "volume up" in command:
        for _ in range(5): pyautogui.press('volumeup')
        voice_label = "VOLUME UP"
        voice_label_time = time.time()
    elif "volume down" in command:
        for _ in range(5): pyautogui.press('volumedown')
        voice_label = "VOLUME DOWN"
        voice_label_time = time.time()
    elif "mute" in command:
        pyautogui.press('volumemute')
        voice_label = "MUTED"
        voice_label_time = time.time()
 
    elif "start typing" in command or "begin typing" in command:
        typing_mode = True
        voice_label = "TYPING MODE ON"
        voice_label_time = time.time()
    elif "type" in command:
        text = command.replace("type", "").strip()
        if text:
            pyautogui.write(text, interval=0.05)
            voice_label = f"TYPED: {text}"
            voice_label_time = time.time()
 
    elif "switch mode" in command or "change mode" in command:
        system_state = "mode_select"
        voice_label = "SWITCHING TO MODE SELECT"
        voice_label_time = time.time()
 
    else:
        voice_label = f'Unknown: "{command}"'
        voice_label_time = time.time()
 
def voice_listener():
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=1)
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    print("Voice recognition started!")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            command = recognizer.recognize_google(audio)
            handle_voice_command(command)
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print(f"Voice error: {e}")
 
voice_thread = threading.Thread(target=voice_listener, daemon=True)
voice_thread.start()
 
prev_zoom_dist = None
cv2.namedWindow("Gesture Mouse", cv2.WINDOW_NORMAL)
 
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
 
    now = time.time()
    label = ""
    right_hand = None
    left_hand = None
 
    if result.hand_landmarks and result.handedness:
        for i, handedness in enumerate(result.handedness):
            hl = handedness[0].category_name
            if hl == "Left":
                right_hand = result.hand_landmarks[i]
            else:
                left_hand = result.hand_landmarks[i]
 
    # ════════════════════════════════════════
    # STATE: LOCKED
    # ════════════════════════════════════════
    if system_state == "locked":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
 
        cv2.putText(frame, "SYSTEM LOCKED",
                    (w//2 - 190, h//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.putText(frame, "Show SPIDER SIGN",
                    (w//2 - 175, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(frame, "and hold for 2 seconds",
                    (w//2 - 210, h//2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        if right_hand and is_spider_sign(right_hand):
            if password_hold_start is None:
                password_hold_start = now
            hold = now - password_hold_start
            remaining = PASSWORD_HOLD_TIME - hold
            bar = int((hold / PASSWORD_HOLD_TIME) * 300)
            cv2.rectangle(frame, (w//2 - 150, h//2 + 70),
                          (w//2 - 150 + bar, h//2 + 100), (0, 255, 0), -1)
            cv2.rectangle(frame, (w//2 - 150, h//2 + 70),
                          (w//2 + 150, h//2 + 100), (255, 255, 255), 2)
            cv2.putText(frame, f"Unlocking... {remaining:.1f}s",
                        (w//2 - 130, h//2 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if hold >= PASSWORD_HOLD_TIME:
                system_state = "mode_select"
                password_hold_start = None
                mode_select_start = None
        else:
            password_hold_start = None
 
    # ════════════════════════════════════════
    # STATE: MODE SELECT
    # ════════════════════════════════════════
    elif system_state == "mode_select":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 50), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
 
        cv2.putText(frame, "SELECT MODE", (w//2 - 155, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 60, 80), -1)
        cv2.putText(frame, "3 FINGERS = SIGN LANGUAGE TYPING MODE",
                    (w//2 - 215, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
 
        cv2.rectangle(frame, (15, 60), (w//2 - 8, h - 60), (30, 60, 30), -1)
        cv2.rectangle(frame, (15, 60), (w//2 - 8, h - 60), (0, 255, 0), 2)
        cv2.putText(frame, "1 FINGER", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "GESTURE MODE", (22, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(frame, "- Move cursor", (25, 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(frame, "- Left/Right click", (25, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(frame, "- Scroll up/down", (25, 252),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(frame, "- Zoom in/out", (25, 284),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(frame, "- Screenshot", (25, 316),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(frame, "- PPT control", (25, 348),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(frame, "- Sign language", (25, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
 
        cv2.rectangle(frame, (w//2 + 8, 60), (w - 15, h - 60), (40, 0, 60), -1)
        cv2.rectangle(frame, (w//2 + 8, 60), (w - 15, h - 60), (180, 105, 255), 2)
        cv2.putText(frame, "2 FINGERS", (w//2 + 22, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 105, 255), 2)
        cv2.putText(frame, "VOICE MODE", (w//2 + 22, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 105, 255), 2)
        cv2.putText(frame, "+ Hand navigate", (w//2 + 18, 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
        cv2.putText(frame, "- Open apps", (w//2 + 18, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
        cv2.putText(frame, "- PPT commands", (w//2 + 18, 252),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
        cv2.putText(frame, "- Scroll/Zoom", (w//2 + 18, 284),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
        cv2.putText(frame, "- Click/Type", (w//2 + 18, 316),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
        cv2.putText(frame, "- Volume/Tabs", (w//2 + 18, 348),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
        cv2.putText(frame, "- And much more", (w//2 + 18, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 180, 255), 2)
 
        if right_hand and not is_spider_sign(right_hand):
            up = fingers_up(right_hand)
            only_index    = up[1] and not up[2] and not up[3] and not up[4]
            index_middle   = up[1] and up[2] and not up[3] and not up[4]
            index_mid_ring = up[1] and up[2] and up[3] and not up[4]
 
            if only_index or index_middle or index_mid_ring:
                if only_index:
                    chosen = "gesture"
                elif index_middle:
                    chosen = "voice"
                else:
                    chosen = "sign_type"
                if mode_select_start is None:
                    mode_select_start = now
                hold = now - mode_select_start
                remaining = MODE_SELECT_HOLD - hold
                bar = int((hold / MODE_SELECT_HOLD) * 250)
                if chosen == "gesture":
                    bar_color = (0, 255, 0)
                elif chosen == "voice":
                    bar_color = (180, 105, 255)
                else:
                    bar_color = (0, 200, 255)
                cv2.rectangle(frame, (w//2 - 125, h - 55),
                              (w//2 - 125 + bar, h - 25), bar_color, -1)
                cv2.rectangle(frame, (w//2 - 125, h - 55),
                              (w//2 + 125, h - 25), (255, 255, 255), 2)
                mode_names = {"gesture": "GESTURE MODE", "voice": "VOICE MODE", "sign_type": "SIGN TYPING MODE"}
                cv2.putText(frame, f"Selecting {mode_names[chosen]}... {remaining:.1f}s",
                            (w//2 - 185, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
                if hold >= MODE_SELECT_HOLD:
                    system_state = chosen
                    mode_select_start = None
            else:
                mode_select_start = None
 
        if right_hand and is_spider_sign(right_hand):
            if password_hold_start is None:
                password_hold_start = now
            if now - password_hold_start >= PASSWORD_HOLD_TIME:
                system_state = "locked"
                password_hold_start = None
        else:
            password_hold_start = None
 
    # ════════════════════════════════════════
    # STATE: GESTURE MODE
    # ════════════════════════════════════════
    elif system_state == "gesture":
 
        cv2.rectangle(frame, (0, 0), (230, 40), (0, 80, 0), -1)
        cv2.putText(frame, "GESTURE MODE", (6, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
 
        if right_hand and not is_spider_sign(right_hand):
            up = fingers_up(right_hand)
            ix = int(right_hand[8].x * w)
            iy = int(right_hand[8].y * h)
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
 
            move_cursor(right_hand)
 
            index_thumb_dist = get_distance(right_hand[8], right_hand[4])
            if now - last_click_time > CLICK_COOLDOWN:
                if index_thumb_dist < 0.05 and not up[1]:
                    pyautogui.click()
                    last_click_time = now
                    label = "LEFT CLICK"
 
            if is_middle_ring_up(right_hand):
                if now - last_rclick_time > RCLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_rclick_time = now
                    label = "RIGHT CLICK"
 
            thumb_dir = get_thumb_direction(right_hand)
            if thumb_dir and now - last_ppt_time > PPT_COOLDOWN:
                if thumb_dir == "right":
                    pyautogui.press('right')
                    label = "PPT NEXT"
                    last_ppt_time = now
                elif thumb_dir == "left":
                    pyautogui.press('left')
                    label = "PPT PREV"
                    last_ppt_time = now
 
            cv2.circle(frame, (int(right_hand[4].x * w),
                                int(right_hand[4].y * h)), 8, (255, 0, 0), -1)
 
            all_fingers_up = all(up[1:])
            current_x = right_hand[9].x
            if all_fingers_up:
                if prev_hand_x is not None:
                    diff_x = current_x - prev_hand_x
                    if abs(diff_x) > 0.02:
                        wave_directions.append("right" if diff_x > 0 else "left")
                        if len(wave_directions) > 6:
                            wave_directions.pop(0)
                        if len(wave_directions) >= 4:
                            changes = sum(1 for k in range(len(wave_directions) - 1)
                                          if wave_directions[k] != wave_directions[k+1])
                            if changes >= 3 and now - last_sign_time > SIGN_COOLDOWN:
                                label = "SIGN: HELLO!"
                                last_sign_time = now
                                wave_directions.clear()
                prev_hand_x = current_x
            else:
                prev_hand_x = None
                wave_directions.clear()
 
            if detect_thankyou(right_hand) and now - last_sign_time > SIGN_COOLDOWN:
                label = "SIGN: THANK YOU!"
                last_sign_time = now
 
        if left_hand:
            up = fingers_up(left_hand)
            lx = int(left_hand[8].x * w)
            ly = int(left_hand[8].y * h)
            cv2.circle(frame, (lx, ly), 10, (255, 165, 0), -1)
 
            if all(up):
                if now - last_screenshot_time > SCREENSHOT_COOLDOWN:
                    pyautogui.screenshot().save(
                        f'C:/Users/j/Desktop/asep/screenshot_{int(now)}.png')
                    last_screenshot_time = now
                    label = "SCREENSHOT SAVED!"
            elif up[1] and up[2] and not up[3] and not up[4]:
                if now - last_scroll_time > SCROLL_COOLDOWN:
                    pyautogui.scroll(15)
                    label = "SCROLL UP"
                    last_scroll_time = now
            elif not up[1] and up[2] and up[3] and not up[4]:
                if now - last_scroll_time > SCROLL_COOLDOWN:
                    pyautogui.scroll(-15)
                    label = "SCROLL DOWN"
                    last_scroll_time = now
            elif up[1] and not up[2] and not up[3] and not up[4]:
                zoom_dist = get_distance(left_hand[8], left_hand[4])
                if prev_zoom_dist is not None:
                    diff = zoom_dist - prev_zoom_dist
                    if abs(diff) > 0.01:
                        pyautogui.hotkey('ctrl', '+' if diff > 0 else '-')
                        label = "ZOOM IN" if diff > 0 else "ZOOM OUT"
                prev_zoom_dist = zoom_dist
            else:
                prev_zoom_dist = None
 
        if right_hand and is_spider_sign(right_hand):
            if password_hold_start is None:
                password_hold_start = now
            hold = now - password_hold_start
            if hold >= PASSWORD_HOLD_TIME:
                system_state = "mode_select"
                password_hold_start = None
                mode_select_start = None
            else:
                cv2.putText(frame, f"Back to menu... {PASSWORD_HOLD_TIME - hold:.1f}s",
                            (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            password_hold_start = None
 
        cv2.putText(frame, "R: 1finger=move | pinch=Lclick | mid+ring=Rclick",
                    (5, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        cv2.putText(frame, "R: fist+thumb=PPT | wave=HELLO | flat=THANKYOU",
                    (5, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
 
    # ════════════════════════════════════════
    # STATE: VOICE MODE
    # ════════════════════════════════════════
    elif system_state == "voice":
 
        cv2.rectangle(frame, (0, 0), (200, 40), (50, 0, 80), -1)
        cv2.putText(frame, "VOICE MODE", (6, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 105, 255), 2)
 
        if right_hand and not is_spider_sign(right_hand):
            ix = int(right_hand[8].x * w)
            iy = int(right_hand[8].y * h)
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
            move_cursor(right_hand)
 
        if right_hand and is_spider_sign(right_hand):
            if password_hold_start is None:
                password_hold_start = now
            hold = now - password_hold_start
            if hold >= PASSWORD_HOLD_TIME:
                system_state = "mode_select"
                password_hold_start = None
                mode_select_start = None
            else:
                cv2.putText(frame, f"Back to menu... {PASSWORD_HOLD_TIME - hold:.1f}s",
                            (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            password_hold_start = None
 
        if typing_mode:
            cv2.rectangle(frame, (0, h - 35), (190, h), (255, 0, 150), -1)
            cv2.putText(frame, "TYPING MODE ON", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "SAY: scroll up/down | next/prev slide | click",
                        (5, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 180, 255), 2)
            cv2.putText(frame, "SAY: open chrome/word/ppt | zoom in/out | start typing",
                        (5, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 180, 255), 2)
            cv2.putText(frame, "SAY: copy | paste | save | mute | switch mode",
                        (5, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 180, 255), 2)
 
    # ════════════════════════════════════════
    # STATE: SIGN TYPING MODE
    # ════════════════════════════════════════
    elif system_state == "sign_type":
 
        cv2.rectangle(frame, (0, 0), (260, 40), (60, 60, 0), -1)
        cv2.putText(frame, "SIGN TYPING MODE", (6, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
 
        # Right hand — navigation + clicks
        if right_hand and not is_spider_sign(right_hand):
            up_r = fingers_up(right_hand)
            ix = int(right_hand[8].x * w)
            iy = int(right_hand[8].y * h)
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
 
            move_cursor(right_hand)
 
            index_thumb_dist = get_distance(right_hand[8], right_hand[4])
            if now - last_click_time > CLICK_COOLDOWN:
                if index_thumb_dist < 0.05 and not up_r[1]:
                    pyautogui.click()
                    last_click_time = now
                    label = "LEFT CLICK"
 
            if is_middle_ring_up(right_hand):
                if now - last_rclick_time > RCLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_rclick_time = now
                    label = "RIGHT CLICK"
 
            cv2.circle(frame, (int(right_hand[4].x * w),
                                int(right_hand[4].y * h)), 8, (255, 0, 0), -1)
 
        # Left hand — ASL letter typing
        if left_hand:
            up_l = fingers_up(left_hand)
            lx = int(left_hand[8].x * w)
            ly = int(left_hand[8].y * h)
            cv2.circle(frame, (lx, ly), 10, (255, 165, 0), -1)
 
            # Backspace — all 5 fingers open and held
            if all(up_l):
                if now - last_letter_time > LETTER_COOLDOWN:
                    pyautogui.press('backspace')
                    label = "BACKSPACE"
                    last_letter_time = now
                    last_typed_letter = ""
            # Space — closed fist
            elif not any(up_l):
                if now - last_letter_time > LETTER_COOLDOWN:
                    pyautogui.press('space')
                    label = "SPACE"
                    last_letter_time = now
                    last_typed_letter = ""
            # Enter — only ring+pinky up
            elif not up_l[1] and not up_l[2] and up_l[3] and up_l[4]:
                if now - last_letter_time > LETTER_COOLDOWN:
                    pyautogui.press('enter')
                    label = "ENTER"
                    last_letter_time = now
                    last_typed_letter = ""
            else:
                letter = detect_asl_letter(left_hand)
                if letter and now - last_letter_time > LETTER_COOLDOWN:
                    pyautogui.write(letter.lower(), interval=0.02)
                    label = f"LETTER: {letter}"
                    last_letter_time = now
                    last_typed_letter = letter
 
        # Spider sign to go back to mode select
        if right_hand and is_spider_sign(right_hand):
            if password_hold_start is None:
                password_hold_start = now
            hold = now - password_hold_start
            if hold >= PASSWORD_HOLD_TIME:
                system_state = "mode_select"
                password_hold_start = None
                mode_select_start = None
            else:
                cv2.putText(frame, f"Back to menu... {PASSWORD_HOLD_TIME - hold:.1f}s",
                            (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            password_hold_start = None
 
        cv2.putText(frame, "R: navigate+click | L: ASL letters A,B,D,F,I,L,O,U,V,W,Y",
                    (5, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)
        cv2.putText(frame, "L: fist=space | open palm=backspace | ring+pinky=enter",
                    (5, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)
 
    # ── Gesture label ──
    if label:
        cv2.putText(frame, label, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
 
    # ── Voice label ──
    if voice_label and (now - voice_label_time) < VOICE_LABEL_DURATION:
        cv2.putText(frame, f"VOICE: {voice_label}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 105, 255), 2)
    elif now - voice_label_time >= VOICE_LABEL_DURATION:
        voice_label = ""
 
    cv2.imshow("Gesture Mouse", frame)
    cv2.setWindowProperty("Gesture Mouse", cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow("Gesture Mouse", 400, 300)
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()