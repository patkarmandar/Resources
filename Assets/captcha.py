import requests
import time
import re
import sys
from io import BytesIO
from PIL import Image
import pytesseract
import cv2
from collections import Counter
import numpy as np

BASE = "https://lg.jio.com"

URL_SESSION = f"{BASE}/py/session"
URL_CAPTCHA_META = f"{BASE}/py/getValidCaptcha"
URL_CAPTCHA_IMG = f"{BASE}/captchaImg/"
URL_VALIDATE = f"{BASE}/py/getValidCaptcha"
URL_COMMAND = f"{BASE}/py/commandSend"
URL_RECV = f"{BASE}/py/recvData"

HEADERS = {
    "Host": "lg.jio.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) Gecko/20100101 Firefox/146.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://lg.jio.com",
    "Sec-Gpc": "1",
    "Referer": "https://lg.jio.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Priority": "u=0",
    "Content-Length": "0",
    "Te": "trailers"
}

RECV_RETRIES = 5

session = requests.Session()
session.headers.update(HEADERS)

SESSION_COOKIE = None

# ================== SESSION HANDLING ==================

def attach_session_cookie():
    session.headers.update({
        "Cookie": SESSION_COOKIE
    })

def get_session():
    global SESSION_COOKIE
    r = session.post(URL_SESSION)
    r.raise_for_status()

    if not r.cookies:
        raise Exception("No cookies returned from /py/session")

    cookie = list(r.cookies)[0]
    SESSION_COOKIE = f"{cookie.name}={cookie.value}"
    print(f"[+] Session cookie captured: {SESSION_COOKIE}")

# ================== CAPTCHA HANDLING ==================

def get_captcha_name():
    attach_session_cookie()
    r = session.post(URL_CAPTCHA_META, json={"T": "get"})
    r.raise_for_status()
    print(f"[+] Captcha File: {r.json()['captchaText']}")
    return r.json()["captchaText"]

def download_captcha(name):
    attach_session_cookie()
    r = session.get(URL_CAPTCHA_IMG + name)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

# Preprocessing for OCR
def preprocess(img, block_size=31):
    img = img.convert("RGB")
    open_cv_image = np.array(img)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.bitwise_not(th)  # invert for tesseract
    return th

# OCR with 5 attempts + voting
def solve_captcha(img, max_tries=5):
    results = []
    OCR_CONFIG = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    for i in range(max_tries):
        block_size = 21 + i*5
        if block_size % 2 == 0:
            block_size += 1  # ensure odd
        preprocessed_img = preprocess(img, block_size)
        text = pytesseract.image_to_string(preprocessed_img, config=OCR_CONFIG)
        text = text.strip().replace(" ", "").replace("\n", "")
        if text:
            results.append(text)

    if not results:
        raise Exception("OCR failed after multiple attempts")

    # Voting: pick most common result
    most_common = Counter(results).most_common(1)[0][0]
    print(f"[+] Captcha Text: {most_common}")
    return most_common

def validate_captcha(text):
    attach_session_cookie()
    r = session.post(URL_VALIDATE, json={"T": "valid", "captcha": text})
    r.raise_for_status()
    data = r.json() 
    return data.get("captchaResult", "").lower() == "true"

# ================== COMMAND & RECV ==================

def send_command(text):
    attach_session_cookie()
    payload = {
        "sourceIP": "AMD-NLD-01",
        "command": "1",
        "destName": "223.228.36.116",
        "captcha": text
    }
    r = session.post(URL_COMMAND, json=payload)
    r.raise_for_status()
    return r.json()["R"]

def recv_data_with_retry(R):
    attach_session_cookie()
    for attempt in range(1, RECV_RETRIES + 1):
        try:
            r = session.post(URL_RECV, json={"R": R})
            r.raise_for_status()
            print(f"\n=== RECV SUCCESS (attempt {attempt}) ===")
            print(r.text)
            return
        except Exception as e:
            print(f"recvData retry {attempt} failed")
            time.sleep(1)
    raise Exception("recvData failed after retries")

# ================== MAIN FLOW ==================

get_session()

captcha_name = get_captcha_name()
captcha_img = download_captcha(captcha_name)
captcha_text = solve_captcha(captcha_img)

if not validate_captcha(captcha_text):
    raise Exception("Captcha invalid")

R = send_command(captcha_text)

recv_data_with_retry(R)
