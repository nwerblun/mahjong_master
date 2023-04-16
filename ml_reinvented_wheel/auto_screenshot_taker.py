import pyautogui as pa
import time
import win32gui
import os

scs = 0
try:
    start = time.time()
    while True:
        if time.time() - start >= 5:
            hwnd = win32gui.GetForegroundWindow()
            win_name = win32gui.GetWindowText(hwnd)
            # Cheap hack :)
            if "mahjong" in win_name.lower() and "mozilla" in win_name.lower():
                # Firefox is displayed as 'web name — Mozilla Firefox'. — is a double hyphen character.
                website_name = win_name.replace(" ", "_").split("—")[0].lower()
                files = os.listdir(".\\img\\")
                if len(files) > 0:
                    prev_scs = [int(s.replace(".png", "").replace(website_name, ""))
                                for s in files if website_name in s]
                    if len(prev_scs) > 0:
                        prev_scs = sorted(prev_scs)
                        prev_max = prev_scs[-1]
                    else:
                        prev_max = -1
                    sc = pa.screenshot(".\\img\\"+website_name+str(prev_max+1)+".png")
                    print("Took a screenshot. Saved at", ".\\img\\" + website_name + str(prev_max) + ".png")
                else:
                    sc = pa.screenshot(".\\img\\"+website_name+"0.png")
                    print("Took a screenshot. Saved at", ".\\img\\" + website_name + "0.png")
                scs += 1
                print("Now taken", str(scs), "screenshots this session")
            else:
                print("Time passed but no game detected.")
            start = time.time()
except KeyboardInterrupt as e:
    print("All done I guess..")
    print("Captured", str(scs), "screenshots")


