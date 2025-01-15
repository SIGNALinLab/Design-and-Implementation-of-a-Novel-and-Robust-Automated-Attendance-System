import cv2
import time
import random
import requests
# Time variables (All times in seconds)
T1 = 30  
T2 = 30  
end_time = 100  

# Set the URL for the flashlight control
camera_url_flash = 'http://192.168.1.250:8080/RPC2'

# Example headers
headers = {
    'Content-Type': 'application/json'
}

# Flashlight ON payload
flashlight_on_payload = {
    "method": "CoaxialControlIO.control",
    "params": {
        "info": [{"Type": 1, "IO": 1, "TriggerMode": 2}],  # Adjust these values if needed
        "channel": 0
    },
    "id": 253,
    "session": "e053f01956ff750bb8eac11d748ab3b1"  # Use your session ID
}

def turn_on_flashlight():
    # Send the request to turn on the flashlight
    response_on = requests.post(camera_url_flash, headers=headers, json=flashlight_on_payload)
    print("Flashlight ON:", response_on.status_code, response_on.text)

def capture_image(image_name):
    # Turn on the flashlight
    turn_on_flashlight()

    # Wait for 5 seconds
    time.sleep(5)

    # Capture image from camera
    camera_url = 'rtsp://admin:12345-Qwert@192.168.1.250:554/live'
    cap = cv2.VideoCapture(camera_url)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image_path = f'{image_name}.jpg'
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as '{image_path}'")
            return image_path
        else:
            print("Failed to capture image")
            return None
    else:
        print("Failed to open camera stream")
        return None

    cap.release()


# Get a random time for the third capture between second image and session end
def get_random_time():
    return T2 + random.randint(1, end_time - (T1 + T2))

def main():
    print(f"Waiting for {T1} seconds to capture the first image...")
    time.sleep(T1)
    first_image_path = capture_image('captured_image_T1')

    if first_image_path:
        print(f"First image captured at {T1} seconds")

    print(f"Waiting for another {T2} seconds to capture the second image...")
    time.sleep(T2)
    second_image_path = capture_image('captured_image_T2')

    if second_image_path:
        print(f"Second image captured at {T1 + T2} seconds")

    random_time = get_random_time()
    print(f"Waiting for {random_time} seconds to capture the third image...")
    time.sleep(random_time)
    third_image_path = capture_image('captured_image_T3')

    if third_image_path:
        print(f"Third image captured at {T1 + T2 + random_time} seconds")

if __name__ == "__main__":
    main()
