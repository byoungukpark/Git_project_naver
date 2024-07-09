import json
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import asyncio
from tensorflow.keras.models import load_model


mp_holistic = mp.solutions.holistic  # holistic: 얼굴, 손 등 감지

def process_frame(image_data):
    # base64 형식의 이미지 데이터를 bytes로 디코딩
    image_bytes = base64.b64decode(image_data)
    # bytes를 NumPy 배열로 변환
    np_arr = np.frombuffer(image_bytes, np.uint8)
    # NumPy 배열을 이미지로 변환
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # 이미지 수정 불가 (결과 왜곡 방지?)
    results = model.process(image)  # 모델을 사용해 입력 이미지에 대한 예측 수행
    image.flags.writeable = True  # 이미지 다시 수정가능
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results



script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(script_directory, "dataset")
print("현재 작업 디렉토리:", script_directory)

MODEL_PATH = os.path.join(script_directory,'final_model/model_ko.h5.keras')
model = load_model(MODEL_PATH, compile=False) # 코랩 사용시 compile=False 필수

data_file_list = os.listdir(dataset_directory)
data_file_list = sorted(data_file_list, reverse=True)

actions = {}
for file_name in data_file_list:
    parts = file_name.split("_")
    if parts[1] not in actions:
        actions[int(parts[0])] = parts[1]
        # actions.append((parts[1], parts[0])) # 파일 목록(npy)에서 단어 추출

print(actions)

sentence_length = 10
seq_length = 60

#기본 설정
seq = []
action_seq = [] 
previous = '' # 이전 단어

dc=0 # debug_count




# 미디어 파이프 포즈 모델 정의??
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


#카메라 키기
cap = cv2.VideoCapture(0)



while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.\n웹캠을 사용중인 프로세스를 중지해주세요.")
        continue

    image2 = image
    image2 = cv2.cvtColor(cv2.flip(image2, 1), cv2.COLOR_BGR2RGB)
    image2.flags.writeable = False
    results = pose.process(image2)
    image2.flags.writeable = True
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image2,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
    #화면
    cv2.imshow('MediaPipe Pose', cv2.flip(image2, 1))


    # # frame = process_frame(image)
    image, result = mediapipe_detection(image, pose)

    if result.pose_landmarks is not None:
        #데이터 넣을 빈 배열
        d = []
        for k in range(33):
            d.append(result.pose_landmarks.landmark[k].x)
            d.append(result.pose_landmarks.landmark[k].y)
            d.append(result.pose_landmarks.landmark[k].z)
            d.append(result.pose_landmarks.landmark[k].visibility)

        seq.append(d)


        dc+=1 
        # print(dc, "debug1.seq:", len(seq))

        if len(seq) < seq_length: # 시퀀스 최소치가 쌓인 이후부터 판별
            continue

        if len(seq) > seq_length * 5:  # 대충 한번씩 쌓인거 지움
            seq = seq[-seq_length:]
        
        # 시퀀스 데이터를 신경망 모델에 입력으로 사용할 수 있는 형태로 변환
        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze() # 각 동작에 대한 예측 결과 (동작이 각 라벨의 동작일 확률 반환)

        i_pred = int(np.argmax(y_pred)) # 최댓값 인덱스: 예측값이 가장 높은 값(동작)의 인덱스
        conf = y_pred[i_pred] # 가장 확률 높은 동작의 확률이

        # print("conf debug", conf)
        if conf < 0.7:   # 70% 이상일 때만 수행
            continue

        # debug
        # print("debug2.예측동작:", i_pred)


        action = actions[i_pred]
        
        if previous == action: 
            continue  # 중복 전달 회피
        else:
            print("debug2.예측동작:", i_pred)
        previous = action

        seq=[] # 초기화


    key = cv2.waitKey(1)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()