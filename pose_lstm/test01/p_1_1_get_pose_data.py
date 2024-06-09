import cv2
import mediapipe as mp
import numpy as np
import os, time

seq_length = 30
speed = 0.03
time_to_start = 2 # 초


# 미디어 파이프 포즈 모델 정의
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#카메라 키기
cap = cv2.VideoCapture(0)

#파일 만들기 
os.makedirs('dataset', exist_ok=True)

action, idx = None, None
stop_=False

a=0 # frame debug

# print(f'1회 데이터 입력 시간: {secs_for_action}초')
print(f'데이터 입력 시작 시 딜레이: {time_to_start}초')

anounce_for_user = f'''
웹캠 화면에서 메뉴 선택
l 데이터를 쌓을 단어 입력
. 데이터 입력 시작
y 데이터 입력 준비 시간 변경(defalut: {time_to_start}s)
ESC 종료
'''
print(anounce_for_user)

while cap.isOpened():
    ###
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.\n웹캠을 사용중인 프로세스를 중지해주세요.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
    
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    key = cv2.waitKey(1)
    #=====================여기까진 실행 가능 ==================


    if key == ord('l'):
        # 단어, 라벨 입력
        action = input('단어 입력: ')
        while True:
            idx = input('라벨 입력: ')
            try:
                int_value = int(idx)
                break
            except ValueError:
                print("정수 값을 입력해주세요.")
        print(f'({action}, {idx}) 입력 완료')
    

    if key == ord('y'):
        while True:
            time_to_start = input(f'{time_to_start} -> ')
            try:
                time_to_start = float(time_to_start)
                break
            except ValueError:
                print("실수 값을 입력해주세요.")
        print('데이터 입력 준비 시간 변경 완료')
        
    if key == ord('.'):
        if action is None or idx is None:
            print("l을 눌러 입력할 단어를 설정해주세요.")
            continue
        print(f'({action}, {idx}): 데이터 생성을 {time_to_start}초 뒤 시작합니다.')
        print('q: 중단(준비 시간 이후 중단 가능)')
        data = []
        ###
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.putText(img, f'Ready...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(int(time_to_start*1000)) 

        i=0
        while i < seq_length:
            time.sleep(speed)
            a+=1
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

               
            #데이터 넣을 빈 배열
            d = []

            for k in range(33):
                d.append(results.pose_landmarks.landmark[k].x)
                d.append(results.pose_landmarks.landmark[k].y)
                d.append(results.pose_landmarks.landmark[k].z)
                d.append(results.pose_landmarks.landmark[k].visibility)
            
            d=np.concatenate([d, [idx]])
            data.append(d)
            i+=1
            
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                stop_=True
                break
        if stop_:
            stop_=False
            print("데이터 생성 중단")
            continue
        print("frame: ", a)
        ###
        data = np.array(data)
        print(action, data.shape) #debug
        


        # 시퀀스 데이터 생성
        full_seq_data = []
        full_seq_data.append(data)

        # list -> numpy
        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape) # debug

        # 저장할 npy 파일 이름
        file_name = str(idx) + '_' + str(action) + '_s_0.npy'
        
        # 저장
        script_directory = os.path.dirname(os.path.abspath(__file__))
        save_data = os.path.join(script_directory, "dataset", file_name)
        ##
        save_file_num = 0
        while os.path.exists(save_data):
            save_file_num += 1
            length_except_num = len(save_data.split('_')[-1])
            save_data = save_data[:-length_except_num] # 'label_action_'
            save_data += str(save_file_num) + '.npy'
        ##
        np.save(save_data, full_seq_data)

        # 프레임 단위 데이터 저장 디렉토리
        file_name = str(idx) + '_' + str(action) + '_s_'+str(save_file_num)+'.npy'
        frame_data = os.path.join(script_directory, "dataset_frame", file_name) # 프레임 데이터
        np.save(frame_data, data)   # 저장

        print(f'({action}, {idx}):', data.shape, full_seq_data.shape, f'\n{file_name} 데이터 생성 완료')

        
        

    if key == 27:  # ESC 키를 누르면 루프 종료
        break
    ###



