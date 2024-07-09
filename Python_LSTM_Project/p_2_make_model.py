import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

data = np.concatenate([
    np.load('total_dataset.npy'),
], axis=0)


# 분류 라벨 분리
# data = data[0]
# print(data)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

# print("x_data.shape : ")
# print(x_data.shape)

# 라벨 개수
# label_length = len(list(set(labels)))

# print(label_length)

labels = [int(float(label)) for label in labels]
label_length = len(labels)

print("labals len :")
print(len(labels))
print("labels :")
print(labels)
# y_data = to_categorical(labels, num_classes=label_length) # 원핫 인코딩
y_data = tf.keras.utils.to_categorical(labels, num_classes=label_length)



# 데이터를 학습(train)과 검증(validation) 세트로 나눔 -> 모델의 성능을 평가 && 과적합 방지
# 과적합: 데이터 크기 작을 때 || 단일 샘플 데이터 세트 장기간 훈련
# x: 입력, y: 출력
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)
x_train = x_data.astype(np.float32)
y_train = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2024) # test_size 조정



# 모델 구조(입력층, 히든층, 출력층) 정의, 컴파일
model = Sequential([
    # input층 따로 분리해도 좋음
    # 활성함수(비선형성 추가 함수)는 relu 주로 사용
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]), # (512,30,198)[1:3] => (30,198): (시퀀스 길이, 특징 수)
                                                    # input 데이터 전체 개수는 LSTM 레이어에서 자동 연산
    Dense(64, activation='relu'), # 완전연결층
    Dropout(0.3),  # 과적합 방지층 (노드를 0.3만큼 꺼줌)
    Dense(64, activation='relu'), 
    Dense(label_length, activation='softmax')  # 출력층 softmax: 분류 모델에 적합
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()



# 모델 훈련 (epochs=200: 학습량)
# 모델 저장 경로 (Modelcheckpoint()) 확인 !! 
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=[
        ModelCheckpoint('dataset/final_model/model_ko.h5.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.4, patience=10, verbose=1, mode='auto')
    ]
)