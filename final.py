import cv2
import numpy as np
import RPi.GPIO as GPIO
import time


# GPIO 핀 설정 및 초기화
GPIO.setmode(GPIO.BCM)  # GPIO 핀 번호 체계를 BCM 방식으로 설정
GPIO.setwarnings(False)  # GPIO 경고 메시지 비활성화


# 핀 번호 정의
TRIG = 23  # 초음파 송신 핀
ECHO = 24  # 초음파 수신 핀
PWMA = 18  # 왼쪽 모터 속도 제어
AIN1 = 22  # 왼쪽 모터 방향 제어 1
AIN2 = 27  # 왼쪽 모터 방향 제어 2
PWMB = 13  # 오른쪽 모터 속도 제어
BIN1 = 26  # 오른쪽 모터 방향 제어 1
BIN2 = 19  # 오른쪽 모터 방향 제어 2


# GPIO 핀 초기화
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)


GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)


GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)


# 모터 PWM 설정 (주파수 100Hz)
L_Motor = GPIO.PWM(PWMA, 100)
R_Motor = GPIO.PWM(PWMB, 100)
L_Motor.start(0)
R_Motor.start(0)


# HSV 색상 범위 설정 (차선과 정지선 감지를 위한 색상 범위)
lower_blue = (110-3, 120, 150)  # 파란색 하한값 (HSV)
upper_blue = (110+5, 255, 255)  # 파란색 상한값 (HSV)


lower_red = (6-6, 110, 120)  # 빨간색 하한값 (HSV)
upper_red = (6+4, 255, 255)  # 빨간색 상한값 (HSV)


lower_yellow = (19-1, 110, 120)  # 노란색 하한값 (HSV)
upper_yellow = (19+5, 255, 255)  # 노란색 상한값 (HSV)


# 모터 제어 함수
def motor_go(speed):
   #차량을 직진시키는 함수.


   GPIO.output(AIN1, False)
   GPIO.output(AIN2, True)
   L_Motor.ChangeDutyCycle(speed)


   GPIO.output(BIN1, False)
   GPIO.output(BIN2, True)
   R_Motor.ChangeDutyCycle(speed)


def motor_left(speed):
   #차량을 좌회전시키는 함수.


   GPIO.output(AIN1, False)
   GPIO.output(AIN2, False)
   L_Motor.ChangeDutyCycle(0)


   GPIO.output(BIN1, False)
   GPIO.output(BIN2, True)
   R_Motor.ChangeDutyCycle(speed)


def motor_right(speed):
   #차량을 우회전시키는 함수.


   GPIO.output(AIN1, False)
   GPIO.output(AIN2, True)
   L_Motor.ChangeDutyCycle(speed)


   GPIO.output(BIN1, False)
   GPIO.output(BIN2, False)
   R_Motor.ChangeDutyCycle(0)


def motor_stop():
   # 차량을 정지시키는 함수.


   L_Motor.ChangeDutyCycle(0)
   R_Motor.ChangeDutyCycle(0)


# 초음파 센서를 이용한 거리 측정 함수
def getDistance():


   # 초음파 센서를 사용하여 장애물과의 거리를 측정.
   # return: 장애물까지의 거리(cm)


   GPIO.output(TRIG, True)
   time.sleep(0.00001)
   GPIO.output(TRIG, False)


   while GPIO.input(ECHO) == 0:
       pulse_start = time.time()


   while GPIO.input(ECHO) == 1:
       pulse_end = time.time()


   pulse_duration = pulse_end - pulse_start
   distance = pulse_duration * 17150  # 거리 계산 공식 (34300/2)
   return round(distance, 2)


# 관심 영역(ROI) 설정 함수
def region_of_interest(img, vertices):
   #관심 영역(Region of Interest, ROI)을 설정하여 불필요한 부분 제거.


   mask = np.zeros_like(img)  # 입력 이미지와 동일한 크기의 검정 마스크 생성
   if len(img.shape) > 2:  # 컬러 이미지인 경우
       channel_count = img.shape[2]
       ignore_mask_color = (255,) * channel_count
   else:  # 흑백 이미지인 경우
       ignore_mask_color = 255


   cv2.fillPoly(mask, vertices, ignore_mask_color)  # 다각형 영역을 흰색으로 채움
   return cv2.bitwise_and(img, mask)  # 마스크를 적용하여 ROI 추출


# 이미지 전처리 함수
def preprocessing(img, low_threshold, high_threshold, kernel_size):
   # 입력 이미지를 전처리하여 차선 및 정지선 탐지 준비.
   img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR 이미지를 HSV로 변환


   # ROI 설정
   vertices = np.array([[(20, 315), (20, 210), (160, 130), (470, 130), (620, 210), (620, 315)]], dtype=np.int32)
   masked_image = region_of_interest(img_hsv, vertices)


   # HSV 색상 필터링
   mask_red = cv2.inRange(masked_image, lower_red, upper_red)
   mask_blue = cv2.inRange(masked_image, lower_blue, upper_blue)
   mask_yellow = cv2.inRange(masked_image, lower_yellow, upper_yellow)


   # 세 가지 색상 마스크를 합침
   mask = cv2.bitwise_or(mask_red, mask_blue)
   mask = cv2.bitwise_or(mask, mask_yellow)


   # 블러링과 이진화
   blur = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
   _, thresh = cv2.threshold(blur, low_threshold, high_threshold, cv2.THRESH_BINARY)
   return thresh


# 차선 및 정지선 감지 함수
def process_frame(frame):
   """
   입력 프레임에서 차선 및 정지선을 감지.
   :param frame: 입력 이미지 프레임
   :return: 감지된 윤곽선 리스트
   """
   frame = cv2.resize(frame, (640, 360))  # 이미지 크기 조정
   thresh = preprocessing(frame, low_threshold=120, high_threshold=255, kernel_size=11)


   # 윤곽선 찾기
   contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   return contours


# 메인 루프
def main():
   """
   차량의 카메라 입력과 센서 데이터를 처리하여 자율주행 동작을 수행.
   """
   cap = cv2.VideoCapture(0)  # 카메라 입력 초기화
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # 프레임 높이 설정


   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           print("카메라 프레임을 읽을 수 없습니다.")
           break


       contours = process_frame(frame)  # 차선 및 정지선 감지


       # 윤곽선 분석
       for contour in contours:
           area = cv2.contourArea(contour)
           if area > 500:  # 정지선 크기 조건
               x, y, w, h = cv2.boundingRect(contour)
               if y > 250:  # 정지선의 Y 좌표 기준
                   print("정지선 감지: 정지합니다.")
                   motor_stop()
                   time.sleep(3)
               else:
                   print("선 탐지")
                   motor_go(40)


   cap.release()  # 카메라 리소스 해제
   cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
   GPIO.cleanup()  # GPIO 핀 정리




# 실행
"""
프로그램의 진입점
프로그램이 직접 실행할 때 자동으로 실행
"""
if __name__ == "__main__":
   try:
       main()
       prev_distance = getDistance()  # 초기 거리 측정
       while True:
           distance = getDistance()  # 현재 거리 측정


           # 현재 거리와 이전 거리의 차이를 계산
           distance_diff = prev_distance - distance


           if distance > 50:  # 거리가 50cm 이상이면
               motor_go(60)  # 빠르게 전진
               print(f"Forward: {distance} cm")
           elif distance <= 50 and distance > 10:
               motor_go(30)  # 느리게 전진
               print(f"SLOW: {distance} cm")


               if distance_diff > 5:  # 앞차의 속도 감소 감지
                   motor_go(max(10, 30 - 10))  # 속도를 줄이지만 최소값은 10
                   print(f"Reduce speed: {distance} cm")
           else:
               motor_go(0)  # 10cm 이내면 정지
               print(f"STOP: {distance} cm")


           if distance_diff > 10:  # 갑자기 튀어나온 사물 감지
               motor_go(0)  # 급정거
               print(f"Sudden STOP: {distance} cm")


           prev_distance = distance  # 이전 거리를 현재 거리로 갱신


   except KeyboardInterrupt:
       print("Program terminated.")
       GPIO.cleanup()



