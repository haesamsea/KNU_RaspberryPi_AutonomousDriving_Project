
import cv2
import numpy as np
import RPi.GPIO as GPIO

PWMA = 18  # 좌측 모터 속도 제어 PWM 핀
AIN1   =  22 # 좌측 모터 방향 제어 핀 1
AIN2   =  27 # 좌측 모터 방향 제어 핀 2

PWMB = 23  # 우측 모터 속도 제어 PWM 핀
BIN1   = 25 # 우측 모터 방향 제어 핀 1
BIN2  =  24 # 우측 모터 방향 제어 핀 2

# PWM판: 모터의 속도를 제어하기 위해 신호 출력
#방향 제어 핀: 모터의 회전 방향 설정    

#전진 함수
def motor_go(speed): 
    L_Motor.ChangeDutyCycle(speed) # 좌측 모터 속도 설정
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(speed) # 우측 모터 속도 설정
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1

#우회전 함수    
def motor_right(speed):
    L_Motor.ChangeDutyCycle(speed) # 좌측 모터 속도 설정 (동작)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(0) # 우측 방향 설정 (정지 상태에 필요 없음)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1

#좌회전 함수    
def motor_left(speed):
    L_Motor.ChangeDutyCycle(0)  # 좌측 모터 정지
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1
    R_Motor.ChangeDutyCycle(speed) # 우측 모터 속도 설정 (동작)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1
        
#GPIO 초기화
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)

L_Motor= GPIO.PWM(PWMA,100)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB,100)
R_Motor.start(0)

def main():
    camera = cv2.VideoCapture(0)  #카메라를 비디오 입력으로 사용. -1은 기본설정이라는 뜻
    camera.set(3,160)  #띄울 동영상의 가로사이즈 160픽셀
    camera.set(4,120)  #띄울 동영상의 세로사이즈 120픽셀

    while( camera.isOpened() ): #카메라가 Open되어 있다면,
        ret, frame = camera.read() #카메라를 읽어서 image값에 넣습니다.
        frame = cv2.flip(frame,-1)#카메라 이미지를 flip, 뒤집습니다. -1은 180도 뒤집는다
        cv2.imshow('normal',frame) #'normal'이라는 이름으로 영상을 출력
        
        crop_img =frame[60:120, 0:160] #세로는 60~120픽셀, 가로는 0~160픽셀로 crop(잘라냄)한다.
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) #이미지를 회색으로 변경
    
        blur = cv2.GaussianBlur(gray,(5,5),0) #가우시간 블러로 블러처리를 한다.
        
         _, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)
        #임계점 처리로, 123보다 크면, 255로 변환
        #123밑의 값은 0으로 처리한다. 흑백으로 색을 명확하게 처리하기 위해서
        
        #이미지를 압축해서 노이즈를 없앤다.
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask',mask) #mask이라는 이름으로 영상을 출력
    
        height, width = mask.shape
        left_half = mask[:, :width // 2]
        right_half = mask[:, width // 2:]

        # 왼쪽 선 감지
        left_contours, _ = cv2.findContours(left_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_cx = None
        if left_contours:
            largest_left = max(left_contours, key=cv2.contourArea)
            M_left = cv2.moments(largest_left)
            if M_left['m00'] > 0:
                left_cx = int(M_left['m10'] / M_left['m00'])

        # 오른쪽 선 감지
        right_contours, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_cx = None
        if right_contours:
            largest_right = max(right_contours, key=cv2.contourArea)
            M_right = cv2.moments(largest_right)
            if M_right['m00'] > 0:
                right_cx = int(M_right['m10'] / M_right['m00']) + width // 2

        # 두 선의 중점 계산
        if left_cx is not None and right_cx is not None:
            midpoint = (left_cx + right_cx) // 2
            center = width // 2
            error = center - midpoint

            # ERROR 값을 기반으로 차량 방향 조정
            if error > 15:
                print("Turn Left!")
                motor_left(40)
            elif error < -15:
                print("Turn Right!")
                motor_right(40)
            else:
                print("Go!")
                motor_go(40)
        else:
            # 선을 감지하지 못했을 때 정지
            print("No lines detected. Stopping.")
            motor_go(0)

        if cv2.waitKey(1) == ord('q'):  #만약 q라는 키보드값을 읽으면 종료합니다.
            break
    
    cv2.destroyAllWindows() #이후 openCV창을 종료합니다.

if __name__ == '__main__':
    main()
    GPIO.cleanup()