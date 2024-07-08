import cv2
import numpy as np

def main():
    # 모델 경로 및 설정
    model_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pb'
    config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

    # 클래스 레이블 로딩 (COCO 데이터셋 클래스)
    with open('coco_labels.txt', 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')

    # 모델 로딩
    net = cv2.dnn_DetectionModel(model_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # 카메라 또는 비디오 스트림 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오를 읽을 수 없습니다.")
            break

        # 객체 인식 수행
        classes, confidences, boxes = net.detect(frame, confThreshold=0.5)

        # 인식된 객체 표시
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            label = f'{labels[classId]}: {confidence:.2f}'
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 화면에 표시
        cv2.imshow('Object Detection', frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
