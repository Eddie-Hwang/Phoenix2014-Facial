# fashion_pose.py : MPII를 사용한 신체부위 검출
import cv2

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
# 각 파일 path
# protoFile = "D:\\python_D\\fashion_data\\pose_deploy_linevec_faster_4_stages.prototxt"
# weightsFile = "D:\\python_D\\fashion_data\\pose_iter_160000.caffemodel"

protoFile = './external_lib/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
weightsFile = './external_lib/openpose/models/pose/mpi/pose_iter_160000.caffemodel'

hand_proto = './external_lib/openpose/models/hand/pose_deploy.prototxt'
hand_weights = './external_lib/openpose/models/hand/pose_iter_102000.caffemodel'
 
# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
hand_net = cv2.dnn.readNetFromCaffe(hand_proto, hand_weights)

# 이미지 읽어오기
# image = cv2.imread("D:\\python_D\\fashion_data\\full_body7.png")
image = cv2.imread('./PHOENIX-2014-T/features/fullFrame-227x227px/dev/01April_2010_Thursday_heute-6697/images0008.png')
image = cv2.resize(image, dsize=(200,200), interpolation=cv2.INTER_LINEAR)

# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape
 
# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 
# network에 넣어주기
net.setInput(inpBlob)
hand_net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()
hand_output = net.forward()

# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]
print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

# 키포인트 검출시 이미지에 그려줌
points = []
for i in range(0,15):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob > 0.1 :    
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else :
        points.append(None)

# hand_points = []
# for j in range(0, 21):
#     probMap = hand_output[0, j, :, :]
#     minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
#     h_x = (imageWidth * point[0]) / W
#     h_y = (imageHeight * point[1]) / H 

#     if prob > 0.1:
#         cv2.circle(image, (int(h_x), int(h_y)), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#         cv2.putText(image, "{}".format(j), (int(h_x), int(h_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
#         hand_points.append((int(h_x), int(h_y)))
#     else:
#         hand_points.append(None)
# cv2.imshow("Output-Keypoints",image)
# cv2.waitKey(0)
cv2.imwrite('./output_keypoints.png', image)

# 이미지 복사
imageCopy = image

# 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
for pair in POSE_PAIRS:
    partA = pair[0]             # Head
    partA = BODY_PARTS[partA]   # 0
    partB = pair[1]             # Neck
    partB = BODY_PARTS[partB]   # 1
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)


# # cv2.imshow("Output-Keypoints",imageCopy)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
cv2.imwrite('./output_keypoints_copy.png', imageCopy)