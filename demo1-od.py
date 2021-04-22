import cv2
from azure.cogniteservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

cam = cv2.VideoCapture(0)
dim = (640, 480)
ret, image = cam.read()
image = cv2.resize(image, dim)

credentials = ApiKeyCredentials(in_headers={'Prediction-key':"<PREDICTION_KEY"})
predictor = CustomVisionPredictionClient("<ENDPOINT_URL>", credentials)

cv2.imwrite('capture.jpg', image)

with open('capture.jpg', mode="rb") as captured_image:
    res = predictor.detect_image("<PROJECT_ID>", "<ITERATION_NAME>", captured_image)

for prediction in res.predictions:
    if prediction.probability > 0.9:
        bbox = prediction.bounding_box
        result_image = cv2.rectangle(image, (int(bbox.left * 640), int(bbox.top * 480)), (int((bbox.left + bbox.width) * 640), int((bbox.top + bbox.height) * 480)), (0, 255, 0), 3)
        cv2.imwrite('result.jpg', result_image)
        
cam.release()