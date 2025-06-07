import os
import json
import pandas as pd
from flask import Flask, request, jsonify
import recommend_service

import cv2
import numpy as np
import onnxruntime as ort
import base64

app = Flask(__name__)

# 클래스 레이블
class_labels = ['back', 'negative', 'neutral', 'positive']
label_mapping = {
    'negative': 'Negative',
    'neutral': 'Neutral',
    'positive': 'Positive'
}

# ONNX 모델 로드
onnx_model_path = "best.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, 640, 640]

liked_songs = {
    'Negative': [],
    'Neutral': [],
    'Positive': []
}

# 이미지 전처리 함수
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[3], input_shape[2]))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# base64 문자열을 OpenCV 이미지로 변환
def decode_base64_image(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        return None


# ONNX 출력 후처리 함수
def postprocess(output, frame, confidence_threshold=0.5):
    predictions = output[0][0]
    top_score = 0
    top_label = None

    for pred in predictions:
        objectness = float(pred[4])
        if objectness > confidence_threshold:
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_score = float(class_scores[class_id])

            if class_score > top_score:
                top_score = class_score
                top_label = class_labels[class_id]

    return top_label, top_score

@app.route('/emotion', methods=['POST'])
def emotion_inference():
    try:
        request_json = request.get_json(force=True)
        if 'encodedFaceImage' not in request_json:
            return error_message("No image provided in base64 format", 400)

        base64_image = request_json['encodedFaceImage']
        frame = decode_base64_image(base64_image)

        if frame is None:
            return error_message("Invalid base64 image", 400)

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        emotion_label, score = postprocess(outputs, frame)

        if emotion_label in label_mapping:
            state = label_mapping[emotion_label]

            recommended_song = recommend_service.recommend(state, liked_songs[state])
            liked_songs[state].append(recommended_song[:-4])
            recommended_song_url = return_url(recommended_song)

            response = format_response(recommended_song, recommended_song_url, state, score)
            return ok(response)
        else:
            return error_message("Face not detected or emotion unclear", 200)

    except Exception as e:
        return error_message(str(e), 500)

@app.route('/request', methods=['POST'])
def recommend_songs():
    try:
        request_json = request.get_json(force=True)
        state = request_json['state']

        recommended_song = recommend_service.recommend(state, liked_songs[state])
        liked_songs[state].append(recommended_song[:-4])
        recommended_song_url = return_url(recommended_song)

        response = format_response(recommended_song, recommended_song_url, state)
        return ok(response)

    except KeyError:
        return error_message('Please enter state!', 400)

def return_url(recommended_song):
    origin_filename = "final_processed_data.csv"
    origin_df = pd.read_csv(origin_filename, index_col=0)
    recommended_song_title = recommended_song[:-4]
    matching_row = origin_df[origin_df['title'] == recommended_song_title]

    if not matching_row.empty:
        return matching_row['url'].iloc[0]
    else:
        return "해당 노래에 대한 URL을 찾을 수 없습니다."

def format_response(recommended_song, recommended_song_url, state, score=None):
    response = {
        "recommended_song": recommended_song,
        "recommended_song_url": recommended_song_url,
        "detected_emotion": state
    }
    if score:
        response["confidence"] = round(score, 2)
    return response

def ok(response):
    return json.dumps(response, ensure_ascii=False).encode('utf8')

def error_message(message, status):
    return jsonify({'response': message}), status

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)