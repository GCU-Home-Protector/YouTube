import json
import pandas as pd
from flask import Flask, request, jsonify
from main import recommend_service

app = Flask(__name__)

liked_songs = []

@app.route('/request', methods=['POST'])
def recommend_songs():
    try:
        request_json = request.get_json(force=True)

        state = request_json['state']

        recommended_song = recommend_service.recommend(state, liked_songs)
        liked_songs.append(recommended_song[:-4])
        recommended_song_url = return_url(recommended_song)

        response = format_response(recommended_song, recommended_song_url)
        return ok(response)

    except KeyError:
        return error_message('Please enter question!', 400)

def return_url(recommended_song):
    origin_filename = "final_processed_data.csv"
    origin_df = pd.read_csv(origin_filename, index_col=0)

    # 확장자를 제거하고 추천 곡을 제목과 매칭
    recommended_song_title = recommended_song[:-4]
    matching_row = origin_df[origin_df['title'] == recommended_song_title]

    # URL 반환
    if not matching_row.empty:
        return matching_row['url'].iloc[0]
    else:
        return "해당 노래에 대한 URL을 찾을 수 없습니다."

def format_response(recommended_song, recommended_song_url):
    return {
        "recommended_song": recommended_song,
        "recommended_song_url": recommended_song_url
    }

def ok(response):
    return json.dumps(response, ensure_ascii=False).encode('utf8')

def error_message(message, status):
    return jsonify({
        'response': message
    }), status

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)