import os
import pandas as pd
import numpy as np

def recommend(state, liked_songs):
    sim_filename = os.path.join(os.path.dirname(__file__), "..", "main", f"{state}_similarity_matrix.csv")
    # sim_filename = f"{state}_similarity_matrix.csv"
    sim_df = pd.read_csv(sim_filename, index_col=0)
    # data = pd.read_csv('final_processed_data.csv')  # 여기로 이동
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "main", "final_processed_data.csv"))  # 여기로 이동

    if not liked_songs:
        return filter_videos_corrected(data, tag=state, top_n=1)

    liked_songs_with_extension = [song + ".mp3" for song in liked_songs]
    valid_liked_songs = [song for song in liked_songs_with_extension if song in sim_df.index]

    if not valid_liked_songs:
        return filter_videos_corrected(data, tag=state, top_n=1)

    avg_similarities = {}
    for song in sim_df.columns:
        if song not in valid_liked_songs:
            similarities = [sim_df.loc[liked_song, song] for liked_song in valid_liked_songs]
            avg_similarities[song] = np.mean(similarities)

    if not avg_similarities:
        return filter_videos_corrected(data, tag=state, top_n=1)

    recommended_song = max(avg_similarities, key=avg_similarities.get)
    return recommended_song



def filter_videos_corrected(dataframe, tag, top_n=5):
    # 조회수 당 좋아요 비율
    dataframe['like_ratio'] = (dataframe['like_count'] / dataframe['view_count']) * 100
    emotion = dataframe[dataframe['tag'] == tag]
    # 내림차순 정렬
    sorted_videos = emotion.sort_values(by='like_ratio', ascending=False)
    # 추천 진행
    return sorted_videos.head(top_n)['title'].iloc[0] + '.mp3'