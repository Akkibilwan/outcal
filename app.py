# requirements.txt
# streamlit
# pandas
# numpy
# requests
# plotly

# .streamlit/secrets.toml
# YT_API_KEY = "your_youtube_api_key_here"

import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import re
from datetime import timedelta

st.set_page_config(page_title="Keyword Video Outlier Analyzer", layout="wide")

# API Key Setup
yt_api_key = st.secrets.get("YT_API_KEY")
if not yt_api_key:
    st.error("YouTube API key not found in st.secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

# --- Helper Functions ---
def parse_duration(duration_str):
    hours = re.search(r'(\d+)H', duration_str)
    minutes = re.search(r'(\d+)M', duration_str)
    seconds = re.search(r'(\d+)S', duration_str)
    return sum([int(hours.group(1)) * 3600 if hours else 0,
                int(minutes.group(1)) * 60 if minutes else 0,
                int(seconds.group(1)) if seconds else 0])

def fetch_search_results(keyword, published_after):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=25&q={keyword}&publishedAfter={published_after}&key={yt_api_key}"
    res = requests.get(url).json()
    results = []
    for item in res.get("items", []):
        if 'videoId' in item['id'] and 'channelId' in item['snippet']:
            video_id = item['id']['videoId']
            channel_id = item['snippet']['channelId']
            results.append((video_id, channel_id))
    return results

def fetch_single_video(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={yt_api_key}"
    res = requests.get(url).json()
    if not res.get('items'): return None
    data = res['items'][0]
    try:
        duration = parse_duration(data['contentDetails']['duration'])
        return {
            'videoId': video_id,
            'channelId': data['snippet']['channelId'],
            'title': data['snippet']['title'],
            'publishedAt': data['snippet']['publishedAt'],
            'viewCount': int(data['statistics'].get('viewCount', 0)),
            'duration': duration,
            'isShort': duration <= 60
        }
    except:
        return None

def fetch_channel_videos(channel_id, max_videos):
    uploads_url = f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails&id={channel_id}&key={yt_api_key}"
    res = requests.get(uploads_url).json()
    if not res.get('items'): return []
    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    videos = []
    next_page_token = ""
    while len(videos) < max_videos:
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=contentDetails&maxResults=50&playlistId={playlist_id}&key={yt_api_key}"
        if next_page_token:
            url += f"&pageToken={next_page_token}"
        items = requests.get(url).json()
        for item in items.get('items', []):
            if 'contentDetails' in item and 'videoId' in item['contentDetails']:
                videos.append(item['contentDetails']['videoId'])
                if len(videos) >= max_videos:
                    break
        next_page_token = items.get('nextPageToken')
        if not next_page_token: break
    return videos

def fetch_video_batch(video_ids):
    chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
    video_data = {}
    for chunk in chunks:
        ids = ','.join(chunk)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet,contentDetails&id={ids}&key={yt_api_key}"
        res = requests.get(url).json()
        for item in res.get('items', []):
            try:
                vid = item.get('id')
                if not vid or 'contentDetails' not in item or 'duration' not in item['contentDetails']:
                    continue
                duration = parse_duration(item['contentDetails']['duration'])
                published = item['snippet']['publishedAt']
                video_data[vid] = {
                    'viewCount': int(item['statistics'].get('viewCount', 0)),
                    'duration': duration,
                    'publishedAt': published,
                    'isShort': duration <= 60
                }
            except Exception as e:
                continue
    return video_data

def calculate_outlier_score(current_views, channel_avg):
    return round(current_views / channel_avg, 2) if channel_avg > 0 else 0

def simulate_average(vids, video_data, is_short):
    now = datetime.datetime.now(datetime.timezone.utc)
    views = []
    for vid in vids:
        data = video_data.get(vid)
        if data and data['isShort'] == is_short:
            published = datetime.datetime.fromisoformat(data['publishedAt'].replace('Z', '+00:00'))
            age_days = max(1, (now - published).days)
            avg = data['viewCount'] / age_days
            views.append(avg)
    return np.mean(views) if views else 0

# --- UI ---
st.title("Keyword-Based YouTube Outlier Finder")

keyword = st.text_input("Search Keyword")
timeframe = st.selectbox("Uploaded Within", [
    "24 hours", "48 hours", "1 week", "1 month", "3 months", "6 months", "1 year", "Lifetime"
])
type_filter = st.radio("Content Type", ["Both", "Shorts", "Videos"])

if st.button("Search"):
    now = datetime.datetime.utcnow()
    delta_map = {
        "24 hours": timedelta(days=1),
        "48 hours": timedelta(days=2),
        "1 week": timedelta(weeks=1),
        "1 month": timedelta(days=30),
        "3 months": timedelta(days=90),
        "6 months": timedelta(days=180),
        "1 year": timedelta(days=365),
        "Lifetime": timedelta(days=4000),
    }
    after = now - delta_map[timeframe]
    after_str = after.strftime('%Y-%m-%dT%H:%M:%SZ')

    results = fetch_search_results(keyword, after_str)
    outlier_rows = []
    for video_id, channel_id in results:
        vid_info = fetch_single_video(video_id)
        if not vid_info:
            continue

        if type_filter == "Shorts" and not vid_info.get('isShort', False):
            continue
        elif type_filter == "Videos" and vid_info.get('isShort', False):
            continue

        ch_videos = fetch_channel_videos(channel_id, 50)
        ch_video_data = fetch_video_batch([v for v in ch_videos if v != video_id])
        ch_avg = simulate_average(ch_videos, ch_video_data, vid_info['isShort'])

        outlier_score = calculate_outlier_score(vid_info['viewCount'], ch_avg)

        outlier_rows.append({
            "Title": vid_info['title'],
            "Views": vid_info['viewCount'],
            "Avg Views/Day": int(ch_avg),
            "Outlier Score": outlier_score
        })

    if outlier_rows:
        st.subheader("Results")
        df = pd.DataFrame(outlier_rows)
        st.dataframe(df.sort_values("Outlier Score", ascending=False))
    else:
        st.info("No matching videos found or not enough channel data.")
