import librosa
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

DL_PATH = "data/audio"
QUALITY_KBPS = 128
DATASET = "data/dataset_500.csv"

VALID_TAGS_CAL500 = [
    "Emotion-Angry_/_Agressive",
    "Emotion-Bizarre_/_Weird",
    "Emotion-Powerful_/_Strong",
    "Emotion-Loving_/_Romantic",
    "Emotion-Arousing_/_Awakening",
    "Emotion-Exciting_/_Thrilling",
    "Emotion-Cheerful_/_Festive",
    "Emotion-Happy",
    "Emotion-Positive_/_Optimistic",
    "Emotion-Light_/_Playful",
    "Emotion-Touching_/_Loving",
    "Emotion-Sad",
    "Emotion-Emotional_/_Passionate",
    "Emotion-Calming_/_Soothing",
    "Emotion-Laid-back_/_Mellow",
    "Emotion-Carefree_/_Lighthearted",
    "Emotion-Pleasant_/_Comfortable",
    "Emotion-Tender_/_Soft",
]


def map_emotion_cal500(tag):
    mapping = {
        # Angry emotions
        "Emotion-Angry_/_Agressive": "Angry",
        "Emotion-Bizarre_/_Weird": "Angry",
        # Happy emotions
        "Emotion-Powerful_/_Strong": "Happy",
        "Emotion-Loving_/_Romantic": "Happy",
        "Emotion-Arousing_/_Awakening": "Happy",
        "Emotion-Exciting_/_Thrilling": "Happy",
        "Emotion-Cheerful_/_Festive": "Happy",
        "Emotion-Happy": "Happy",
        "Emotion-Positive_/_Optimistic": "Happy",
        "Emotion-Light_/_Playful": "Happy",
        # Sad emotions
        "Emotion-Touching_/_Loving": "Sad",
        "Emotion-Sad": "Sad",
        "Emotion-Emotional_/_Passionate": "Sad",
        # Relaxed emotions
        "Emotion-Calming_/_Soothing": "Relaxed",
        "Emotion-Laid-back_/_Mellow": "Relaxed",
        "Emotion-Carefree_/_Lighthearted": "Relaxed",
        "Emotion-Pleasant_/_Comfortable": "Relaxed",
        "Emotion-Tender_/_Soft": "Relaxed",
    }

    return mapping.get(tag, None)


def map_emotion(valence, arousal):
    if not (0 <= valence <= 1) or not (0 <= arousal <= 1):
        return "Invalid input: coordinates should be between 0 and 1."

    if valence >= 0.5 and arousal >= 0.5:
        return "Happy"
    elif valence < 0.5 and arousal >= 0.5:
        return "Angry"
    elif valence < 0.5 and arousal < 0.5:
        return "Sad"
    elif valence >= 0.5 and arousal < 0.5:
        return "Relaxed"


def get_features(audio_path: str, N_CHROMA=12, N_MFCC=20):
    """Extract features from audio file path"""

    # y = audio time series, sr = sampling rate
    y, sr = librosa.load(audio_path)

    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    tempo = librosa.feature.tempo(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    result = {
        "audio_path": audio_path.split("/")[-1].replace(".mp3", ""),
        "rms": np.mean(rms),
        "spectral_centroid": np.mean(spec_cent),
        "spectral_bandwidth": np.mean(spec_bw),
        "rolloff": np.mean(rolloff),
        "zero_crossing_rate": np.mean(zcr),
        "tempo": np.mean(tempo),
        "tonnetz": np.mean(tonnetz),
    }
    # result.update(
    #     {f"chroma_stft_{i}": np.mean(value) for i, value in enumerate(chroma_stft)}
    # )
    result.update({f"mfcc_{i}": np.mean(value) for i, value in enumerate(mfcc)})

    return result


def read_files_in_folder(folder_path):
    file_contents = []
    file_names = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()
                file_contents.append(content)
                file_names.append(filename.replace(".txt", ""))
    return pd.Series(file_contents, index=file_names)


def load_lyrics_from_dir(LYRICS_DIR):
    lyrics = read_files_in_folder(LYRICS_DIR)
    lyrics.index = lyrics.index.str.replace(".lrc", "").astype(int)
    lyrics.sort_index(inplace=True)
    lyrics = lyrics.to_frame()
    lyrics.columns = ["lyrics"]

    return lyrics


def load_annotations_from_path(ANNOTATIONS_PATH):
    annotations = pd.read_csv(ANNOTATIONS_PATH, index_col=0)
    annotations.columns = ["arousal", "valence"]

    # Scale values from -1 to 1
    # annotations = annotations.apply(lambda x: x * 2 - 1, axis=0)
    annotations["Emotion"] = annotations.apply(lambda x: map_emotion(*x), axis=1)
    annotations.index.name = None

    return annotations


def get_audio_paths(AUDIO_BASE_DIR):
    AUDIO_FILE_PATHS = [
        os.path.join(AUDIO_BASE_DIR, f)
        for f in os.listdir(AUDIO_BASE_DIR)
        if f.endswith(".mp3")
    ]
    AUDIO_FILE_PATHS.sort()
    AUDIO_FILE_PATHS = pd.Series(AUDIO_FILE_PATHS)

    return AUDIO_FILE_PATHS


def get_features_from_paths(AUDIO_FILE_PATHS):

    def process_audio(audio_path):
        features = get_features(audio_path)
        return features

    executor = ThreadPoolExecutor(max_workers=8)
    results = []
    with tqdm(total=len(AUDIO_FILE_PATHS), desc="Processing audio files") as pbar:
        for result in executor.map(process_audio, AUDIO_FILE_PATHS):
            results.append(result)
            pbar.update(1)
    executor.shutdown()

    df = pd.DataFrame.from_records(results, index="audio_path")
    df.sort_index(inplace=True)

    return df


def download_mp3(song_name, index):
    song_name = song_name.replace("'", "")
    os.system("clear")
    os.system(
        "yt-dlp -x --audio-format mp3" + " "
        f"ytsearch1:'{song_name}'" + " "
        f"--audio-quality {QUALITY_KBPS}K" + " "
        f"-o 'data/audio/{index}.%(ext)s'"
    )
