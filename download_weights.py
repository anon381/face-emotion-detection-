import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == '__main__':
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Haar Cascade
    cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        download_file(cascade_url, cascade_path)
    else:
        print(f"{cascade_path} already exists.")
        
    # FER2013 Model
    model_url = 'https://github.com/GSNCodes/Emotion-Detection-FER2013/raw/master/emotion_detection_model.h5'
    model_path = os.path.join(model_dir, 'emotion_model.h5')
    if not os.path.exists(model_path):
        download_file(model_url, model_path)
    else:
        print(f"{model_path} already exists.")
