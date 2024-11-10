from flask import Flask, render_template, request, url_for
import os, cv2
from collections import defaultdict
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from textblob import TextBlob
from googletrans import Translator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['STATIC_FOLDER'] = './static/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get patient name, language, and both videos from the form
    patient_name = request.form['patient_name']
    language_code = request.form['language']  # Language code for transcription
    before_video = request.files['before_video']
    after_video = request.files['after_video']
    
    # Save both videos to the upload folder
    before_path = os.path.join(app.config['UPLOAD_FOLDER'], 'before_' + before_video.filename)
    after_path = os.path.join(app.config['UPLOAD_FOLDER'], 'after_' + after_video.filename)
    before_video.save(before_path)
    after_video.save(after_path)

    # Process both videos and get analysis results
    before_result = process_video(before_path, patient_name, 'Before Treatment', language_code)
    after_result = process_video(after_path, patient_name, 'After Treatment', language_code)

    # Render the results with comparison on the same page
    return render_template('index.html', before_result=before_result, after_result=after_result)

def process_video(video_path, patient_name, stage, language_code):
    # Initialize variables for analysis
    cap = cv2.VideoCapture(video_path)
    emotion_counts = defaultdict(int)
    recognizer = sr.Recognizer()
    translator = Translator()  # Initialize the Google Translate API
    overall_sentiment = {'polarity': 0, 'subjectivity': 0}
    total_frames_analyzed = 0
    frame_skip = 10
    image_path = os.path.join(app.config['STATIC_FOLDER'], f'{stage.lower().replace(" ", "_")}_patient_image.jpg')

    # Extract the first frame with a face and save it as a patient image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            cv2.imwrite(image_path, frame)
            break
    cap.release()

    # Extract audio and perform sentiment analysis
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{stage.lower().replace(" ", "_")}_audio.wav')
    if audio:
        audio.write_audiofile(audio_path)
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                # Transcribe audio with the specified language
                text = recognizer.recognize_google(audio_data, language=language_code)
                
                # Translate text if it's not English (assuming English sentiment analysis)
                if language_code != "en":
                    translated_text = translator.translate(text, src=language_code, dest="en").text
                else:
                    translated_text = text
                
                # Perform sentiment analysis on the translated text
                sentiment = TextBlob(translated_text).sentiment
                overall_sentiment['polarity'] = sentiment.polarity
                overall_sentiment['subjectivity'] = sentiment.subjectivity
            except Exception as e:
                print(f"Error processing audio for {stage}: {e}")

    # Perform emotion detection on video frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_counts[dominant_emotion] += 1
                total_frames_analyzed += 1
            except Exception as e:
                print(f"Error processing frame {frame_count} for {stage}: {e}")
        frame_count += 1
    cap.release()

    # Calculate emotion percentages
    emotion_percentages = {
        emotion: (count / total_frames_analyzed) * 100 for emotion, count in emotion_counts.items()
    }
    dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)

    # Combine all results into a dictionary
    result = {
        'patient_name': patient_name,
        'stage': stage,
        'image_path': url_for('static', filename=image_path.split('/')[-1]),
        'dominant_emotion': dominant_emotion,
        'emotion_percentages': emotion_percentages,
        'overall_sentiment': overall_sentiment
    }
    return result

if __name__ == '__main__':
    app.run(debug=True)
