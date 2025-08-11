import streamlit as st
import cv2
import time
import numpy as np
import os
import sounddevice as sd
from deepface import DeepFace
import speech_recognition as sr
from scipy.io.wavfile import write
import threading
import collections

# ---------------------------
# 1. Audio Recording Function
# ---------------------------

def record_audio(filename="audio.wav", duration=5, fs=44100):
    """
    Records audio for a specified duration and saves it to a WAV file.
    This function runs in a separate thread, so it should not directly
    update Streamlit's session state or UI elements.
    """
    print(f"Recording audio for {duration} seconds...") # Print to console for debugging
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait() # Wait until recording is finished
        write(filename, fs, recording) # Save as PCM WAV directly using scipy
        print("Audio recording complete.") # Print to console for debugging
    except Exception as e:
        print(f"Error during audio recording in thread: {e}")


# ---------------------------
# 2. Video Capture for Emotion Detection
# ---------------------------

def capture_video_frames(duration=10, interval=0.1):
    """
    Captures video frames from the webcam for a specified duration.
    Displays the live feed in a Streamlit placeholder and stores frames for analysis.
    """
    st.session_state.status_message.info(f"Capturing video for {duration} seconds...")
    cap = cv2.VideoCapture(0) # Open the default camera

    # Add a small delay to allow the camera to initialize
    time.sleep(0.5) 

    if not cap.isOpened():
        st.session_state.status_message.error("Error: Could not open video stream. Please ensure webcam is available and not in use by another application. Check camera permissions for your browser/system.")
        print("Error: Could not open video stream.")
        return []

    start_time = time.time()
    frames = []
    
    # Create a placeholder for the live video feed in Streamlit
    video_placeholder = st.empty()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            # Convert frame to RGB for Streamlit display (OpenCV reads BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True, caption="Live Video Feed")
            frames.append(frame.copy()) # Store BGR frame for DeepFace analysis
            time.sleep(interval)
        else:
            st.session_state.status_message.warning("Warning: Could not read frame from webcam. Stream may have ended unexpectedly.")
            print("Warning: Could not read frame from webcam.")
            break

    cap.release() # Release the camera resources
    video_placeholder.empty() # Clear the video display after capture is done
    st.session_state.status_message.success(f"Captured {len(frames)} video frames.")
    print(f"Captured {len(frames)} frames.")
    return frames

# ---------------------------
# 3. Emotion Detection
# ---------------------------

def analyze_emotions(frames):
    """
    Analyzes emotions from a list of video frames using DeepFace.
    Updates Streamlit status messages.
    """
    st.session_state.status_message.info("Analyzing emotions from captured video frames...")
    emotions = []
    if not frames:
        st.session_state.status_message.warning("No frames available to analyze for emotions.")
        return emotions

    for idx, frame in enumerate(frames):
        try:
            # DeepFace expects BGR image, which is what we stored.
            # prog_bar=False prevents DeepFace's internal progress bar from interfering with Streamlit's UI.
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, prog_bar=False)
            if analysis and analysis[0].get('dominant_emotion'):
                emotions.append(analysis[0]['dominant_emotion'])
        except Exception as e:
            # print(f"Error on frame {idx} during emotion analysis: {e}") # Keep this for debugging if needed
            pass # Continue processing other frames even if one fails
    st.session_state.status_message.success("Emotion analysis complete.")
    return emotions

# ---------------------------
# 4. Speech-to-Text
# ---------------------------

def transcribe_audio(filename="audio.wav"):
    """
    Transcribes audio from a WAV file using Google Speech Recognition.
    Updates Streamlit status messages.
    """
    st.session_state.status_message.info("Transcribing audio to text...")
    r = sr.Recognizer()
    try:
        if not os.path.exists(filename):
            st.session_state.status_message.error(f"Audio file not found: {filename}")
            return ""
        with sr.AudioFile(filename) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        st.session_state.status_message.success("Audio transcription complete.")
        return text
    except sr.UnknownValueError:
        st.session_state.status_message.warning("Google Speech Recognition could not understand the audio.")
        return ""
    except sr.RequestError as e:
        st.session_state.status_message.error(f"Could not request results from Google Speech Recognition service; check internet connection: {e}")
        return ""
    except Exception as e:
        st.session_state.status_message.error(f"An unexpected error occurred during transcription: {e}")
        return ""

# ---------------------------
# 5. Text Sentiment Heuristic
# ---------------------------

def simple_sentiment_analysis(text):
    """
    Performs a simple keyword-based sentiment analysis on text.
    Returns counts of positive/negative keywords and a normalized score.
    """
    positive_keywords = ["great", "happy", "excited", "love", "strong", "innovative", "successful", "good", "positive", "excellent", "amazing", "fantastic", "brilliant", "optimistic", "confident"]
    negative_keywords = ["bad", "sad", "difficult", "struggle", "failed", "challenge", "negative", "poor", "terrible", "awful", "problem", "worried", "nervous"]

    words = text.lower().split()
    pos = sum(word in positive_keywords for word in words)
    neg = sum(word in negative_keywords for word in words)

    # Calculate score: (pos - neg) / (pos + neg)
    # Normalize to 0-1 range: (score + 1) / 2
    score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
    normalized_score = round((score + 1) / 2, 2)
    return pos, neg, normalized_score

# ---------------------------
# Main Streamlit App Logic
# ---------------------------

def main():
    # Configure Streamlit page settings
    # Removed 'icon' argument as it might not be supported by all Streamlit versions
    st.set_page_config(page_title="Interview Analysis Tool", layout="centered")

    st.title("🗣️ Real-time Interview Analysis Tool")
    st.markdown("""
    This application captures your **video** and **audio** in real-time to analyze your emotions and the sentiment of your speech during an interview.
    Click the '🚀 Start Interview' button to begin a **15-second** recording.
    """)
    st.write("---")

    # Initialize session state variables to store status messages and analysis results
    # st.empty() creates a placeholder that can be updated dynamically
    if 'status_message' not in st.session_state:
        st.session_state.status_message = st.empty()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Button to start the interview process
    if st.button("🚀 Start Interview", help="Click to begin recording video and audio for analysis."):
        # Clear any previous analysis results and status messages
        st.session_state.analysis_results = None 
        st.session_state.status_message.empty() 

        # Use st.spinner to show a global loading indicator during the process
        with st.spinner("Preparing to record... Please ensure your webcam and microphone are accessible."):
            # Clean up the temporary audio file if it exists from a previous run
            if os.path.exists("audio.wav"):
                os.remove("audio.wav")

            recording_duration = 15 # Set the duration for recording in seconds

            # Start audio recording in a separate thread to run in parallel with video capture
            st.session_state.status_message.info(f"Starting audio recording for {recording_duration} seconds...")
            audio_thread = threading.Thread(target=record_audio, args=("audio.wav", recording_duration))
            audio_thread.start()
            
            # Video capture runs in the main thread to allow live display updates via st.image
            video_frames = capture_video_frames(duration=recording_duration, interval=0.1) 
            
            audio_thread.join() # Wait for the audio recording thread to complete
            st.session_state.status_message.success("Audio recording complete.")


            if not video_frames:
                st.session_state.status_message.error("No video frames were captured. Analysis cannot proceed. Please check your webcam and try again.")
                return # Exit if no frames were captured

        st.session_state.status_message.info("Recording complete. Starting analysis...")

        with st.spinner("Analyzing captured data... This may take a few moments."):
            # Perform emotion analysis on the captured video frames
            emotions = analyze_emotions(video_frames)
            emotion_summary = collections.Counter(emotions)

            # Transcribe the recorded audio and perform sentiment analysis on the text
            text = transcribe_audio("audio.wav")
            pos, neg, sentiment_normalized_score = simple_sentiment_analysis(text)

            # Calculate the final aggregated score
            # Emotion score: proportion of 'happy' or 'surprise' frames
            emotion_score = sum(1 for e in emotions if e in ["happy", "surprise"]) / len(emotions) if emotions else 0
            # Text score is already normalized by simple_sentiment_analysis
            text_score = sentiment_normalized_score
            
            # Final score is an average of emotion and text scores
            final_score = round((emotion_score + text_score) / 2, 2)

            # Store all results in Streamlit's session state to persist across reruns
            st.session_state.analysis_results = {
                "emotion_summary": emotion_summary,
                "transcribed_text": text,
                "positive_keywords": pos,
                "negative_keywords": neg,
                "sentiment_score": sentiment_normalized_score,
                "final_score": final_score
            }

            # Clean up the temporary audio file after analysis
            if os.path.exists("audio.wav"):
                os.remove("audio.wav")
        
        st.session_state.status_message.empty() # Clear the spinner/status message once analysis is done

    # Display analysis results if they are available in session state
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.subheader("📊 Analysis Results")
        st.markdown("---")

        st.markdown("#### 😊 Emotion Analysis Summary")
        if results["emotion_summary"]:
            for emo, count in results["emotion_summary"].items():
                st.write(f"- **{emo.capitalize()}**: {count} frames")
        else:
            st.write("No dominant emotions detected or frames captured for analysis.")

        st.markdown("#### 📝 Text Analysis")
        if results["transcribed_text"]:
            st.markdown(f"**Transcribed Text:** *{results['transcribed_text']}*")
            st.write(f"**Positive Keywords:** {results['positive_keywords']}")
            st.write(f"**Negative Keywords:** {results['negative_keywords']}")
            st.write(f"**Sentiment Score (Normalized):** {results['sentiment_score']}")
        else:
            st.write("No speech was transcribed.")

        st.markdown("---")
        st.markdown(f"### ✨ Final Interview Score: **{results['final_score']}**")
        st.markdown("---")

    st.write("---")
    st.info("Note: This tool requires access to your webcam and microphone. Please ensure they are enabled and not in use by other applications.")
    st.caption("Developed using Streamlit, OpenCV, DeepFace, and SpeechRecognition.")

if __name__ == "__main__":
    main()
