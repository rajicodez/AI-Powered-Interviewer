AI Interview Analysis Tool
This Python-based Streamlit application analyzes interviews by capturing real-time video and audio. It uses DeepFace for emotion detection, SpeechRecognition for transcription, and a keyword-based heuristic for sentiment analysis to provide a final, aggregated score.

Features
Real-time Analysis: Captures and processes video and audio from your webcam and microphone in real time.

Emotion Detection: Uses the DeepFace library to analyze your facial expressions for dominant emotions (e.g., happy, sad, neutral).

Speech-to-Text: Transcribes spoken words into text using the Google Speech Recognition service.

Sentiment Analysis: Evaluates the sentiment of the transcribed text using a simple keyword-based approach.

Aggregated Score: Calculates a final interview score based on a combination of emotional and textual sentiment.

Interactive UI: A simple and clean web interface built with Streamlit to display live video feed and analysis results.

Getting Started
Follow these steps to get the application up and running on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation
Clone the repository to your local machine:

git clone https://github.com/rajicodez/AI-Powered-Interviewer

Install the required Python libraries using the requirements.txt file:

pip install -r requirements.txt

Usage
Ensure your webcam and microphone are working and not in use by other applications.

Run the Streamlit application from your terminal:

streamlit run app.py

Your default web browser will open a new tab with the application. Click the "Start Interview" button to begin the analysis.

Credits
This project was developed using the following open-source libraries:

Streamlit for the web application framework.

DeepFace for facial emotion detection.

OpenCV (cv2) for video capture.

sounddevice and scipy for audio recording.

SpeechRecognition for transcribing audio.
