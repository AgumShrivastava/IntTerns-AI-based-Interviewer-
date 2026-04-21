import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import pyttsx3
import warnings
import os
import queue
import threading
import collections
import struct

warnings.filterwarnings("ignore")

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("System: webrtcvad not installed — falling back to fixed-duration recording.")
    print("        Install with:  pip install webrtcvad")


class AudioHandler:
    def __init__(self):
        print("System: Initializing Offline Audio Hardware (pyttsx3 + VAD)...")
        self.recognizer = sr.Recognizer()

        # TTS engine (unchanged)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)

        # ── VAD CONFIG ────────────────────────────────────────────────────────
        # Sample rate MUST be 8000, 16000, 32000, or 48000 for webrtcvad.
        self.SAMPLE_RATE = 16000
        self.FRAME_DURATION_MS = 30      # 30 ms frames (10, 20, or 30 ms supported)
        self.FRAME_SIZE = int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000)

        # How many consecutive silent frames before we stop recording.
        # 50 frames × 30 ms = 1 500 ms ≈ 1.5 seconds of silence → feels natural.
        self.SILENCE_FRAMES = 50

        # VAD aggressiveness: 0 (least aggressive) to 3 (most aggressive at
        # filtering non-speech).  2 works well for interviews.
        self.VAD_AGGRESSIVENESS = 2

        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)

        # Fallback duration (only used when webrtcvad is not installed)
        self.fallback_seconds = 7

    # ─────────────────────────────────────────────────────────────────────────
    # TTS  (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────
    def text_to_speech(self, text):
        """Converts text into offline audio bytes using pyttsx3."""
        try:
            temp_file = "temp_interviewer_voice.wav"
            self.engine.save_to_file(text, temp_file)
            self.engine.runAndWait()
            with open(temp_file, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # STT — NEW: VAD-driven streaming
    # ─────────────────────────────────────────────────────────────────────────
    def speech_to_text(self):
        """
        Records from the microphone using Voice Activity Detection.

        HOW IT WORKS:
          - Opens a continuous audio stream in small 30ms frames.
          - Each frame is checked by webrtcvad: is this speech or silence?
          - We keep recording as long as speech is detected.
          - Once we see SILENCE_FRAMES consecutive silent frames *after*
            some speech has occurred, we stop and transcribe.
          - Maximum recording time cap: 45 seconds (safety net).
        """
        if not VAD_AVAILABLE:
            return self._speech_to_text_fallback()

        print("System: Listening... (speak now, pause when done)")
        audio_frames = []
        silent_frame_count = 0
        speech_detected = False
        max_frames = int(45 * 1000 / self.FRAME_DURATION_MS)  # 45-second cap

        # Ring buffer of recent frames so we don't miss the start of speech
        ring_buffer = collections.deque(maxlen=10)

        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.FRAME_SIZE,
                dtype='int16',
                channels=1,
            ) as stream:
                frames_recorded = 0
                while frames_recorded < max_frames:
                    raw_frame, _ = stream.read(self.FRAME_SIZE)
                    # webrtcvad needs raw bytes
                    frame_bytes = bytes(raw_frame)

                    is_speech = self.vad.is_speech(frame_bytes, self.SAMPLE_RATE)

                    if not speech_detected:
                        # Pre-speech: accumulate in ring buffer
                        ring_buffer.append(frame_bytes)
                        if is_speech:
                            # Speech just started — flush ring buffer into audio
                            speech_detected = True
                            audio_frames.extend(ring_buffer)
                            ring_buffer.clear()
                    else:
                        audio_frames.append(frame_bytes)
                        if is_speech:
                            silent_frame_count = 0  # reset silence counter
                        else:
                            silent_frame_count += 1
                            if silent_frame_count >= self.SILENCE_FRAMES:
                                # Comfortable pause detected — done!
                                break

                    frames_recorded += 1

        except Exception as e:
            return f"Audio error: {str(e)}"

        if not audio_frames:
            return "Sorry, I couldn't detect any speech. Please try again."

        # Save and transcribe
        temp_file = 'temp_candidate_answer.wav'
        combined = b''.join(audio_frames)
        # Convert raw PCM bytes back to numpy-compatible format for soundfile
        import numpy as np
        audio_array = np.frombuffer(combined, dtype=np.int16)
        sf.write(temp_file, audio_array, self.SAMPLE_RATE, subtype='PCM_16')

        return self._transcribe_file(temp_file)

    def _transcribe_file(self, filepath):
        """Sends a WAV file to Google Speech Recognition."""
        try:
            with sr.AudioFile(filepath) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't catch that clearly. Could you rephrase?"
        except sr.RequestError:
            return "Speech service unavailable. Please check your internet connection."
        except Exception as e:
            return f"Transcription error: {str(e)}"

    def _speech_to_text_fallback(self, record_seconds=None):
        """
        Fallback: original fixed-duration recording.
        Only used when webrtcvad is not installed.
        """
        seconds = record_seconds or self.fallback_seconds
        fs = 44100
        print(f"System: Recording for {seconds} seconds (install webrtcvad for natural VAD)...")
        try:
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait()
            temp_file = 'temp_candidate_answer.wav'
            sf.write(temp_file, recording, fs)
            return self._transcribe_file(temp_file)
        except sr.UnknownValueError:
            return "Sorry, I couldn't catch that. Please try typing your answer."
        except sr.RequestError:
            return "Speech service unavailable."
        except Exception as e:
            return f"Audio error: {str(e)}"
