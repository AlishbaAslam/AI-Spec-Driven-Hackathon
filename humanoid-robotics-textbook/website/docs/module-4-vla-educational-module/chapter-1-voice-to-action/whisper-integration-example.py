"""
Example implementation of Whisper integration for voice-to-action systems in robotics.
This demonstrates how to use OpenAI's Whisper model to transcribe voice commands
and map them to robotic actions.
"""

import whisper
import numpy as np
import torch
import pyaudio
import wave
import json
import time

class WhisperVoiceCommandProcessor:
    """
    A processor that uses OpenAI Whisper to convert voice commands to robotic actions.
    """

    def __init__(self, model_size="base"):
        """
        Initialize the voice command processor.

        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")

        # Define command mappings
        self.command_mappings = {
            "move forward": "MOVE_FORWARD:0.5",  # Move forward 0.5 meters
            "move backward": "MOVE_BACKWARD:0.5",  # Move backward 0.5 meters
            "turn left": "TURN_LEFT:90",  # Turn left 90 degrees
            "turn right": "TURN_RIGHT:90",  # Turn right 90 degrees
            "stop": "STOP:",
            "pick up object": "PICK_OBJECT:",
            "put down object": "PLACE_OBJECT:",
            "raise arm": "RAISE_ARM:",
            "lower arm": "LOWER_ARM:",
        }

        # Initialize audio recording parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best with 16kHz audio
        self.chunk = 1024
        self.record_seconds = 5

    def record_audio(self, filename="command.wav"):
        """
        Record audio from the microphone and save to file.

        Args:
            filename (str): Name of the file to save the recording
        """
        audio = pyaudio.PyAudio()

        # Start recording
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print("Recording... Speak now!")
        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished!")

        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio to a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path (str): Path to the audio file to transcribe

        Returns:
            str: Transcribed text
        """
        result = self.model.transcribe(audio_path)
        return result["text"]

    def map_command(self, transcribed_text):
        """
        Map transcribed text to a robotic command.

        Args:
            transcribed_text (str): Text transcribed from speech

        Returns:
            str: Robotic command in the format ACTION:PARAMETER
        """
        text_lower = transcribed_text.lower().strip()

        # Check for exact matches first
        for command_phrase, robot_command in self.command_mappings.items():
            if command_phrase in text_lower:
                return robot_command

        # If no exact match, try to find closest match
        best_match = None
        best_score = 0

        for command_phrase in self.command_mappings.keys():
            score = self.calculate_similarity(text_lower, command_phrase)
            if score > best_score:
                best_score = score
                best_match = command_phrase

        # If we found a match with high similarity (> 0.7), use it
        if best_score > 0.7 and best_match:
            return self.command_mappings[best_match]

        return "UNKNOWN:"  # Return unknown if no good match found

    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two texts using a simple method.
        This is a basic implementation - in practice, you might want to use
        more sophisticated NLP techniques.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def process_voice_command(self, audio_path=None):
        """
        Complete pipeline: record audio, transcribe, and map to command.

        Args:
            audio_path (str, optional): Path to existing audio file.
                                      If None, record new audio.

        Returns:
            dict: Dictionary containing transcription and mapped command
        """
        if audio_path is None:
            # Record new audio
            audio_file = "temp_command.wav"
            self.record_audio(audio_file)
        else:
            audio_file = audio_path

        # Transcribe the audio
        transcribed_text = self.transcribe_audio(audio_file)

        # Map to robotic command
        robot_command = self.map_command(transcribed_text)

        # Clean up temporary file if needed
        if audio_path is None:
            import os
            os.remove(audio_file)

        return {
            "transcribed_text": transcribed_text,
            "robot_command": robot_command,
            "confidence": 0.9 if robot_command != "UNKNOWN:" else 0.1
        }

def main():
    """
    Main function to demonstrate the Whisper voice command processor.
    """
    print("Initializing Whisper Voice Command Processor...")

    # Initialize the processor
    processor = WhisperVoiceCommandProcessor(model_size="base")

    print("\nWhisper Voice Command Processor initialized!")
    print("Available commands:")
    for cmd_phrase in processor.command_mappings.keys():
        print(f"  - {cmd_phrase}")

    while True:
        print("\nOptions:")
        print("1. Record and process new voice command")
        print("2. Process existing audio file")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            print("\nRecording voice command...")
            result = processor.process_voice_command()

            print(f"\nTranscribed: {result['transcribed_text']}")
            print(f"Robot Command: {result['robot_command']}")
            print(f"Confidence: {result['confidence']:.2f}")

        elif choice == "2":
            audio_file = input("Enter path to audio file: ").strip()
            try:
                result = processor.process_voice_command(audio_file)

                print(f"\nTranscribed: {result['transcribed_text']}")
                print(f"Robot Command: {result['robot_command']}")
                print(f"Confidence: {result['confidence']:.2f}")

            except Exception as e:
                print(f"Error processing audio file: {e}")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()