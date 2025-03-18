import os
import json
import whisperx
import subprocess
from transformers import pipeline

# Configurazione
VIDEO_FILE = "input.mp4"
AUDIO_FILE = "audio.wav"
TRANSCRIPTION_FILE = "transcription.txt"
CHAPTERS_FILE = "chapters.json"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cpu"

def extract_audio(video_file, audio_file):
    command = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_file]
    subprocess.run(command, check=True)
    print("Audio estratto con successo.")

def transcribe_audio(audio_file, transcription_file):
    model = whisperx.load_model("tiny", DEVICE, compute_type="int8")
    result = model.transcribe(audio_file)
    
    with open(transcription_file, "w") as f:
        f.write(json.dumps(result, indent=4))
    print("Trascrizione completata.")

def split_into_chapters(transcription_file, chapters_file):
    with open(transcription_file, "r") as f:
        transcription = json.load(f)
    text = " ".join([seg['text'] for seg in transcription['segments']])
    
    prompt = f"""Ecco la trascrizione di un video. Suddividila in capitoli con un titolo e un timestamp d'inizio per ciascuno:\n{text}"""
    llm = pipeline("text-generation", model=LLM_MODEL)
    response = llm(prompt, max_length=1024, truncation=True)[0]['generated_text']
    
    chapters = json.loads(response)  # Supponiamo che l'LLM restituisca JSON ben formattato
    with open(chapters_file, "w") as f:
        json.dump(chapters, f, indent=4)
    print("Capitoli generati.")

def split_video_by_chapters(video_file, chapters_file):
    with open(chapters_file, "r") as f:
        chapters = json.load(f)
    
    for i, chapter in enumerate(chapters):
        start_time = chapter["timestamp"]
        end_time = chapters[i+1]["timestamp"] if i+1 < len(chapters) else None
        output_file = f"chapter_{i+1}.mp4"
        
        command = ["ffmpeg", "-i", video_file, "-ss", str(start_time)]
        if end_time:
            command.extend(["-to", str(end_time)])
        command.extend(["-c", "copy", output_file])
        
        subprocess.run(command, check=True)
        print(f"Creato {output_file}")

if __name__ == "__main__":
    extract_audio(VIDEO_FILE, AUDIO_FILE)
    transcribe_audio(AUDIO_FILE, TRANSCRIPTION_FILE)
    split_into_chapters(TRANSCRIPTION_FILE, CHAPTERS_FILE)
    split_video_by_chapters(VIDEO_FILE, CHAPTERS_FILE)
    print("Processo completato!")
