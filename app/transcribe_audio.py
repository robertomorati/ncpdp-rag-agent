from faster_whisper import WhisperModel
from pathlib import Path


AUDIO_PATH = "data/raw/ncpdp_audio.mp3"
OUTPUT_PATH = "data/processed/audio_text.txt"


def transcribe_audio(audio_path: str) -> str:
    print("Loading Whisper model...")
    model = WhisperModel("base", compute_type="int8")  # fast + light

    print(f"Transcribing: {audio_path}")
    segments, _ = model.transcribe(audio_path)

    full_text = []
    for segment in segments:
        full_text.append(segment.text)

    return " ".join(full_text)


def save_text(text: str, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    text = transcribe_audio(AUDIO_PATH)
    save_text(text, OUTPUT_PATH)

    print(f"Saved transcription to {OUTPUT_PATH}")
    print(f"Length: {len(text)} characters")


if __name__ == "__main__":
    main()