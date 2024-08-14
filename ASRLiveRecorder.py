import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor, \
    WhisperForConditionalGeneration
from gtts import gTTS
import pygame
import io
from datasets import load_dataset, Dataset, DatasetDict
import torchaudio
import pydub
import librosa
import sounddevice as sd
import queue
import webrtcvad
import numpy as np
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

data = {
    "train": [
        {"source": "مرحبا، كيف حالك؟", "target": "Hallo, wie geht es Ihnen?"},
        {"source": "ما اسمك؟", "target": "Wie heißt du?"},
        {"source": "أين تسكن؟", "target": "Wo wohnst du?"},
        {"source": "ما هو عملك؟", "target": "Was ist dein Beruf?"},
        {"source": "أحب البرمجة.", "target": "Ich liebe Programmierung."}
    ],
    "validation": [
        {"source": "مرحبا", "target": "Hallo"},
        {"source": "شكرا", "target": "Danke"}
    ]
}


def convert_to_wav(input_file, output_file):
    try:
        audio = pydub.AudioSegment.from_file(input_file, format="m4a")
        audio.export(output_file, format="wav")
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting {input_file} to {output_file}: {e}")
        raise


def load_processor_ASR():
    try:
        processor = WhisperProcessor.from_pretrained("ASR/openai/whisper-large-v3", local_files_only=True)
        print("Loaded from local machine")
        return processor
    except Exception as e:
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        processor.save_pretrained(f'ASR/openai/whisper-large-v3')
        print("Downloaded and saved")
        return processor


def load_model_ASR():
    try:
        model = WhisperForConditionalGeneration.from_pretrained("ASR/openai/whisper-large-v3",
                                                                local_files_only=True).to(device)
        print("Loaded from local machine")
        return model
    except Exception as e:
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)
        model.save_pretrained(f'ASR/openai/whisper-large-v3')
        print("Downloaded and saved")
        return model


def load_audio(file_path):
    audio, rate = librosa.load(file_path, sr=16000, mono=True)
    audio = torch.tensor(audio).unsqueeze(0)  # Convert to torch tensor and add channel dimension
    return audio


def split_audio(audio, segment_length):
    num_samples = audio.shape[1]
    segments = []
    for start in range(0, num_samples, segment_length):
        end = min(start + segment_length, num_samples)
        segments.append(audio[:, start:end])
    return segments


def transcribe_audio_segment(segment, processor, model):
    inputs = processor(segment.squeeze(), return_tensors="pt", sampling_rate=16000).to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features, max_length=500)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]


def transcribe_long_audio(file_path, processor_ASR, model_ASR, segment_length_ms=30000):
    audio = load_audio(file_path)
    segment_length_samples = int(segment_length_ms / 1000 * 16000)
    segments = split_audio(audio, segment_length_samples)

    batch_size = 4  # Define a batch size for processing segments
    transcriptions = []
    for i in range(0, len(segments), batch_size):
        batch_segments = segments[i:i + batch_size]
        batch_transcriptions = [transcribe_audio_segment(segment, processor_ASR, model_ASR) for segment in
                                batch_segments]
        transcriptions.extend(batch_transcriptions)

    return " ".join(transcriptions)


def fineTuning_Model(dataset, model, tokenizer, src_lang, tgt_lang):
    train_dataset = Dataset.from_list(dataset["train"])
    val_dataset = Dataset.from_list(dataset["validation"])

    def preprocess_function(examples):
        inputs = [ex for ex in examples['source']]
        targets = [ex for ex in examples['target']]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, padding=True)
        return model_inputs

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(f'./fine-tuned-marianmt-{src_lang}-{tgt_lang}')
    tokenizer.save_pretrained(f'./fine-tuned-marianmt-{src_lang}-{tgt_lang}')


def translate_text(text, premodel, tokenizer, src_lang, tgt_lang):
    translated = premodel.generate(**tokenizer(text, return_tensors="pt", padding=True).to(device))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    return tgt_text


def load_model(src_lang, tgt_lang):
    try:
        model_name = f"./fine-tuned-marianmt-{src_lang}-{tgt_lang}"
        model = MarianMTModel.from_pretrained(model_name).to(device)
        print("Loaded from local machine")
        return model
    except Exception as e:
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        model = MarianMTModel.from_pretrained(model_name).to(device)
        model.save_pretrained(f'./fine-tuned-marianmt-{src_lang}-{tgt_lang}')
        print("Downloaded and saved")
        return model


def load_tokenizer(src_lang, tgt_lang):
    try:
        model_name = f"./fine-tuned-marianmt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        print("Loaded from local machine")
        return tokenizer
    except Exception as e:
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(f'./fine-tuned-marianmt-{src_lang}-{tgt_lang}')
        print("Downloaded and saved")
        return tokenizer


def record_and_process_audio(model_ASR, processor_ASR, premodel, tokenizer, source_language, target_language):
    q = queue.Queue()
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Set aggressiveness level of VAD (0-3), higher values mean more aggressive

    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        q.put(indata.copy())

    def is_speech(audio_data):
        """Check if the audio data contains speech."""
        try:
            audio_data = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
            if len(audio_data) != 160:  # Ensure the frame length is 10ms at 16kHz
                raise ValueError(f"Unexpected frame length: {len(audio_data)}")
            return vad.is_speech(audio_data.tobytes(), 16000)
        except Exception as e:
            print(f"Error in VAD: {e}")
            return False

    with sd.InputStream(samplerate=16000, channels=1, callback=callback):
        print("Recording... Press Ctrl+C to stop.")
        try:
            transcription = ""
            translated_text = ""
            frame_buffer = np.zeros((0,), dtype=np.int16)
            while True:
                audio_data = q.get()
                if audio_data is None or len(audio_data) == 0:
                    # print("Received empty audio data.")
                    continue

                # print(f"Received audio data with shape: {audio_data.shape}")

                # Flatten the audio data and append to the buffer
                frame_buffer = np.concatenate((frame_buffer, (audio_data * 32767).astype(np.int16).flatten()))

                # Process in chunks of 10ms (160 samples at 16kHz)
                while len(frame_buffer) >= 160:
                    frame = frame_buffer[:160]
                    frame_buffer = frame_buffer[160:]

                    if is_speech(frame):
                        # print("Speech detected, processing audio...")
                        try:
                            audio_tensor = torch.tensor(frame).float().unsqueeze(0).unsqueeze(0).to(device)
                            # print(f"Audio tensor shape: {audio_tensor.shape}")

                            transcription += transcribe_audio_segment(audio_tensor, processor_ASR, model_ASR)
                            print("Transcription:", transcription)
                            translated_text = translate_text(transcription, premodel, tokenizer, source_language,
                                                             target_language)
                            print("Translation:", translated_text)
                        except Exception as e:
                            print(f"Error during transcription or translation: {e}")
                       # print("No speech detected.")
        except KeyboardInterrupt:
            print("Recording stopped")
        except Exception as e:
            print(f"Error: {e}")


source_language = 'en'
target_language = 'de'
To_translate_text = "ترجم هذا النص من فضلك"
file_path = "videoplayback.m4a"


def mainFunction(state, source, target, To_translate_text, data, file_path):
    model = load_model(source_language, target_language)
    tokenizers = load_tokenizer(source_language, target_language)

    if 1 == state:
        translated_text = translate_text(To_translate_text, model, tokenizers, source_language, target_language)
        print(translated_text)

    if 2 == state:
        model_ASR = load_model_ASR()
        processor_ASR = load_processor_ASR()

        wav_file_path = file_path.replace('.m4a', '.wav')
        convert_to_wav(file_path, wav_file_path)

        transcription = transcribe_long_audio(wav_file_path, processor_ASR, model_ASR)
        print("Transcription:", transcription)
        translated_text = translate_text(transcription, model, tokenizers, source_language, target_language)
        print(translated_text)

    if 3 == state:
        fineTuning_Model(data, model, tokenizers, source_language, target_language)

    if 4 == state:
        model_ASR = load_model_ASR()
        processor_ASR = load_processor_ASR()

        record_and_process_audio(model_ASR, processor_ASR, model, tokenizers, source_language, target_language)


mainFunction(4, source_language, target_language, To_translate_text, data, file_path)
