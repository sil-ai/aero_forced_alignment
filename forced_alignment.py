import io
import modal
import pandas as pd
from pydub import AudioSegment
import boto3
from typing import List, Dict, Union

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "espeak-ng", "sox", "libsox-dev"])
    .pip_install([
        "torch",
        "transformers",
        "phonemizer",
        "pydub",
        "numpy",
        "pandas",
        "uroman",
        "inflect",
        "boto3",
        "datasets",
        "librosa",
        "soundfile",
        "fuzzywuzzy",
    ])
)

volume = modal.NetworkFileSystem.from_name("forced-alignment-cache-vol", create_if_missing=True)
CACHE_DIR = "/cache"
s3_access_credentials = modal.Secret.from_name("my-aws-secret")

app = modal.App("general-forced-alignment", image=image)

@app.cls(
    network_file_systems={CACHE_DIR: volume},
    gpu="t4",
    secrets=[modal.Secret.from_name("my-huggingface-secret"), s3_access_credentials],
    timeout=36000,
)

class ForcedAligner:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        model_name = "facebook/wav2vec2-base-960h"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_DIR).to(self.device)
        self.labels = self.processor.tokenizer.convert_ids_to_tokens(list(range(self.model.lm_head.out_features)))
        self.label_dict = {c: i for i, c in enumerate(self.labels)}

    @modal.method()
    def align_from_s3(self, 
                      audio_files: List[Dict[str, Union[str, bytes, List[Dict[str, str]]]]],
                      romanize: bool = False) -> Dict[str, str]:
        """
        
        Args:
            audio_files:[{
                "filename": ..., 
                "s3_path": ..., 
                "ref_text": [{"key": "...", "text": "..."}, ...]
        }]
            romanize: Whether to romanize the text
        
        Returns:
            Dictionary with timestamps for splitting source audio
        """
        import gc
        import numpy as np
        import os
        import re
        import torch
        import uroman as ur
        from dataclasses import dataclass
        from fuzzywuzzy import fuzz
        
        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float

            def __repr__(self):
                return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

        def audiosegment_to_waveform(audio_segment):
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2)).T  # shape (channels, samples)
            else:
                samples = samples[np.newaxis, :]  # shape (1, samples)
            samples /= (1 << (8 * audio_segment.sample_width - 1))  # normalize to [-1, 1]
            return torch.from_numpy(samples), audio_segment.frame_rate

        def get_trellis(emission, tokens, blank_id=0):
            num_frame, num_tokens = emission.size(0), len(tokens)
            trellis = torch.zeros((num_frame, num_tokens), device=self.device)
            trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
            trellis[0, 1:] = -float("inf")
            trellis[-num_tokens + 1 :, 0] = float("inf")

            for t in range(num_frame - 1):
                trellis[t + 1, 1:] = torch.maximum(
                    trellis[t, 1:] + emission[t, blank_id],
                    trellis[t, :-1] + emission[t, tokens[1:]],
                )
            return trellis

        def backtrack(trellis, emission, tokens, blank_id=0):
            t, j = trellis.size(0) - 1, trellis.size(1) - 1
            path = [Point(j, t, emission[t, blank_id].exp().item())]
            while j > 0:
                assert t > 0
                p_stay = emission[t - 1, blank_id]
                p_change = emission[t - 1, tokens[j]]
                stayed = trellis[t - 1, j] + p_stay
                changed = trellis[t - 1, j - 1] + p_change
                t -= 1
                if changed > stayed:
                    j -= 1
                prob = (p_change if changed > stayed else p_stay).exp().item()
                path.append(Point(j, t, prob))
            while t > 0:
                prob = emission[t - 1, blank_id].exp().item()
                path.append(Point(0, t - 1, prob))
                t -= 1
            return path[::-1]

        def merge_repeats(path, transcript):
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(
                    Segment(
                        transcript[path[i1].token_index],
                        path[i1].time_index,
                        path[i2 - 1].time_index + 1,
                        score,
                    )
                )
                i1 = i2
            return segments
        
        def process_audio_and_align(waveform, transcript_text):
            waveform = waveform.mean(dim=0)
            print("Transcript Text: " + transcript_text.strip().replace("\n", "|"))
            transcript_tokens = list(transcript_text.strip().replace("\n", "|"))

            token_ids = [self.label_dict[p] for p in transcript_tokens if p in self.label_dict]
            if not token_ids:
                raise ValueError("None of the tokens in transcript were found in the model label set.")

            try:
                input_values = self.processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
                with torch.no_grad():
                    emissions = self.model(input_values).logits
                emissions = torch.log_softmax(emissions, dim=-1)[0]

                trellis = get_trellis(emissions, token_ids)
                path = backtrack(trellis, emissions, token_ids)
                segments = merge_repeats(path, transcript_tokens)
                print("Segments: " + str(segments[:5]))
                break_segs = [seg for seg in segments if seg.label == "|"]

                return break_segs, trellis, emissions

            finally:
                # Clean up GPU memory
                del input_values
                if 'emissions' in locals():
                    del emissions
                if 'segments' in locals():
                    del segments
                gc.collect()
                torch.cuda.empty_cache()

        def extract_asr_segments(emissions, break_segs, ratio):
            asr_segments = []
            
            for i in range(len(break_segs) - 1):
                # Calculate frame boundaries for this segment
                start_frame = int(ratio * break_segs[i].end)
                if i + 1 < len(break_segs):
                    end_frame = int(ratio * break_segs[i + 1].start)
                else:
                    end_frame = emissions.size(0) * ratio
                
                # Convert to emission frame indices
                start_emission_frame = int(start_frame / ratio)
                end_emission_frame = int(end_frame / ratio)
                
                # Ensure bounds are within emissions tensor
                start_emission_frame = max(0, min(start_emission_frame, emissions.size(0) - 1))
                end_emission_frame = max(start_emission_frame + 1, min(end_emission_frame, emissions.size(0)))
                
                # Extract emissions for this segment
                segment_emissions = emissions[start_emission_frame:end_emission_frame]
                
                # Get predicted tokens for this segment
                predicted_ids = torch.argmax(segment_emissions, dim=-1)
                
                # Decode the segment
                segment_transcription = self.processor.batch_decode(predicted_ids.unsqueeze(0))[0]
                
                asr_segments.append(segment_transcription)
            
            return asr_segments

        output_records = []

        if isinstance(audio_files, dict):
            audio_files = [audio_files]
        elif isinstance(audio_files, str):
            raise ValueError("audio_files must be a list of dicts, not a string")
        
        for i in range(len(audio_files)):

            try:
                s3_path = audio_files[i]['s3_path']
                bucket, key = s3_path.replace("s3://", "").split("/", 1)
                if 'filename' in audio_files[i]:
                    filename = audio_files[i]['filename']
                else:
                    filename = key.split('/')[-1].split('.')[0]
                ref_text = audio_files[i]['ref_text']  # List of dicts with 'key' and 'text'

                text_lines = [v['text'] for v in ref_text]
                if 'key' in ref_text[0]:
                    text_refs = [v['key'] for v in ref_text]
                else:
                    text_refs = [filename + '_' + str(i) for i in range(len(text_lines))]
                chunk_text = '|' + '|'.join(text_lines).upper() + '|'

                if romanize:
                    uroman = ur.Uroman()
                    chunk_text = uroman.romanize_string(chunk_text)
                    print("Romanized")
                else:
                    print("Not Romanized")

                # Filter unwanted characters
                text_block = ''.join(c for c in chunk_text if c.isalpha() or c == '|')
                text_block = text_block.replace('ʼ', '')
                if not text_block.strip():
                    print(f"[WARNING] Empty or invalid text_block for file: {filename}")
                    continue

                s3 = boto3.client("s3")
                audio_obj = s3.get_object(Bucket=bucket, Key=key)
                ext = os.path.splitext(s3_path)[-1].lower().lstrip('.')
                section_audio = AudioSegment.from_file(io.BytesIO(audio_obj['Body'].read()), format=ext)

                waveform, sr = audiosegment_to_waveform(section_audio)
                if sr != 16000:
                    section_audio = section_audio.set_frame_rate(16000)
                    section_audio = section_audio.set_channels(1)
                    waveform, sr = audiosegment_to_waveform(section_audio)
                    sr = 16000
                break_segs, trellis, emissions = process_audio_and_align(waveform, text_block)
                ratio = waveform.size(1) / trellis.size(0)
                
                # Extract ASR segments
                asr_segments = extract_asr_segments(emissions, break_segs, ratio)
                
                for i in range(len(break_segs)-1):
                    output_filename = text_refs[i].replace(
                        ' ', '_').replace(':', '_').replace('\n', '')
                    output_key = f'output/{output_filename}.wav'
                    x0_frames = int(ratio * break_segs[i].end)
                    if i + 1 < len(break_segs):
                        x1_frames = int(ratio * break_segs[i+1].start)
                    else:
                        # For the last segment, set x1 to the end of the waveform
                        x1_frames = waveform.size(1)
                    x0_seconds = round(x0_frames / sr, 3)
                    x1_seconds = round(x1_frames / sr, 3)
                    
                    # Use the corresponding ASR segment
                    asr_transcription = asr_segments[i].lower() if i < len(asr_segments) else ""
                    match_score = fuzz.ratio(text_lines[i], asr_transcription)
                    
                    # Append record as dictionary
                    output_records.append({
                        'filename': output_key,
                        'text': text_lines[i],
                        'source_file': filename,
                        'start': x0_seconds,
                        'end': x1_seconds,
                        'asr_transcription': asr_transcription,
                        'match_score': match_score
                    })
                    
                if 'emissions' in locals():
                    del emissions
                if 'segments' in locals():
                    del segments
                if 'trellis' in locals():
                    del trellis
                if 'path' in locals():
                    del path
                if 'asr_segments' in locals():
                    del asr_segments
                if 'asr_segments' in locals():
                    del asr_segments
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                print(f"Skipping {filename} due to OOM: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

        return output_records
