import io
import modal
import pandas as pd
from pydub import AudioSegment
import fsspec
from typing import List, Dict, Optional, Union

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
    ])
)

volume = modal.NetworkFileSystem.from_name("forced-alignment-cache-vol", create_if_missing=True)
CACHE_DIR = "/cache"
s3_bucket_name = "forcedalignment"
s3_access_credentials = modal.Secret.from_name("my-aws-secret")

app = modal.App("forced-alignment", image=image)

@app.cls(
    network_file_systems={CACHE_DIR: volume},
    gpu="t4",
    secrets=[modal.Secret.from_name("my-huggingface-secret"), s3_access_credentials],
    timeout=36000,
)

class ForcedAligner:
    @modal.method()
    def load_model(self, model_name="facebook/wav2vec2-base-960h"):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.labels = self.processor.tokenizer.convert_ids_to_tokens(list(range(self.model.lm_head.out_features)))
        self.label_dict = {c: i for i, c in enumerate(self.labels)}

    @modal.method()
    def align_from_bytes(self, 
                        audio_files: List[Dict[str, Union[bytes, str]]], 
                        text_lines: List[str],
                        ref_lines: List[str],
                        romanize: bool = False) -> pd.DataFrame:
        """
        Align audio files with text using bytes data instead of S3 links.
        
        Args:
            audio_files: List of dicts with 'data' (bytes) and 'filename' (str) keys
            text_lines: List of text lines corresponding to the audio
            ref_lines: List of reference lines (vref or similar)
            romanize: Whether to romanize the text
        
        Returns:
            DataFrame with timestamps for splitting source audio
        """
        import gc
        import numpy as np
        import re
        import torch
        import uroman as ur
        from dataclasses import dataclass
        
        print("starting alignment from bytes")

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

                return break_segs, trellis

            finally:
                # Clean up GPU memory
                del input_values
                if 'emissions' in locals():
                    del emissions
                if 'segments' in locals():
                    del segments
                gc.collect()
                torch.cuda.empty_cache()

        # Utility function: e.g. expands 'MRK 1:36-37' → ['MRK 1:36', 'MRK 1:37']
        def expand_verse_range(ref_line):
            match = re.match(r'([A-Za-z]+) (\d+):(\d+)(?:-(\d+))?', ref_line.strip())
            if not match:
                return []

            book, ch, vs_start, vs_end = match.groups()
            ch = int(ch)
            vs_start = int(vs_start)
            vs_end = int(vs_end) if vs_end else vs_start

            return [f"{book} {ch}:{v}" for v in range(vs_start, vs_end + 1)]

        output_df = pd.DataFrame(columns=['filename', 'text', 'source_file', 'start', 'end'])

        for audio_file in audio_files:
            try:
                audio_data = audio_file['data']
                filename = audio_file['filename']
                
                # Load audio from bytes
                section_audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                
                # Expected format is [book]_[startchapter]_[startverse]-[endchapter]_[endverse]
                # (e.g. MRK_001_001-001_013.mp3)
                match = re.search(r'([A-Za-z]+)_(\d{3})_(\d{3})-(\d{3})_(\d{3})', filename)
                if not match:
                    raise ValueError(f"Could not parse chapter/verse range from filename: {filename}")
                book, start_ch, start_vs, end_ch, end_vs = match.groups()
                start_ref = f"{book} {int(start_ch)}:{int(start_vs)}"
                end_ref   = f"{book} {int(end_ch)}:{int(end_vs)}"

                print(f"Processing {start_ref} to {end_ref}")

                # Build full verse map: line index → list of individual references
                ref_map = []
                for i, line in enumerate(ref_lines):
                    refs = expand_verse_range(line)
                    ref_map.append((i, refs))

                # Find matching indices
                start_idx = None
                end_idx = None

                for i, refs in ref_map:
                    if start_ref in refs and start_idx is None:
                        start_idx = i
                    if end_ref in refs:
                        end_idx = i

                if start_idx is None or end_idx is None:
                    raise ValueError(f"Couldn't find verse range from {start_ref} to {end_ref} in ref_lines.")

                # Extract the corresponding lines
                verses = text_lines[start_idx:end_idx + 1]
                verse_refs = ref_lines[start_idx:end_idx + 1]
                chunk_text = '|' + '|'.join(verses).upper() + '|'

                if romanize:
                    uroman = ur.Uroman()
                    chunk_text = uroman.romanize_string(chunk_text)
                    print("Romanized")
                else:
                    print("Not Romanized")

                # Filter unwanted characters
                text_block = ''.join(c for c in chunk_text if c.isalpha() or c == '|')
                text_block = text_block.replace('ʼ', '')

                waveform, sr = audiosegment_to_waveform(section_audio)
                if sr != 16000:
                    section_audio = section_audio.set_frame_rate(16000)
                    section_audio = section_audio.set_channels(1)
                    waveform, sr = audiosegment_to_waveform(section_audio)
                    sr = 16000
                break_segs, trellis = process_audio_and_align(waveform, text_block)
                ratio = waveform.size(1) / trellis.size(0)
                for i in range(len(break_segs)-1):
                    output_filename = verse_refs[i].replace(
                        ' ', '_').replace(':', '_').replace('\n', '')
                    output_key = f'output/{output_filename}.wav'
                    print(output_key)
                    print(verses[i])
                    x0_frames = int(ratio * break_segs[i].end)
                    if i + 1 < len(break_segs):
                        x1_frames = int(ratio * break_segs[i+1].start)
                    else:
                        # For the last segment, set x1 to the end of the waveform
                        x1_frames = waveform.size(1)
                    x0_seconds = round(x0_frames / sr, 3)
                    x1_seconds = round(x1_frames / sr, 3)
                    output_df.loc[len(output_df)] = [output_key, verses[i], filename, x0_seconds, x1_seconds]
                if 'emissions' in locals():
                    del emissions
                if 'segments' in locals():
                    del segments
                if 'trellis' in locals():
                    del trellis
                if 'path' in locals():
                    del path
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                print(f"Skipping {filename} due to OOM: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

        return output_df

    @modal.method()
    def align(self, s3_audio_key: str = None, s3_text_key: str = None, s3_ref_key: str = None,
              bucket: str = "forcedalignment", romanize: bool = False):
        import boto3
        import gc
        import math
        import numpy as np
        import re
        import torch
        import uroman as ur
        from dataclasses import dataclass
        
        # Configure fsspec to handle glob characters in URLs
        fsspec.config.conf['open_expand'] = True
        
        print("starting alignment")

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

                return break_segs, trellis

            finally:
                # Clean up GPU memory
                del input_values
                if 'emissions' in locals():
                    del emissions
                if 'segments' in locals():
                    del segments
                gc.collect()
                torch.cuda.empty_cache()

        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_audio_key)
        mp3_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.mp3')]
        print(mp3_keys[:5])
        ref_text = s3.get_object(Bucket=bucket, Key=s3_ref_key)["Body"].read().decode("utf-8").splitlines()
        all_lines = s3.get_object(Bucket=bucket, Key=s3_text_key)['Body'].read().decode('utf-8').splitlines()
        output_df = pd.DataFrame(columns=['filename', 'text', 'source_file', 'start', 'end'])

        for key in mp3_keys:
            try:
                audio_obj = s3.get_object(Bucket=bucket, Key=key)
                section_audio = AudioSegment.from_file(io.BytesIO(audio_obj['Body'].read()), format="mp3")
                # Expected format is [book]_[startchapter]_[startverse]-[endchapter]_[endverse]
                # (e.g. MRK_001_001-001_013.mp3)
                match = re.search(r'([A-Za-z]+)_(\d{3})_(\d{3})-(\d{3})_(\d{3})', key)
                if not match:
                    raise ValueError(f"Could not parse chapter/verse range from key: {key}")
                book, start_ch, start_vs, end_ch, end_vs = match.groups()
                start_ref = f"{book} {int(start_ch)}:{int(start_vs)}"
                end_ref   = f"{book} {int(end_ch)}:{int(end_vs)}"

                print(f"Processing {start_ref} to {end_ref}")

                # Utility function: e.g. expands 'MRK 1:36-37' → ['MRK 1:36', 'MRK 1:37']
                def expand_verse_range(ref_line):
                    match = re.match(r'([A-Za-z]+) (\d+):(\d+)(?:-(\d+))?', ref_line.strip())
                    if not match:
                        return []

                    book, ch, vs_start, vs_end = match.groups()
                    ch = int(ch)
                    vs_start = int(vs_start)
                    vs_end = int(vs_end) if vs_end else vs_start

                    return [f"{book} {ch}:{v}" for v in range(vs_start, vs_end + 1)]

                # Build full verse map: line index → list of individual references
                ref_map = []
                for i, line in enumerate(ref_text):
                    refs = expand_verse_range(line)
                    ref_map.append((i, refs))

                # Find matching indices
                start_idx = None
                end_idx = None

                for i, refs in ref_map:
                    if start_ref in refs and start_idx is None:
                        start_idx = i
                    if end_ref in refs:
                        end_idx = i

                if start_idx is None or end_idx is None:
                    raise ValueError(f"Couldn't find verse range from {start_ref} to {end_ref} in ref_text.")

                # Extract the corresponding lines
                verses = all_lines[start_idx:end_idx + 1]
                verse_refs = ref_text[start_idx:end_idx + 1]
                chunk_text = '|' + '|'.join(verses).upper() + '|'

                if romanize:
                    uroman = ur.Uroman()
                    chunk_text = uroman.romanize_string(chunk_text)
                    print("Romanized")
                else:
                    print("Not Romanized")

                # Filter unwanted characters
                text_block = ''.join(c for c in chunk_text if c.isalpha() or c == '|')
                text_block = text_block.replace('ʼ', '')

                waveform, sr = audiosegment_to_waveform(section_audio)
                if sr != 16000:
                    section_audio = section_audio.set_frame_rate(16000)
                    section_audio = section_audio.set_channels(1)
                    waveform, sr = audiosegment_to_waveform(section_audio)
                    sr = 16000
                break_segs, trellis = process_audio_and_align(waveform, text_block)
                ratio = waveform.size(1) / trellis.size(0)
                
                for i in range(len(break_segs)-1):
                    output_filename = verse_refs[i].replace(' ', '_').replace(':', '_')
                    output_key = f'output/{output_filename}.wav'
                    print(output_key)
                    print(verses[i])
                    x0_frames = int(ratio * break_segs[i].end)
                    if i + 1 < len(break_segs):
                        x1_frames = int(ratio * break_segs[i+1].start)
                    else:
                        # For the last segment, set x1 to the end of the waveform
                        x1_frames = waveform.size(1)
                    x0_seconds = round(x0_frames / sr, 3)
                    x1_seconds = round(x1_frames / sr, 3)
                    output_df.loc[len(output_df)] = [output_key, verses[i], key.split('/')[-1], x0_seconds, x1_seconds]
                if 'emissions' in locals():
                    del emissions
                if 'segments' in locals():
                    del segments
                if 'trellis' in locals():
                    del trellis
                if 'path' in locals():
                    del path
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                print(f"Skipping {key} due to OOM: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

        #s3.put_object(Bucket=bucket, Key="output/forced_alignment_result.csv", Body=output_df.to_csv(index=False))
        return output_df
