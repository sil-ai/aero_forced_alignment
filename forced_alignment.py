import io
import modal
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "espeak-ng", "sox", "libsox-dev"])
    .pip_install([
        "torch==2.7.1",
        "torchcodec==0.5",
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

# Model selection - use English by default, MMS for other languages
DEFAULT_MODEL = "facebook/wav2vec2-base-960h"  # English
MMS_MODEL = "facebook/mms-1b-all"  # Massively Multilingual Speech model


@app.function(
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=1200,
)
def push_to_hf_dataset(output_records: List[Dict[str, str]],
                    output_dataset_name: Optional[str] = None,
                    token: Optional[str] = None,
                    include_audio: bool = True,
                    original_audio_data: Optional[Dict[str, any]] = None) -> str:
    """
    Push the alignment results to a Hugging Face dataset, preserving original splits.

    Args:
        output_records: List of dictionaries with alignment results
        output_dataset_name: Name of the dataset to create or update
        include_audio: Whether to include audio segments in the dataset
        original_audio_data: Dictionary mapping source file identifiers to audio data
                            Format: {source_id: {"waveform": tensor, "sample_rate": int}}

    Returns:
        The name of the created or updated dataset
    """
    import os
    import numpy as np
    from datasets import Dataset, DatasetDict, Features, Value, Audio
    import torch

    if not output_records:
        raise ValueError("No records to push to dataset")

    print(f"Processing {len(output_records)} records for dataset creation")

    # Group records by split
    split_records = {}
    for record in output_records:
        split_name = record.get('split', 'train')  # Default to 'train' if no split specified
        if split_name not in split_records:
            split_records[split_name] = []
        split_records[split_name].append(record)

    print(f"Found splits: {list(split_records.keys())}")
    for split_name, records in split_records.items():
        print(f"  {split_name}: {len(records)} records")

    # Process each split separately
    datasets_by_split = {}

    for split_name, records in split_records.items():
        print(f"Processing split: {split_name}")

        # Create a copy of records to avoid modifying the original
        processed_records = []

        for i, record in enumerate(records):
            # Create a copy of each record
            new_record = record.copy()
            
            if 'mms_language' in new_record:
                del new_record['mms_language']

            # If including audio, extract segments for each record
            if include_audio and original_audio_data:
                source_id = record.get('source_sample_id') or record.get('source_file')

                if source_id and source_id in original_audio_data:
                    try:
                        audio_info = original_audio_data[source_id]
                        waveform = audio_info['waveform']
                        sample_rate = audio_info['sample_rate']

                        # Extract audio segment based on start/end times
                        start_sample = int(record['start'] * sample_rate)
                        end_sample = int(record['end'] * sample_rate)

                        # Ensure we don't go out of bounds
                        if isinstance(waveform, torch.Tensor):
                            if waveform.dim() > 1:
                                audio_segment = waveform[:, start_sample:end_sample]
                                # Convert to mono if needed
                                if audio_segment.shape[0] > 1:
                                    audio_segment = audio_segment.mean(dim=0)
                                else:
                                    audio_segment = audio_segment.squeeze(0)
                            else:
                                audio_segment = waveform[start_sample:end_sample]
                        else:
                            # Handle numpy arrays
                            if len(waveform.shape) > 1:
                                audio_segment = waveform[:, start_sample:end_sample]
                                if audio_segment.shape[0] > 1:
                                    audio_segment = audio_segment.mean(axis=0)
                                else:
                                    audio_segment = audio_segment.squeeze(0)
                            else:
                                audio_segment = waveform[start_sample:end_sample]

                        # Ensure it's a proper 1D numpy array with float32 dtype
                        audio_segment = np.array(audio_segment, dtype=np.float32)
                        if audio_segment.ndim > 1:
                            audio_segment = audio_segment.flatten()

                        # Validate the audio segment
                        if len(audio_segment) == 0:
                            print(f"Warning: Empty audio segment for record {i} in {split_name}, source: {source_id}")
                            # Skip adding audio for this record but keep the record
                            processed_records.append(new_record)
                            continue

                        # Debug: print type and shape info for first few records
                        if i < 3:
                            print(f"Record {i} ({split_name}): audio_segment type={type(audio_segment)}, shape={audio_segment.shape}, dtype={audio_segment.dtype}")
                            print(f"Record {i} ({split_name}): sample_rate={sample_rate}, audio length={len(audio_segment)}")

                        # Add audio data to record in the exact format expected by HF
                        new_record['audio'] = {
                            'array': audio_segment,
                            'sampling_rate': int(sample_rate)
                        }

                    except Exception as e:
                        print(f"Error processing audio for record {i} in {split_name}, source {source_id}: {e}")
                        # Skip this record's audio but keep the record
                        processed_records.append(new_record)
                        continue
                else:
                    print(f"Warning: No audio data found for source: {source_id}")

            processed_records.append(new_record)

        # Filter records that have audio if we're supposed to include audio
        if include_audio:
            records_with_audio = [r for r in processed_records if 'audio' in r]
            print(f"Records with audio in {split_name}: {len(records_with_audio)} out of {len(processed_records)}")

            if len(records_with_audio) == 0:
                print(f"Warning: No records have audio data in {split_name}, creating dataset without audio")
                include_audio_for_split = False
                dataset_records = processed_records
            else:
                include_audio_for_split = True
                dataset_records = records_with_audio
        else:
            include_audio_for_split = False
            dataset_records = processed_records

        print(f"Creating dataset for {split_name} from {len(dataset_records)} records")

        # Define the features schema explicitly
        if include_audio_for_split:
            features = Features({
                'chunk_id': Value('string'),
                'text': Value('string'),
                'source_sample_id': Value('string'),
                'chunk_index': Value('int64'),
                'start': Value('float64'),
                'end': Value('float64'),
                'duration': Value('float64'),
                'word_count': Value('int64'),
                'asr_transcription': Value('string'),
                'match_score': Value('int64'),
                'dataset_name': Value('string'),
                'split': Value('string'),
                'audio': Audio(sampling_rate=16000)
            })
        else:
            features = Features({
                'chunk_id': Value('string'),
                'text': Value('string'),
                'source_sample_id': Value('string'),
                'chunk_index': Value('int64'),
                'start': Value('float64'),
                'end': Value('float64'),
                'duration': Value('float64'),
                'word_count': Value('int64'),
                'asr_transcription': Value('string'),
                'match_score': Value('int64'),
                'dataset_name': Value('string'),
                'split': Value('string')
            })

        # Create dataset with explicit features
        try:
            dataset = Dataset.from_list(dataset_records, features=features)
            print(f"Dataset for {split_name} created successfully with explicit features")
        except Exception as e:
            print(f"Error creating dataset for {split_name} with features: {e}")

            # Try to fix the data format
            print(f"Attempting to fix audio data format for {split_name}...")

            # Double-check all audio data is properly formatted
            for i, record in enumerate(dataset_records):
                if 'audio' in record:
                    audio_data = record['audio']
                    if isinstance(audio_data['array'], list):
                        print(f"Converting list to numpy array for record {i} in {split_name}")
                        audio_data['array'] = np.array(audio_data['array'], dtype=np.float32)
                    elif not isinstance(audio_data['array'], np.ndarray):
                        print(f"Converting {type(audio_data['array'])} to numpy array for record {i} in {split_name}")
                        audio_data['array'] = np.array(audio_data['array'], dtype=np.float32)

                    # Ensure it's 1D
                    if audio_data['array'].ndim > 1:
                        audio_data['array'] = audio_data['array'].flatten()

                    # Ensure correct dtype
                    if audio_data['array'].dtype != np.float32:
                        audio_data['array'] = audio_data['array'].astype(np.float32)

            # Try again with corrected data
            try:
                dataset = Dataset.from_list(dataset_records, features=features)
                print(f"Dataset for {split_name} created successfully after fixing audio format")
            except Exception as e2:
                print(f"Still failing for {split_name} after audio format fix: {e2}")
                # Final fallback: create dataset without audio and warn user
                records_without_audio = []
                for record in dataset_records:
                    new_record = {k: v for k, v in record.items() if k != 'audio'}
                    records_without_audio.append(new_record)

                features_no_audio = Features({k: v for k, v in features.items() if k != 'audio'})
                dataset = Dataset.from_list(records_without_audio, features=features_no_audio)
                print(f"WARNING: Created dataset for {split_name} WITHOUT audio due to formatting issues")

        datasets_by_split[split_name] = dataset

    # Create DatasetDict with all splits
    dataset_dict = DatasetDict(datasets_by_split)

    # If no name provided, use a default name
    if not output_dataset_name:
        output_dataset_name = "forced_alignment_results"

    # Save the dataset with splits
    print(f"Pushing dataset '{output_dataset_name}' with {len(datasets_by_split)} splits to HuggingFace Hub")
    dataset_dict.push_to_hub(output_dataset_name, token=token, private=True)

    return output_dataset_name


@app.function(
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=3600,
)
def align_from_dataset_parallel(
    dataset_name: str,
    text_column: str = "text",
    audio_column: str = "audio",
    id_column: str = None,
    max_words: int = 100,
    romanize: bool = False,
    mms_lang: Optional[str] = None,
    split: Optional[str] = None,
    limit: int = None,
    output_dataset_name: Optional[str] = None,
    batch_size: int = 100,
) -> List[Dict[str, str]]:
    """
    Perform forced alignment on audio and text from a Hugging Face dataset using parallel processing.

    Args:
        dataset_name: Name of the HF dataset
        mms_lang: MMS language code
        batch_size: Number of samples to process in parallel (limited by GPU availability)
        ... (other args same as original method)

    Returns:
        List of dictionaries with alignment results for each chunk
    """
    from datasets import load_dataset, Audio
    import os
    import torch

    # Determine which splits to process
    if split is None:
        print(f"Loading dataset info for: {dataset_name}")
        dataset_info = load_dataset(dataset_name, streaming=False)
        splits_to_process = list(dataset_info.keys())
        print(f"Found splits: {splits_to_process}")
    else:
        splits_to_process = [split]

    all_output_records = []
    original_audio_data = {}  # Store original audio for each sample

    # Process each split
    for current_split in splits_to_process:
        print(f"Processing split: {current_split}")
        # Load dataset for current split
        dataset = load_dataset(dataset_name, split=current_split, streaming=False)

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Loaded {len(dataset)} samples from split '{current_split}'")

        texts = dataset[text_column]
        ids   = dataset[id_column] if id_column else [None]*len(dataset)

        dataset = dataset.cast_column(audio_column, Audio(decode=True))
        dataset = dataset.with_format("numpy", columns=[audio_column])

        sample_dicts = []
        for idx, ex in enumerate(dataset):
            sample_id = ids[idx] if ids[idx] is not None else f"{current_split}_sample_{idx}"

            # Store original audio data for later use
            audio_array = ex[audio_column]["array"]
            sample_rate = ex[audio_column]["sampling_rate"]
            waveform, sr = audioarray_to_waveform(audio_array, sample_rate)

            original_audio_data[sample_id] = {
                'waveform': waveform,
                'sample_rate': sr
            }

            sample_dicts.append({
                "waveform": audio_array,
                "sr": sample_rate,
                "text": texts[idx],
                "id": sample_id,
                "idx": idx,
            })

        # Process samples in parallel batches
        for i in range(0, len(sample_dicts), batch_size):
            batch = sample_dicts[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} ({len(batch)} samples)")
            batch_results = list(process_single_dataset_sample.map(
                batch,
                range(i, i + batch_size),
                [dataset_name] * len(batch),
                [current_split] * len(batch),
                [text_column] * len(batch),
                [audio_column] * len(batch),
                [id_column] * len(batch),
                [max_words] * len(batch),
                [romanize] * len(batch),
                [mms_lang or None] * len(batch)  # Pass MMS language code (None for English)
            ))

            # Flatten results from batch
            for result in batch_results:
                all_output_records.extend(result)

            print(f"Completed batch, total chunks so far: {len(all_output_records)}")

    print(f"Total chunks created across all splits: {len(all_output_records)}")

    if output_dataset_name:
        push_to_hf_dataset.remote(all_output_records, output_dataset_name,
                                token=os.getenv("HF_TOKEN"),
                                original_audio_data=original_audio_data)

    return all_output_records


@app.function(
    timeout=3600,
)
def align_from_s3_parallel(
    audio_files: List[Dict[str, Union[str, bytes, List[Dict[str, str]]]]],
    romanize: bool = False,
    mms_lang: Optional[str] = None,
    batch_size: int = 10,
    output_dataset_name: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Perform forced alignment on S3 audio files using parallel processing with MMS language support.
    
    Args:
        audio_files: List of file specifications
        romanize: Whether to romanize the text
        mms_lang: MMS language code
        batch_size: Number of files to process in parallel (limited by GPU availability)
        output_dataset_name: Name of the dataset to create or update
        
    Returns:
        List of dictionaries with alignment results
    """
    import os

    if isinstance(audio_files, dict):
        audio_files = [audio_files]
    elif isinstance(audio_files, str):
        raise ValueError("audio_files must be a list of dicts, not a string")

    all_output_records = []
    original_audio_data = {}  # Store original audio for each file

    # Process files in parallel batches
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size} ({len(batch)} files)")

        # Use Modal's .map() to process batch in parallel
        batch_results = list(process_single_s3_file.map(
            batch,
            [romanize] * len(batch),
            [mms_lang or None] * len(batch)  # Pass MMS language code (None for English)
        ))

        # Collect original audio data for each file in the batch
        for j, file_data in enumerate(batch):
            filename = file_data.get('filename', file_data['s3_path'].split('/')[-1].split('.')[0])
            # Load the original audio for this file
            try:
                s3_path = file_data['s3_path']
                bucket, key = s3_path.replace("s3://", "").split("/", 1)

                import boto3
                import io
                from pydub import AudioSegment

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

                original_audio_data[filename] = {
                    'waveform': waveform,
                    'sample_rate': sr
                }
            except Exception as e:
                print(f"Warning: Could not load original audio for {filename}: {e}")

        # Flatten results from batch
        for result in batch_results:
            all_output_records.extend(result)
        for record in all_output_records:
            if 'mms_language' in record:
                del record['mms_language']

        print(f"Completed batch, total segments so far: {len(all_output_records)}")

    print(f"Total segments created: {len(all_output_records)}")

    if output_dataset_name:
        push_to_hf_dataset.remote(all_output_records, output_dataset_name,
                                token=os.getenv("HF_TOKEN"),
                                original_audio_data=original_audio_data)

    return all_output_records


def audiosegment_to_waveform(audio_segment):
    """Convert AudioSegment to waveform tensor."""
    import numpy as np
    import torch

    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)).T  # shape (channels, samples)
    else:
        samples = samples[np.newaxis, :]  # shape (1, samples)
    samples /= (1 << (8 * audio_segment.sample_width - 1))  # normalize to [-1, 1]
    return torch.from_numpy(samples), audio_segment.frame_rate


@app.function(
    secrets=[s3_access_credentials],
)
def process_single_s3_file(file_data: Dict[str, Union[str, List[Dict[str, str]]]], 
                          romanize: bool = False, 
                          mms_lang: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Process a single S3 audio file for forced alignment with optional MMS language support.
    
    Args:
        file_data: File specification with s3_path and ref_text
        romanize: Whether to romanize the text
        mms_lang: MMS language code
    """
    import gc
    import numpy as np
    import os
    import re
    import torch
    import uroman as ur
    import boto3
    from fuzzywuzzy import fuzz
    from pydub import AudioSegment
    import io

    output_records = []

    try:
        s3_path = file_data['s3_path']
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        if 'filename' in file_data:
            filename = file_data['filename']
        else:
            filename = key.split('/')[-1].split('.')[0]
        ref_text = file_data['ref_text']  # List of dicts with 'key' and 'text'

        text_lines = [v['text'] for v in ref_text]
        if 'key' in ref_text[0]:
            text_refs = [v['key'] for v in ref_text]
        else:
            text_refs = [filename + '_' + str(i) for i in range(len(text_lines))]
        
        chunk_text = '|' + '|'.join(text_lines).upper() + '|'

        if romanize:
            uroman = ur.Uroman()
            chunk_text = uroman.romanize_string(chunk_text)
            print(f"Romanized text for {'MMS-' + mms_lang if mms_lang else 'standard'} processing")
        else:
            print("Using original text")

        # Filter unwanted characters
        text_block = ''.join(c for c in chunk_text if c.isalpha() or c == '|')
        text_block = text_block.replace('ʼ', '')
        
        if not text_block.strip():
            print(f"[WARNING] Empty or invalid text_block for file: {filename}")
            return output_records

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

        # Perform alignment with appropriate model
        break_segs, trellis, emissions = process_audio_and_align.remote(
            waveform, text_block, mms_lang, romanize
        )
        ratio = waveform.size(1) / trellis.size(0)

        # Extract ASR segments
        asr_segments = extract_asr_segments(emissions, break_segs, ratio, mms_lang, romanize)

        prev_x1_seconds = 0
        for i in range(len(break_segs)-1):
            output_filename = text_refs[i].replace(' ', '_').replace(':', '_').replace('\n', '')
            output_key = f'output/{output_filename}.wav'
            x0_frames = int(ratio * break_segs[i].end)
            if i + 1 < len(break_segs):
                x1_frames = int(ratio * break_segs[i+1].start)
            else:
                x1_frames = waveform.size(1)
            x0_seconds = round(x0_frames / sr, 3)
            x1_seconds = round(x1_frames / sr, 3)
            x0_seconds = x0_seconds - 0.1
            if x0_seconds < prev_x1_seconds:
                x0_seconds = prev_x1_seconds
            
            # Use the corresponding ASR segment
            asr_transcription = asr_segments[i].lower() if i < len(asr_segments) else ""
            # For match score comparison, use romanized text if romanization was applied
            text_for_comparison = text_lines[i]
            if romanize:
                uroman = ur.Uroman()
                text_for_comparison = uroman.romanize_string(text_lines[i], lcode=mms_lang if mms_lang else None)
            match_score = fuzz.ratio(text_for_comparison, asr_transcription)

            output_records.append({
                'filename': output_key,
                'text': text_lines[i],
                'source_file': filename,
                'start': x0_seconds,
                'end': x1_seconds,
                'asr_transcription': asr_transcription,
                'match_score': match_score
            })
            prev_x1_seconds = x1_seconds

        print(f"Processed file: {filename}, created {len(output_records)} segments")

    except torch.cuda.OutOfMemoryError as e:
        print(f"Skipping {filename} due to OOM: {e}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    finally:
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    return output_records


@app.function()
def split_text_into_chunks(text, max_words):
    """Split text into chunks based on sentences with max word limit."""
    import re


    parts = re.split(r'([.!?]+)', text)

    # Reconstruct sentences with their punctuation
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()
        if sentence:  # Only add non-empty sentences
            # Add punctuation if it exists
            if i + 1 < len(parts) and parts[i + 1]:
                sentence += parts[i + 1]
            sentences.append(sentence)

    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_word_count = len(words)

        # If adding this sentence would exceed max_words
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_word_count
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_word_count += sentence_word_count

        # If a single sentence exceeds max_words, split it further
        if sentence_word_count > max_words:
            if current_chunk != sentence:  # Save previous chunk first
                chunks.append(current_chunk.replace(sentence, "").strip())

            # Split long sentence into word chunks
            words = sentence.split()
            for i in range(0, len(words), max_words):
                word_chunk = " ".join(words[i:i + max_words])
                chunks.append(word_chunk)

            current_chunk = ""
            current_word_count = 0

    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def audioarray_to_waveform(audio_array, sample_rate):
    """Convert audio array from dataset to waveform tensor."""
    import numpy as np
    import torch
    import librosa

    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Normalize to [-1, 1]
    if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    waveform = torch.from_numpy(audio_array).unsqueeze(0)  # Add channel dimension
    return waveform, sample_rate


@app.function(
    gpu="L40S",
)
def process_audio_and_align(waveform, transcript_text, mms_lang=None, romanize=False):
    """Core alignment processing with optional MMS language support."""
    import torch
    import gc
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    # Select model based on language
    if mms_lang:
        model_name = MMS_MODEL
        print(f"Using MMS model for language: {mms_lang}")
    else:
        model_name = DEFAULT_MODEL
        print("Using default English model")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)
    
    # Set target language for MMS model
    if mms_lang and hasattr(processor.tokenizer, 'set_target_lang'):
        processor.tokenizer.set_target_lang(mms_lang)
        model.load_adapter(mms_lang)
        print(f"Set MMS target language to: {mms_lang}")
    
    labels = processor.tokenizer.convert_ids_to_tokens(list(range(model.lm_head.out_features)))
    label_dict = {c: i for i, c in enumerate(labels)}

    waveform = waveform.mean(dim=0)
    print("Transcript Text: " + transcript_text.strip().replace("\n", "|"))
    transcript_tokens = list(transcript_text.strip().replace("\n", "|"))

    token_ids = [label_dict[p] for p in transcript_tokens if p in label_dict]
    if not token_ids:
        raise ValueError("None of the tokens in transcript were found in the model label set.")

    try:
        input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            emissions = model(input_values).logits
        emissions = torch.log_softmax(emissions, dim=-1)[0]

        trellis = get_trellis(emissions, token_ids, device)
        path = backtrack(trellis, emissions, token_ids)
        segments = merge_repeats(path, transcript_tokens)
        print("Segments: " + str(segments[:5]))
        break_segs = [seg for seg in segments if seg.label == "|"]
        trellis = trellis.cpu()
        emissions = emissions.cpu()

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


def get_trellis(emission, tokens, device, blank_id=0):
    """Get trellis for forced alignment."""
    import torch

    num_frame, num_tokens = emission.size(0), len(tokens)
    trellis = torch.zeros((num_frame, num_tokens), device=device)
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
    """Backtrack through trellis to find optimal path."""
    
    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float

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
    """Merge repeated tokens in the path."""
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"
    
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


def extract_asr_segments(emissions, break_segs, ratio, mms_lang=None, romanize=False):
    """Extract ASR transcriptions for each segment with optional MMS language support."""
    import torch
    import uroman as ur
    from transformers import Wav2Vec2Processor

    # Use appropriate model
    model_name = MMS_MODEL if mms_lang else DEFAULT_MODEL
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    # Set target language for MMS
    if mms_lang and hasattr(processor.tokenizer, 'set_target_lang'):
        processor.tokenizer.set_target_lang(mms_lang)
    
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
        segment_transcription = processor.batch_decode(predicted_ids.unsqueeze(0))[0]

        asr_segments.append(segment_transcription)

    # Apply romanization to all segments if requested
    if romanize:
        uroman = ur.Uroman()
        asr_segments = [uroman.romanize_string(text, lcode=mms_lang if mms_lang else None) for text in asr_segments]

    return asr_segments


@app.function(timeout=600)
def process_single_dataset_sample(
    sample_dict,
    idx,
    dataset_name: str,
    split: str,
    text_column: str,
    audio_column: str,
    id_column: Optional[str],
    max_words: int,
    romanize: bool,
    mms_lang: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Process a single dataset sample for forced alignment with optional MMS language support.
    """
    import uroman as ur
    from fuzzywuzzy import fuzz

    output_records = []

    text = sample_dict[text_column]

    # Get sample ID
    if id_column and id_column in sample_dict:
        sample_id = str(sample_dict[id_column])
    else:
        sample_id = f"{split}_sample_{idx}"

    # Split text into chunks
    text_chunks = split_text_into_chunks.remote(text, max_words)
    print(f"Text chunks for sample {idx} ({sample_id}): {text_chunks}")

    if not text_chunks:
        print(f"[WARNING] No valid text chunks for sample: {sample_id}")
        return output_records

    print(f"Sample ID: {sample_id}, Text chunks: {len(text_chunks)}")

    waveform = sample_dict["waveform"]  # np.ndarray
    sr       = sample_dict["sr"]        # int
    text     = sample_dict["text"]
    sample_id= sample_dict.get("id") or f"{split}_sample_{sample_dict['idx']}"
    waveform, sr = audioarray_to_waveform(waveform, sr)

    print(f"Processing sample {idx} ({sample_id}) with {len(text_chunks)} text chunks")

    # Create combined text for alignment
    chunk_text = '|' + '|'.join(text_chunks).upper() + '|'

    if romanize:
        uroman = ur.Uroman()
        chunk_text = uroman.romanize_string(chunk_text)
        print(f"Romanized text for {'MMS-' + mms_lang if mms_lang else 'standard'} processing")
    else:
        print("Using original text")

    # Filter unwanted characters
    text_block = ''.join(c for c in chunk_text if c.isalpha() or c == '|')
    text_block = text_block.replace('ʼ', '')

    if not text_block.strip():
        print(f"[WARNING] Empty or invalid text_block for sample: {sample_id}")
        return output_records

    # Perform alignment
    break_segs, trellis, emissions = process_audio_and_align.remote(waveform, text_block, mms_lang, romanize)

    ratio = waveform.size(1) / trellis.size(0)
    
    # Extract ASR segments
    asr_segments = extract_asr_segments(emissions, break_segs, ratio, mms_lang, romanize)

    # Create output records for each chunk
    prev_x1_seconds = 0
    for i in range(len(break_segs) - 1):
        if i < len(text_chunks):
            chunk_id = f"{sample_id}_chunk_{i}"
            x0_frames = int(ratio * break_segs[i].end)
            if i + 1 < len(break_segs):
                x1_frames = int(ratio * break_segs[i + 1].start)
            else:
                x1_frames = waveform.size(1)

            x0_seconds = round(x0_frames / sr, 3)
            x1_seconds = round(x1_frames / sr, 3)
            x0_seconds = x0_seconds - 0.1
            if x0_seconds < prev_x1_seconds:
                x0_seconds = prev_x1_seconds

            # Use the corresponding ASR segment
            asr_transcription = asr_segments[i].lower() if i < len(asr_segments) else ""
            # For match score comparison, use romanized text if romanization was applied
            text_for_comparison = text_chunks[i].lower()
            if romanize:
                uroman = ur.Uroman()
                text_for_comparison = uroman.romanize_string(text_chunks[i], lcode=mms_lang if mms_lang else None).lower()
            match_score = fuzz.ratio(text_for_comparison, asr_transcription)

            # Append record
            output_records.append({
                'chunk_id': chunk_id,
                'text': text_chunks[i],
                'source_sample_id': sample_id,
                'chunk_index': i,
                'start': x0_seconds,
                'end': x1_seconds,
                'duration': round(x1_seconds - x0_seconds, 3),
                'word_count': len(text_chunks[i].split()),
                'asr_transcription': asr_transcription,
                'match_score': match_score,
                'dataset_name': dataset_name,
                'split': split,
            })
            prev_x1_seconds = x1_seconds

    print(f"Processed {split} sample {idx}: {sample_id}, created {len(text_chunks)} chunks")

    return output_records