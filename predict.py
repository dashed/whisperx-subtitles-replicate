import copy
import gc
import math
import os
import re
import shutil
import tempfile
import time
from typing import Any, List, Optional, TypedDict

import ffmpeg
import pysbd
import torch
import whisperx
from cog import BaseModel, BasePredictor, Input, Path
from whisperx.audio import N_SAMPLES, log_mel_spectrogram

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3"


class Word(TypedDict):
    end: Optional[float]
    word: str
    score: Optional[float]
    start: Optional[float]


class Segment(TypedDict):
    end: float
    text: str
    start: float
    words: List[Word]


class Cue(TypedDict):
    text: str
    start: float
    end: float
    word_data: Optional[List[Word]]


Segments = List[Segment]


class Output(BaseModel):
    segments: Any
    detected_language: str
    srt_output: str
    srt_file: Path


class Predictor(BasePredictor):
    def setup(self):
        source_folder = "./models/vad"
        destination_folder = "../root/.cache/torch"
        file_name = "whisperx-vad-segmentation.bin"

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        language: str = Input(
            description="ISO code of the language spoken in the audio, specify None to perform language detection",
            default=None,
        ),
        language_detection_min_prob: float = Input(
            description="If language is not specified, then the language will be detected recursively on different "
            "parts of the file until it reaches the given probability",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="If language is not specified, then the language will be detected following the logic of "
            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
            "retries is reached, the most probable language is kept.",
            default=5,
        ),
        initial_prompt: str = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=64
        ),
        temperature: float = Input(
            description="Temperature to use for sampling", default=0
        ),
        vad_onset: float = Input(description="VAD onset", default=0.500),
        vad_offset: float = Input(description="VAD offset", default=0.363),
        align_output: bool = Input(
            description="Aligns whisper output to get accurate word-level timestamps",
            default=True,
        ),
        diarization: bool = Input(
            description="Assign speaker ID labels", default=False
        ),
        huggingface_access_token: str = Input(
            description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
            "the user agreement for the models specified in the README.",
            default=None,
        ),
        min_speakers: int = Input(
            description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        max_speakers: int = Input(
            description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=False,
        ),
    ) -> Output:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
            }

            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

            audio_duration = get_audio_duration(audio_file)

            if (
                language is None
                and language_detection_min_prob > 0
                and audio_duration > 30000
            ):
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms),
                )

                segments_starts = distribute_segments_equally(
                    audio_duration, segments_duration_ms, language_detection_max_tries
                )

                print(
                    "Detecting languages on segments starting at "
                    + ", ".join(map(str, segments_starts))
                )

                detected_language_details = detect_language(
                    audio_file,
                    segments_starts,
                    language_detection_min_prob,
                    language_detection_max_tries,
                    asr_options,
                    vad_options,
                )

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                print(
                    f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                    f"{detected_language_iterations} iterations."
                )

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(
                whisper_arch,
                device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
            )

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output:
                if (
                    detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH
                    or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF
                ):
                    result = align(audio, result, debug)
                else:
                    print(
                        f"Cannot align output as language {detected_language} is not supported for alignment"
                    )

            if diarization:
                result = diarize(
                    audio,
                    result,
                    debug,
                    huggingface_access_token,
                    min_speakers,
                    max_speakers,
                )

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )

        audio_basename = os.path.basename(str(audio_file)).rsplit(".", 1)[0]
        srt_file = f"/tmp/{audio_basename}.{detected_language}.srt"
        result2 = copy.deepcopy(result)
        srt_output = generate_srt(result2["segments"], language=detected_language)
        with open(srt_file, "w", encoding="utf-8") as srt:
            srt.write(srt_output)

        return Output(
            segments=result["segments"],
            detected_language=detected_language,
            srt_output=srt_output,
            srt_file=Path(srt_file),
        )


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    return float(stream["duration"]) * 1000


def detect_language(
    full_audio_file_path,
    segments_starts,
    language_detection_min_prob,
    language_detection_max_tries,
    asr_options,
    vad_options,
    iteration=1,
):
    model = whisperx.load_model(
        whisper_arch,
        device,
        compute_type=compute_type,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(
        full_audio_file_path, start_ms, 30000
    )

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(
        f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})"
    )

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration,
    }

    if (
        language_probability >= language_detection_min_prob
        or iteration >= language_detection_max_tries
    ):
        return detected_language

    next_iteration_detected_language = detect_language(
        full_audio_file_path,
        segments_starts,
        language_detection_min_prob,
        language_detection_max_tries,
        asr_options,
        vad_options,
        iteration + 1,
    )

    if (
        next_iteration_detected_language["probability"]
        > detected_language["probability"]
    ):
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        print(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg.input(input_file_path, ss=start_time_ms / 1000)
                .output(temp_file.name, t=duration_ms / 1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode("utf-8"))
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=huggingface_access_token, device=device
    )
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result


def generate_srt(segments, language) -> str:
    segmenter = pysbd.Segmenter(language=language, clean=False)

    output_srt = ""

    all_cues = []
    for segment in segments:
        text = segment["text"]
        word_data = segment.get("words", [])

        sentences = split_at_sentence_end(
            segmenter=segmenter, text=text, word_data=word_data
        )
        all_cues.extend(sentences)

    # After merging cues
    merged_cues = merge_short_cues(
        all_cues, min_duration=3, max_line_length=42, max_lines=2
    )

    # Split long cues using word timings
    processed_cues = split_long_cues_with_word_timings(
        merged_cues, max_line_length=42, max_lines=2
    )

    srt_index = 1
    for cue in processed_cues:
        formatted_text = split_subtitle(cue["text"])

        output_srt += f"{srt_index}\n"
        output_srt += (
            f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n"
        )
        output_srt += f"{formatted_text}\n\n"

        srt_index += 1

    return output_srt


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00,000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def split_subtitle(text: str, max_chars=42) -> str:
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_chars and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def extract_words(text: str):
    return set(re.findall(r"\b[\w\']+\b", text.lower()))


def split_at_sentence_end(segmenter, text: str, word_data: List[Word]) -> List[Cue]:
    # sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = segmenter.segment(text)

    result: List[Cue] = []
    current_word_index = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_word_count = len(sentence.split())
            end = current_word_index + sentence_word_count
            sentence_word_data = word_data[current_word_index:end]
            if sentence_word_data:
                start_time = next(
                    (word["start"] for word in sentence_word_data if "start" in word),
                    None,
                )
                end_time = next(
                    (
                        word["end"]
                        for word in reversed(sentence_word_data)
                        if "end" in word
                    ),
                    None,
                )
                if start_time is not None and end_time is not None:
                    result.append(
                        {
                            "text": sentence,
                            "start": start_time,
                            "end": end_time,
                            "word_data": sentence_word_data,
                        }
                    )
                else:
                    # If start or end time is missing, use the previous valid timestamp
                    if result:
                        prev_end = result[-1]["end"]
                        result.append(
                            {
                                "text": sentence,
                                "start": prev_end,
                                "end": prev_end
                                + 1,  # Add 1 second as a placeholder duration
                            }
                        )
                    else:
                        # If it's the first sentence and times are missing, use 0 as start time
                        result.append(
                            {
                                "text": sentence,
                                "start": 0,
                                "end": 1,  # Add 1 second as a placeholder duration
                            }
                        )
            current_word_index += sentence_word_count
    return result


def merge_short_cues(cues: List[Cue], min_duration=3, max_line_length=42, max_lines=2):
    merged_cues: List[Cue] = []
    current_cue: Cue | None = None

    for cue in cues:
        if current_cue is None:
            current_cue = cue
        else:
            duration = cue["end"] - current_cue["start"]
            # Calculate the combined text length
            combined_text = current_cue["text"] + " " + cue["text"]
            # Use the split_subtitle function to see how the text would split
            split_lines = split_subtitle(
                combined_text, max_chars=max_line_length
            ).split("\n")
            num_lines = len(split_lines)
            # Determine if the merged text would exceed acceptable length
            if duration < min_duration and num_lines <= max_lines:
                # Merge the cues
                current_cue["text"] = combined_text
                current_cue["end"] = cue["end"]
            else:
                # Add the current cue to the list and start a new one
                merged_cues.append(current_cue)
                current_cue = cue

    if current_cue:
        merged_cues.append(current_cue)

    return merged_cues


def split_long_cue_without_word_timings(
    cue: Cue, max_line_length=42, max_lines=2
) -> List[Cue]:
    # Split the text into lines
    split_text = split_subtitle(cue["text"], max_chars=max_line_length)
    lines = split_text.split("\n")
    # Split lines into chunks of max_lines lines
    chunks = []
    current_chunk = []
    for line in lines:
        current_chunk.append(line)
        if len(current_chunk) == max_lines:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    # Distribute the cue's duration among the chunks proportionally
    total_text_length = sum(len(chunk.replace("\n", " ")) for chunk in chunks)
    start_time = cue["start"]
    end_time = cue["end"]
    total_duration = end_time - start_time if end_time > start_time else 0
    new_cues = []
    for chunk in chunks:
        chunk_text_length = len(chunk.replace("\n", " "))
        proportion = (
            chunk_text_length / total_text_length if total_text_length > 0 else 0
        )
        chunk_duration = total_duration * proportion if total_duration > 0 else 0
        chunk_end_time = start_time + chunk_duration
        new_cues.append({"text": chunk, "start": start_time, "end": chunk_end_time})
        start_time = chunk_end_time  # Next chunk starts here
    return new_cues


def split_long_cues_with_word_timings(
    cues: List[Cue], max_line_length=42, max_lines=2, min_duration=1.0
) -> List[Cue]:
    new_cues: List[Cue] = []
    for cue in cues:
        words = cue["text"].split()
        word_timings = cue.get("word_data")
        if not word_timings or len(words) != len(word_timings):
            # Handle missing word_data or mismatched lengths
            # Fallback to splitting without word timings
            split_cues = split_long_cue_without_word_timings(
                cue, max_line_length, max_lines
            )
            new_cues.extend(split_cues)
            continue

        chunks = []
        current_chunk_words = []
        current_chunk_timings = []

        for idx, (word, word_timing) in enumerate(zip(words, word_timings)):
            # Tentatively add the word to the current chunk
            temp_chunk_words = current_chunk_words + [word]
            temp_chunk_text = " ".join(temp_chunk_words)
            temp_formatted_text = split_subtitle(
                temp_chunk_text, max_chars=max_line_length
            )
            num_lines = len(temp_formatted_text.split("\n"))

            if num_lines > max_lines and current_chunk_words:
                # Adding this word exceeds max_lines, so finalize the current chunk
                chunk_text = " ".join(current_chunk_words)
                chunk_word_timings = current_chunk_timings.copy()
                chunks.append(
                    {
                        "words": current_chunk_words.copy(),
                        "timings": chunk_word_timings,
                    }
                )
                # Start new chunk with the current word
                current_chunk_words = [word]
                current_chunk_timings = [word_timing]
            else:
                # Add the word to current chunk
                current_chunk_words.append(word)
                current_chunk_timings.append(word_timing)

        # Add any remaining words as a chunk
        if current_chunk_words:
            chunk_word_timings = current_chunk_timings.copy()
            chunks.append(
                {
                    "words": current_chunk_words.copy(),
                    "timings": chunk_word_timings,
                }
            )

        # Create new cues from chunks and adjust for minimum duration
        for i, chunk in enumerate(chunks):
            chunk_words = chunk["words"]
            chunk_word_timings = chunk["timings"]
            chunk_text = " ".join(chunk_words)
            chunk_formatted_text = split_subtitle(chunk_text, max_chars=max_line_length)
            num_lines = len(chunk_formatted_text.split("\n"))
            # If the chunk still exceeds max_lines, handle accordingly
            if num_lines > max_lines:
                # Optional: Implement recursive splitting or accept that this chunk exceeds max_lines
                # For now, we'll proceed without further splitting
                pass

            # Get start and end times
            start_time = next(
                (
                    wt.get("start")
                    for wt in chunk_word_timings
                    if wt.get("start") is not None
                ),
                cue["start"],
            )
            end_time = next(
                (
                    wt.get("end")
                    for wt in reversed(chunk_word_timings)
                    if wt.get("end") is not None
                ),
                cue["end"],
            )
            duration = end_time - start_time

            # If the duration is less than min_duration, attempt to merge with adjacent chunks
            if duration < min_duration:
                # Try to merge with the next chunk
                if i + 1 < len(chunks):
                    # Merge with next chunk
                    next_chunk = chunks[i + 1]
                    merged_words = chunk_words + next_chunk["words"]
                    merged_timings = chunk_word_timings + next_chunk["timings"]
                    merged_text = " ".join(merged_words)
                    merged_formatted_text = split_subtitle(
                        merged_text, max_chars=max_line_length
                    )
                    num_lines = len(merged_formatted_text.split("\n"))

                    # Check if merged cue respects formatting constraints
                    if num_lines <= max_lines + 1:  # Allow one extra line for merging
                        # Update the next chunk with merged data
                        chunks[i + 1] = {
                            "words": merged_words,
                            "timings": merged_timings,
                        }
                        continue  # Skip adding current chunk, as it's merged
                # Else, try to merge with the previous chunk
                elif new_cues:
                    prev_cue = new_cues[-1]
                    merged_text = prev_cue["text"] + " " + chunk_text
                    merged_formatted_text = split_subtitle(
                        merged_text, max_chars=max_line_length
                    )
                    num_lines = len(merged_formatted_text.split("\n"))
                    if num_lines <= max_lines + 1:  # Allow one extra line for merging
                        # Update previous cue with merged data
                        prev_cue["text"] = merged_text
                        prev_cue["end"] = end_time
                        # Handle 'word_data'
                        prev_word_data = prev_cue.get("word_data", None)
                        if prev_word_data is not None and chunk_word_timings:
                            prev_cue["word_data"] = prev_word_data + chunk_word_timings
                        else:
                            # If either 'word_data' is missing, set it to None
                            prev_cue["word_data"] = None
                        continue
                # If cannot merge, proceed with the short duration
                print(f"Cue '{chunk_text}' has short duration ({duration}s)")
            # Create the cue
            new_cues.append(
                {
                    "text": chunk_text,
                    "start": start_time,
                    "end": end_time,
                    "word_data": chunk_word_timings,
                }
            )
    return new_cues
