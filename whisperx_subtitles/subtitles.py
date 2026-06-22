"""Pure subtitle-formatting logic: sentence splitting, cue merging/splitting, SRT.

Imports only the standard library + pysbd, so it is unit-testable without the
GPU/torch/whisperx stack.
"""

from __future__ import annotations

import re

import pysbd

from .config import DESIRED_WPS
from .types import Cue, Word


def generate_srt(segments, language) -> str:
    segmenter = None
    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except Exception as e:
        print(f"Failed to initialize segmenter for language {language}: {e}")

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
        all_cues, min_duration=3, max_line_length=35, max_lines=2
    )

    # Split long cues using word timings
    processed_cues = split_long_cues_with_word_timings(
        merged_cues, max_line_length=35, max_lines=2
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
    lines: list[str] = []
    current_line: list[str] = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + (1 if current_line else 0) > max_chars:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            if current_line:
                current_line.append(word)
                current_length += word_length + 1  # Account for space
            else:
                current_line.append(word)
                current_length += word_length

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def extract_words(text: str):
    return set(re.findall(r"\b[\w\']+\b", text.lower()))


def split_sentence_heuristically(
    sentence: str, max_line_length: int, max_lines: int
) -> list[str]:
    # Check if the sentence exceeds formatting constraints
    formatted_text = split_subtitle(sentence, max_chars=max_line_length)
    num_lines = len(formatted_text.split("\n"))

    if num_lines <= max_lines:
        return [sentence.strip()]

    # If the sentence is too long, split it
    # Define punctuation and conjunctions to split on
    split_pattern = re.compile(
        r"(?<=[,;])\s+|(?<=\s)(?=\b(?:and|but|or|so|because|if|when|while|although|since|after|before|unless|until|where|whereas|whether|as|though)\b)"
    )

    parts = re.split(split_pattern, sentence)
    parts = [part.strip() for part in parts if part.strip()]

    # Further split parts if they are still too long
    final_parts = []
    for part in parts:
        formatted_part = split_subtitle(part, max_chars=max_line_length)
        num_lines_part = len(formatted_part.split("\n"))
        if num_lines_part > max_lines:
            # Split long parts at spaces
            words = part.split()
            mid_point = len(words) // 2
            part1 = " ".join(words[:mid_point])
            part2 = " ".join(words[mid_point:])
            final_parts.extend([part1.strip(), part2.strip()])
        else:
            final_parts.append(part)

    return final_parts


def split_at_sentence_end(
    segmenter: pysbd.Segmenter | None, text: str, word_data: list[Word]
) -> list[Cue]:

    sentences = []
    if segmenter is not None:
        sentences = segmenter.segment(text)
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    result: list[Cue] = []
    current_word_index = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            clause_splits = split_sentence_heuristically(
                sentence, max_line_length=42, max_lines=2
            )
            for clause in clause_splits:
                clause = clause.strip()
                if clause:
                    clause_word_count = len(clause.split())
                    end = current_word_index + clause_word_count
                    clause_word_data = word_data[current_word_index:end]
                    if clause_word_data:
                        start_time = next(
                            (
                                word["start"]
                                for word in clause_word_data
                                if "start" in word
                            ),
                            None,
                        )
                        end_time = next(
                            (
                                word["end"]
                                for word in reversed(clause_word_data)
                                if "end" in word
                            ),
                            None,
                        )
                        if start_time is not None and end_time is not None:
                            result.append(
                                {
                                    "text": clause,
                                    "start": start_time,
                                    "end": end_time,
                                    "word_data": clause_word_data,
                                }
                            )
                        else:
                            # Handle missing start or end times
                            if result:
                                prev_end = result[-1]["end"]
                                result.append(
                                    {
                                        "text": clause,
                                        "start": prev_end,
                                        "end": prev_end + 1,
                                        "word_data": None,
                                    }
                                )
                            else:
                                result.append(
                                    {
                                        "text": clause,
                                        "start": 0,
                                        "end": 1,
                                        "word_data": None,
                                    }
                                )
                    current_word_index += clause_word_count
    return result


def merge_short_cues(
    cues: list[Cue],
    min_duration=3,
    max_line_length=42,
    max_lines=2,
    desired_wps=DESIRED_WPS,
) -> list[Cue]:
    merged_cues: list[Cue] = []
    current_cue: Cue | None = None

    for cue in cues:
        if current_cue is None:
            current_cue = cue
        else:
            # Calculate combined text and duration
            combined_text = current_cue["text"] + " " + cue["text"]
            combined_word_count = len(combined_text.split())
            combined_start = current_cue["start"]
            combined_end = cue["end"]
            combined_duration = combined_end - combined_start

            # Determine optimal duration based on desired reading speed
            optimal_duration = combined_word_count / desired_wps

            # Use split_subtitle to check formatting constraints
            split_lines = split_subtitle(
                combined_text, max_chars=max_line_length
            ).split("\n")
            num_lines = len(split_lines)

            # Decide whether to merge based on duration and formatting constraints
            if (
                combined_duration < min_duration or combined_duration < optimal_duration
            ) and num_lines <= max_lines:
                # Merge the cues
                current_cue["text"] = combined_text
                current_cue["end"] = combined_end
            else:
                # Adjust duration of current cue if needed
                current_word_count = len(current_cue["text"].split())
                current_duration = current_cue["end"] - current_cue["start"]
                optimal_current_duration = current_word_count / desired_wps
                if (
                    current_duration < min_duration
                    or current_duration < optimal_current_duration
                ):
                    current_cue["end"] = min(
                        current_cue["start"]
                        + max(optimal_current_duration, min_duration),
                        cue["start"] - 0.1,
                    )
                merged_cues.append(current_cue)
                current_cue = cue

    if current_cue:
        # Adjust duration of the last cue if needed
        current_word_count = len(current_cue["text"].split())
        current_duration = current_cue["end"] - current_cue["start"]
        optimal_current_duration = current_word_count / desired_wps
        if (
            current_duration < min_duration
            or current_duration < optimal_current_duration
        ):
            current_cue["end"] = current_cue["start"] + max(
                optimal_current_duration, min_duration
            )
        merged_cues.append(current_cue)

    return merged_cues


def split_long_cue_without_word_timings(
    cue: Cue, max_line_length=42, max_lines=2
) -> list[Cue]:
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
    new_cues: list[Cue] = []
    for chunk in chunks:
        chunk_text_length = len(chunk.replace("\n", " "))
        proportion = (
            chunk_text_length / total_text_length if total_text_length > 0 else 0
        )
        chunk_duration = total_duration * proportion if total_duration > 0 else 0
        chunk_end_time = start_time + chunk_duration
        new_cues.append(
            {
                "text": chunk,
                "start": start_time,
                "end": chunk_end_time,
                "word_data": None,
            }
        )
        start_time = chunk_end_time  # Next chunk starts here
    return new_cues


def split_long_cues_with_word_timings(
    cues: list[Cue],
    max_line_length=42,
    max_lines=2,
    min_duration=5.0 / 6.0,
    desired_wps=DESIRED_WPS,
    max_gap_duration=1.5,  # Maximum acceptable time gap between chunks for merging
) -> list[Cue]:
    new_cues: list[Cue] = []
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

        # Chunk the cue based on max_line_length and max_lines
        chunks = []
        current_chunk_words = []
        current_chunk_timings = []

        for word, word_timing in zip(words, word_timings, strict=True):
            # Tentatively add the word to the current chunk
            temp_chunk_words = current_chunk_words + [word]
            temp_chunk_text = " ".join(temp_chunk_words)
            temp_formatted_text = split_subtitle(
                temp_chunk_text, max_chars=max_line_length
            )
            num_lines = len(temp_formatted_text.split("\n"))

            if num_lines > max_lines and current_chunk_words:
                # Adding this word exceeds max_lines, so finalize the current chunk
                chunks.append(
                    {
                        "words": current_chunk_words.copy(),
                        "timings": current_chunk_timings.copy(),
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
            chunks.append(
                {
                    "words": current_chunk_words.copy(),
                    "timings": current_chunk_timings.copy(),
                }
            )

        # Process chunks to create new cues with duration adjustments
        for i, chunk in enumerate(chunks):
            chunk_words = chunk["words"]
            chunk_word_timings = chunk["timings"]
            chunk_text = " ".join(chunk_words)
            # chunk_formatted_text = split_subtitle(chunk_text, max_chars=max_line_length)
            # num_lines = len(chunk_formatted_text.split("\n"))
            # # If the chunk still exceeds max_lines, handle accordingly
            # if num_lines > max_lines:
            #     # Optional: Implement recursive splitting or accept that this chunk exceeds max_lines
            #     # For now, we'll proceed without further splitting
            #     pass

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

            # Calculate speech rate
            chunk_word_count = len(chunk_words)
            speech_rate_wps = (
                chunk_word_count / duration if duration > 0 else float("inf")
            )

            # Determine optimal duration based on desired reading speed
            optimal_duration = chunk_word_count / desired_wps

            # Ensure duration is at least min_duration
            if duration < min_duration:
                # Try to merge with the next chunk
                if i + 1 < len(chunks):
                    # Merge with next chunk
                    next_chunk = chunks[i + 1]
                    next_chunk_start_time = next_chunk["timings"][0].get(
                        "start", cue["end"]
                    )
                    time_gap = next_chunk_start_time - end_time

                    if time_gap <= max_gap_duration:
                        merged_words = chunk_words + next_chunk["words"]
                        merged_timings = chunk_word_timings + next_chunk["timings"]
                        merged_text = " ".join(merged_words)
                        merged_formatted_text = split_subtitle(
                            merged_text, max_chars=max_line_length
                        )
                        num_lines = len(merged_formatted_text.split("\n"))

                        # Check if merged cue respects formatting constraints
                        if (
                            num_lines <= max_lines + 1
                        ):  # Allow one extra line for merging
                            # Update the next chunk with merged data
                            chunks[i + 1] = {
                                "words": merged_words,
                                "timings": merged_timings,
                            }
                            # print(
                            #     f"Merged cues: '{chunk_text}' + '{next_chunk['words']}'"
                            # )
                            continue  # Skip adding current chunk, as it's merged
                # Else, try to merge with the previous chunk
                elif new_cues:
                    prev_cue = new_cues[-1]
                    prev_cue_end_time = prev_cue["end"]
                    time_gap = start_time - prev_cue_end_time

                    if time_gap <= max_gap_duration:
                        merged_text = prev_cue["text"] + " " + chunk_text
                        merged_formatted_text = split_subtitle(
                            merged_text, max_chars=max_line_length
                        )
                        num_lines = len(merged_formatted_text.split("\n"))
                        if (
                            num_lines <= max_lines + 1
                        ):  # Allow one extra line for merging
                            # Update previous cue with merged data
                            prev_cue["text"] = merged_text
                            prev_cue["end"] = end_time
                            # Handle 'word_data'
                            prev_word_data = prev_cue.get("word_data", [])
                            prev_cue["word_data"] = prev_word_data + chunk_word_timings
                            # print(f"Merged cues: '{prev_cue['text']}' + '{chunk_text}'")
                            # print("end_time", end_time)
                            continue
                # If cannot merge, proceed with current chunk
                print(f"Cue '{chunk_text}' has short duration ({duration}s)")

            # Adjust duration based on speech rate
            adjusted_end_time = end_time
            if speech_rate_wps > desired_wps:
                # Speech is faster than desired reading speed; increase duration
                adjusted_duration = max(duration, optimal_duration)
                adjusted_end_time = start_time + adjusted_duration
                # Ensure we do not overlap with next chunk or cue's end
                next_start_time = (
                    chunks[i + 1]["timings"][0].get("start", cue["end"])
                    if i + 1 < len(chunks)
                    else cue["end"]
                )
                if adjusted_end_time > next_start_time:
                    adjusted_end_time = min(next_start_time - 0.1, adjusted_end_time)
            elif speech_rate_wps < desired_wps and duration > optimal_duration:
                # Speech is slower than desired reading speed; decrease duration
                adjusted_duration = max(optimal_duration, min_duration)
                adjusted_end_time = start_time + adjusted_duration
                if adjusted_end_time < end_time:
                    # We should not shorten the duration below the current duration
                    adjusted_end_time = end_time

            # Ensure duration is at least min_duration
            if adjusted_end_time - start_time < min_duration:
                adjusted_end_time = start_time + min_duration

            # Update the cue
            new_cues.append(
                {
                    "text": chunk_text,
                    "start": start_time,
                    "end": adjusted_end_time,
                    "word_data": chunk_word_timings,
                }
            )

    # Adjust durations of new_cues based on speech rate
    adjusted_cues = []
    for i, cue in enumerate(new_cues):
        text = cue["text"]
        start_time = cue["start"]
        end_time = cue["end"]
        duration = end_time - start_time
        word_count = len(text.split())
        optimal_duration = word_count / desired_wps

        # Adjust duration based on speech rate
        adjusted_end_time = end_time
        if duration < optimal_duration:
            adjusted_end_time = start_time + optimal_duration
            # Ensure we do not overlap with the next cue
            next_start_time = (
                new_cues[i + 1]["start"] if i + 1 < len(new_cues) else cue["end"]
            )
            if adjusted_end_time > next_start_time:
                adjusted_end_time = min(next_start_time - 0.01, adjusted_end_time)

        # Ensure duration is at least min_duration
        if adjusted_end_time - start_time < min_duration:
            adjusted_end_time = start_time + min_duration
            # Ensure we do not overlap with the next cue
            next_start_time = (
                new_cues[i + 1]["start"] if i + 1 < len(new_cues) else cue["end"]
            )
            if adjusted_end_time > next_start_time:
                adjusted_end_time = min(next_start_time - 0.01, adjusted_end_time)
            # If still less than start_time, accept the shorter duration
            if adjusted_end_time <= start_time:
                adjusted_end_time = start_time + (next_start_time - start_time) / 2

        # Update cue's end time
        cue["end"] = adjusted_end_time
        adjusted_cues.append(cue)

    return adjusted_cues
