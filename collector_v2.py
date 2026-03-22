"""
collector_v2.py — Enhanced ASL Training Data Collector
═══════════════════════════════════════════════════════
Interactive collector with friendly prompts for:
  1. LETTER COLLECTION: All 26 letters × 2 hands (52 total recordings)
  2. WORD COLLECTION: Custom words × 5 repetitions each (max 10 words)
  3. MANUAL COLLECTION: Original key-press mode for advanced users

Features:
  • Press SPACE to record check before capturing
  • 3-second countdown timer before each recording
  • Auto-progression through collection sequence
  • Clear on-screen guidance and progress tracking
  • Resume from interrupted sessions

Output:  asl_data.csv  (appends videos to existing file)
         Features include upper-body pose, both hands, and head anchors.
"""

import cv2
import numpy as np
import csv
import os
import time
import argparse
from collections import defaultdict
from typing import List, Optional

from feature_utils import (
    compute_velocity,
    create_holistic_model,
    draw_overlays,
    extract_position_features,
    feature_size,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "hand_tracking"
    else SCRIPT_DIR
)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "asl_data.csv")
SAMPLE_RATE = 10  # samples saved per second while recording
RECORDING_DURATION = 3  # seconds to record each sample
FEATURE_SIZE = feature_size()

os.makedirs(DATA_DIR, exist_ok=True)

# Load existing data
sample_counts = defaultdict(int)
existing_rows = 0
csv_format_compatible = True

if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) != (FEATURE_SIZE * 2) + 1:
            print("⚠️  WARNING: existing CSV has wrong number of columns.")
            print(f"   Expected {(FEATURE_SIZE * 2)} feature columns.")
            print(f"   Found {len(header) - 1} feature columns.")
            print("   Delete/rename the existing file and re-collect.")
            csv_format_compatible = False
        else:
            for row in reader:
                if row:
                    sample_counts[row[-1]] += 1
                    existing_rows += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced ASL Training Data Collector")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g., 0) or path to video file.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["letters", "words", "manual"],
        help="Collection mode: letters, words, or manual.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=SAMPLE_RATE,
        help="Samples per second (default: 10)",
    )
    parser.add_argument(
        "--words",
        default=None,
        help="Comma-separated words for word mode (e.g. --words hello,thanks,help).",
    )
    return parser.parse_args()


def resolve_source(source: str):
    return int(source) if source.isdigit() else source


def clear_screen():
    """Clear console screen"""
    os.system("cls" if os.name == "nt" else "clear")


def show_menu() -> str:
    """Interactive mode selection menu"""
    clear_screen()
    print("╔════════════════════════════════════════════════╗")
    print("║   ASL DATA COLLECTOR - Enhanced Version        ║")
    print("╚════════════════════════════════════════════════╝\n")
    print("Select collection mode:\n")
    print("  [L] Letter Collection   - All 26 letters × 2 hands")
    print("  [W] Word Collection     - Custom words (up to 10)")
    print("  [M] Manual Collection   - Original key-press mode")
    print("  [Q] Quit\n")
    
    while True:
        choice = input("Enter choice (L/W/M/Q): ").strip().upper()
        if choice in ["L", "W", "M", "Q"]:
            return choice
        print("Invalid choice. Please enter L, W, M, or Q.")


def parse_words_arg(words_arg: str | None, max_words: int = 10) -> List[str]:
    """Parse comma-separated words argument into a normalized unique list."""
    if not words_arg:
        return []

    words: List[str] = []
    for raw_word in words_arg.split(","):
        word = raw_word.strip().lower()
        if not word:
            continue
        if word not in words:
            words.append(word)
        if len(words) >= max_words:
            break
    return words


def get_words_from_user(max_words: int = 10) -> List[str]:
    """Get word list from user for word collection mode"""
    while True:
        clear_screen()
        print("╔════════════════════════════════════════════════╗")
        print("║         WORD COLLECTION - Setup                ║")
        print("╚════════════════════════════════════════════════╝\n")
        print(f"Enter words to collect (max {max_words}).")
        print("Each word will be recorded 5 times.\n")

        words = []
        while len(words) < max_words:
            try:
                prompt = f"Word {len(words) + 1}/{max_words} (or press Enter to start): "
                word = input(prompt).strip().lower()
            except EOFError:
                print("\nInput stream unavailable in this terminal/session.")
                print("Use --words, for example: --mode words --words hello,thanks,help")
                return []

            if not word:
                break
            if word not in words:
                words.append(word)
                print(f"  ✓ Added '{word}'")
            else:
                print(f"  ✗ '{word}' already in list")

        if words:
            return words

        print("No words entered. Press Enter to try again, or Ctrl+C to cancel.")
        try:
            input()
        except EOFError:
            print("Input stream unavailable. Exiting word setup.")
            return []


def draw_timer(frame, elapsed: float, duration: float = 3.0):
    """Draw countdown timer on frame"""
    remaining = max(0, duration - elapsed)
    if remaining > 0:
        progress = 1 - (remaining / duration)
        bar_width = int(progress * 400)
        
        # Draw timer number (large)
        timer_text = f"{remaining:.1f}"
        cv2.putText(
            frame,
            timer_text,
            (frame.shape[1] // 2 - 60, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.5,
            (50, 200, 50),
            4,
            cv2.LINE_AA,
        )
        
        # Draw progress bar
        cv2.rectangle(frame, (frame.shape[1] // 2 - 200, 150), 
                      (frame.shape[1] // 2 + 200, 180), (100, 100, 100), 1)
        cv2.rectangle(frame, (frame.shape[1] // 2 - 200, 150),
                      (frame.shape[1] // 2 - 200 + bar_width, 180), (50, 200, 50), -1)
        
        cv2.putText(frame, "Recording in progress...",
                    (frame.shape[1] // 2 - 150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)


def draw_waiting_for_space(frame, label: str, recording_num: int = None, total_recordings: int = None):
    """Draw 'Press SPACE to record' prompt"""
    # Main prompt
    cv2.putText(
        frame,
        "Press SPACE to record",
        (frame.shape[1] // 2 - 200, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (100, 150, 255),
        2,
        cv2.LINE_AA,
    )
    
    # Label (what to sign)
    label_text = f"Sign: {label.upper()}"
    cv2.putText(
        frame,
        label_text,
        (frame.shape[1] // 2 - 150, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (200, 100, 255),
        3,
        cv2.LINE_AA,
    )
    
    # Progress info if provided
    if recording_num and total_recordings:
        progress_text = f"Recording {recording_num}/{total_recordings}"
        cv2.putText(
            frame,
            progress_text,
            (frame.shape[1] // 2 - 150, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (150, 150, 150),
            1,
        )
    
    # Press ESC to cancel info
    cv2.putText(
        frame,
        "Press ESC to skip | Q to quit",
        (frame.shape[1] // 2 - 180, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )


def record_sample(
    cap,
    holistic,
    csv_writer,
    csv_file,
    label: str,
    sample_rate: float,
    duration: float = RECORDING_DURATION,
) -> int:
    """Record a single sample for the given duration and label"""
    capture_interval = 1.0 / max(sample_rate, 1e-6)
    last_capture_time = 0
    prev_pos = [0.0] * FEATURE_SIZE
    samples_recorded = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        
        # Extract features
        pos = extract_position_features(result)
        velocity = compute_velocity(pos, prev_pos)
        prev_pos = pos
        
        draw_overlays(frame, result)
        
        # Draw timer
        elapsed = time.time() - start_time
        draw_timer(frame, elapsed, duration)
        
        # Save if hand detected
        hand_found = bool(result.left_hand_landmarks or result.right_hand_landmarks)
        now = time.time()
        if hand_found and (now - last_capture_time) >= capture_interval:
            csv_writer.writerow(pos + velocity + [label])
            csv_file.flush()
            last_capture_time = now
            samples_recorded += 1
            sample_counts[label] += 1
        
        # Check if recording time is done
        if elapsed >= duration:
            break
        
        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return -1  # quit signal
    
    return samples_recorded


def letter_collection_mode(cap, holistic, csv_writer, csv_file, sample_rate: float):
    """Interactive letter collection for all 26 letters × 2 hands"""
    print("\n" + "=" * 50)
    print("LETTER COLLECTION MODE")
    print("=" * 50)
    print("You will collect all 26 letters, twice for each hand.")
    print("Total recordings: 52 (26 letters × right hand + 26 × left hand)")
    print("=" * 50 + "\n")
    
    letters = list("abcdefghijklmnopqrstuvwxyz")
    hands = ["Right Hand", "Left Hand"]
    total_recordings = len(letters) * len(hands)
    recording_count = 0
    
    prev_pos = [0.0] * FEATURE_SIZE
    
    for hand in hands:
        print(f"\n📋 Now collecting {hand.upper()}:")
        
        for letter_idx, letter in enumerate(letters):
            recording_count += 1
            progress = f"[{recording_count}/{total_recordings}]"
            print(f"  {progress} Preparing {letter.upper()} ({hand})...", end="", flush=True)
            
            # Wait for SPACE key to start recording
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    return
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                draw_overlays(frame, result)
                
                draw_waiting_for_space(
                    frame,
                    f"{letter} - {hand}",
                    recording_count,
                    total_recordings
                )
                
                cv2.imshow("ASL Data Collector", frame)
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord(" "):  # SPACE bar
                    waiting = False
                    print("\n    ⏱️  Recording...", end="", flush=True)
                elif key == 27:  # ESC
                    waiting = False
                    print("\n    ⊘ Skipped")
                    break
                elif key == ord("q"):
                    print("\n❌ Quit requested")
                    return
            
            if key == ord(" "):  # Only record if SPACE was pressed (not ESC)
                # Show countdown overlay before recording
                for countdown in [3, 2, 1]:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)
                    draw_overlays(frame, result)
                    
                    cv2.putText(
                        frame,
                        str(countdown),
                        (frame.shape[1] // 2 - 50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.0,
                        (255, 150, 50),
                        5,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("ASL Data Collector", frame)
                    cv2.waitKey(1000)
                
                # Record the sample
                result = record_sample(cap, holistic, csv_writer, csv_file, letter, sample_rate)
                if result == -1:
                    return  # quit signal
                print(f" ✓ Recorded {result} frames")
    
    print("\n✅ Letter collection complete!")
    print(f"   Total letters collected: {len(letters) * len(hands)}")


def word_collection_mode(
    cap,
    holistic,
    csv_writer,
    csv_file,
    sample_rate: float,
    preset_words: List[str] | None = None,
):
    """Interactive word collection - custom words × 5 repetitions"""
    words = preset_words if preset_words else get_words_from_user(max_words=10)
    if not words:
        print("No words provided. Use --words or enter words interactively.")
        return
    
    print("\n" + "=" * 50)
    print("WORD COLLECTION MODE")
    print("=" * 50)
    print(f"Collecting {len(words)} word(s), 5 repetitions each.")
    print(f"Total recordings: {len(words) * 5}")
    print("=" * 50 + "\n")
    
    total_recordings = len(words) * 5
    recording_count = 0
    
    for word_idx, word in enumerate(words, 1):
        print(f"\n📝 Word {word_idx}/{len(words)}: '{word.upper()}'")
        
        for rep in range(1, 6):
            recording_count += 1
            progress = f"[{recording_count}/{total_recordings}]"
            print(f"  {progress} Repetition {rep}/5...", end="", flush=True)
            
            # Wait for SPACE
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    return
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                draw_overlays(frame, result)
                
                draw_waiting_for_space(
                    frame,
                    word,
                    recording_count,
                    total_recordings
                )
                
                cv2.imshow("ASL Data Collector", frame)
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord(" "):
                    waiting = False
                    print("\n    ⏱️  Recording...", end="", flush=True)
                elif key == 27:  # ESC
                    waiting = False
                    print("\n    ⊘ Skipped")
                    break
                elif key == ord("q"):
                    print("\n❌ Quit requested")
                    return
            
            if key == ord(" "):
                # Countdown
                for countdown in [3, 2, 1]:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)
                    draw_overlays(frame, result)
                    
                    cv2.putText(
                        frame,
                        str(countdown),
                        (frame.shape[1] // 2 - 50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.0,
                        (255, 150, 50),
                        5,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("ASL Data Collector", frame)
                    cv2.waitKey(1000)
                
                # Record
                result = record_sample(cap, holistic, csv_writer, csv_file, word, sample_rate)
                if result == -1:
                    return
                print(f" ✓ Recorded {result} frames")
    
    print("\n✅ Word collection complete!")
    print(f"   Words collected: {len(words)}")
    print(f"   Total repetitions: {len(words) * 5}")


def manual_collection_mode(cap, holistic, csv_writer, csv_file, sample_rate: float):
    """Original manual key-press collection mode"""
    print("\n" + "=" * 50)
    print("MANUAL COLLECTION MODE")
    print("=" * 50)
    print("Original mode - hold letter key to record")
    print("Press Q to quit\n")
    
    capture_interval = 1.0 / max(sample_rate, 1e-6)
    last_capture_time = 0
    session_counts = defaultdict(int)
    prev_pos = [0.0] * FEATURE_SIZE
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        
        key = cv2.waitKey(1) & 0xFF
        active_label = None
        
        if ord("a") <= key <= ord("z"):
            active_label = chr(key)
        if ord("0") <= key <= ord("9"):
            active_label = chr(key)
        
        hand_found = bool(result.left_hand_landmarks or result.right_hand_landmarks)
        pos = extract_position_features(result)
        velocity = compute_velocity(pos, prev_pos)
        prev_pos = pos
        
        draw_overlays(frame, result)
        
        vel_mag = np.linalg.norm(velocity)
        vel_color = (0, 200, 255) if vel_mag > 0.05 else (80, 80, 80)
        cv2.putText(
            frame,
            f"motion: {vel_mag:.3f}",
            (frame.shape[1] - 200, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            vel_color,
            1,
            cv2.LINE_AA,
        )
        
        now = time.time()
        if active_label and hand_found and (now - last_capture_time) >= capture_interval:
            csv_writer.writerow(pos + velocity + [active_label])
            csv_file.flush()
            last_capture_time = now
            session_counts[active_label] += 1
            sample_counts[active_label] += 1
        
        if active_label:
            color = (0, 80, 255) if hand_found else (60, 60, 200)
            indicator = ("SAVING: " + f"'{active_label.upper()}'" if hand_found 
                        else f"NO HAND — '{active_label.upper()}'")
            cv2.putText(frame, indicator, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Hold a letter key to capture",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (160, 160, 160), 1, cv2.LINE_AA)
        
        y = 100
        cv2.putText(frame, "Samples this session:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y += 22
        for letter in sorted(session_counts):
            total = sample_counts[letter]
            sess = session_counts[letter]
            bar = "█" * min(sess // 5, 20)
            text = f"  {letter.upper()}: {total:4d} total  (+{sess}) {bar}"
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 160), 1)
            y += 18
        
        total_all = sum(sample_counts.values())
        cv2.putText(frame, f"Total saved: {total_all}",
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        
        cv2.imshow("ASL Data Collector", frame)
        if key == ord("q"):
            break
    
    print("\n✅ Manual collection complete!")


def main():
    args = parse_args()
    preset_words = parse_words_arg(args.words, max_words=10)

    if not csv_format_compatible:
        raise RuntimeError(
            "Incompatible existing dataset format at "
            f"{OUTPUT_CSV}. Rename or delete it before collecting with collector_v2.py"
        )
    
    # Open CSV file
    csv_file = open(OUTPUT_CSV, "a", newline="")
    csv_writer = csv.writer(csv_file)
    
    # Write header only for brand-new files
    if existing_rows == 0:
        pos_header = [f"f{i}" for i in range(FEATURE_SIZE)]
        vel_header = [f"df{i}" for i in range(FEATURE_SIZE)]
        csv_writer.writerow(pos_header + vel_header + ["label"])
    
    print(f"\n✓ Resuming session — {existing_rows} existing samples loaded")
    print(f"✓ Output file: {OUTPUT_CSV}\n")
    
    # Determine mode
    if args.mode:
        mode = args.mode.upper()
        mode_map = {"LETTERS": "L", "WORDS": "W", "MANUAL": "M"}
        mode = mode_map.get(mode, mode)
    else:
        mode = show_menu()
    
    if mode == "Q":
        print("\n👋 Goodbye!")
        csv_file.close()
        return
    
    # Create holistic model and open camera
    with create_holistic_model() as holistic:
        cap = cv2.VideoCapture(resolve_source(str(args.source)))
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            csv_file.close()
            return
        
        try:
            if mode == "L":
                letter_collection_mode(cap, holistic, csv_writer, csv_file, args.sample_rate)
            elif mode == "W":
                word_collection_mode(
                    cap,
                    holistic,
                    csv_writer,
                    csv_file,
                    args.sample_rate,
                    preset_words=preset_words,
                )
            elif mode == "M":
                manual_collection_mode(cap, holistic, csv_writer, csv_file, args.sample_rate)
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    csv_file.close()
    
    # Summary
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    total_samples = sum(sample_counts.values())
    print(f"Total samples in file: {total_samples}\n")
    
    print("Sample breakdown:")
    for label in sorted(sample_counts.keys()):
        count = sample_counts[label]
        bar = "█" * min(count // 50, 20)
        print(f"  {label.upper():6s}: {count:5d} samples  {bar}")
    
    print("\n" + "=" * 50)
    print("Collection saved to: " + OUTPUT_CSV)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
