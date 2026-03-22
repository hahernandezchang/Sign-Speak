import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from feature_utils import (
    compute_velocity,
    create_holistic_model,
    draw_overlays,
    extract_legacy_hand_position_features,
    extract_position_features,
    feature_size,
)
from model_utils import LoadedClassifier, load_classifier, predict_label


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "hand_tracking"
    else SCRIPT_DIR
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
FEATURE_SIZE = feature_size()
EXPECTED_INPUT_SIZE = FEATURE_SIZE * 2
LEGACY_INPUT_SIZE = 126

from voice_engine import VoiceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL runtime with selectable model mode: letters, words, or hybrid."
    )
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument(
        "--mode",
        choices=["letters", "words", "hybrid"],
        default="letters",
        help="Runtime mode: letters-only, words-only, or hybrid fallback.",
    )

    parser.add_argument(
        "--word-model-path",
        default=os.path.join(MODEL_DIR, "asl_words_classifier.pt"),
    )
    parser.add_argument(
        "--word-label-map-path",
        default=os.path.join(MODEL_DIR, "asl_words_label_map.json"),
    )

    parser.add_argument(
        "--alpha-model-path",
        default=os.path.join(MODEL_DIR, "asl_alphabet_classifier.pt"),
    )
    parser.add_argument(
        "--alpha-label-map-path",
        default=os.path.join(MODEL_DIR, "asl_alphabet_label_map.json"),
    )

    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice mode (press V to speak current buffer).",
    )
    parser.add_argument(
        "--word-threshold",
        type=float,
        default=0.70,
        help="Minimum confidence for choosing a word prediction.",
    )
    parser.add_argument(
        "--letter-threshold",
        type=float,
        default=0.70,
        help="Minimum confidence for choosing/displaying a letter prediction.",
    )
    parser.add_argument(
        "--append-threshold",
        type=float,
        default=0.85,
        help="Minimum confidence to append word tokens into phrase buffer.",
    )
    parser.add_argument(
        "--append-cooldown",
        type=float,
        default=1.2,
        help="Seconds between repeated appended word tokens.",
    )
    parser.add_argument(
        "--word-stable-frames",
        type=int,
        default=8,
        help="Frames a word must stay stable before appending to phrase buffer.",
    )
    parser.add_argument(
        "--word-commit-seconds",
        type=float,
        default=0.65,
        help="Seconds a stable word must persist before append.",
    )
    parser.add_argument(
        "--word-reset-seconds",
        type=float,
        default=0.35,
        help="Seconds without confident word before pending word reset.",
    )
    parser.add_argument(
        "--word-gap-seconds",
        type=float,
        default=0.7,
        help="Quiet gap needed before same word can be appended again.",
    )
    parser.add_argument(
        "--buffer-max",
        type=int,
        default=12,
        help="Maximum word tokens kept in phrase buffer.",
    )
    parser.add_argument(
        "--spell-threshold",
        type=float,
        default=0.75,
        help="Minimum alphabet confidence to consider letter spelling.",
    )
    parser.add_argument(
        "--spell-stable-frames",
        type=int,
        default=5,
        help="Frames a letter must stay stable before committing.",
    )
    parser.add_argument(
        "--spell-gap-seconds",
        type=float,
        default=0.40,
        help="Time without confident spelling before same-letter repeat unlock.",
    )
    parser.add_argument(
        "--spell-commit-seconds",
        type=float,
        default=0.70,
        help="How long a stable letter must persist before commit.",
    )
    parser.add_argument(
        "--spell-reset-seconds",
        type=float,
        default=0.25,
        help="How long candidate can disappear before reset.",
    )
    return parser.parse_args()


def resolve_source(source: str):
    return int(source) if source.isdigit() else source


def try_load(model_path: str, label_map_path: str) -> LoadedClassifier | None:
    if os.path.exists(model_path) and os.path.exists(label_map_path):
        clf = load_classifier(model_path, label_map_path)
        print(f"Loaded {len(clf.label_map)} labels from {os.path.basename(model_path)}")
        return clf
    return None


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def draw_motion_meter(frame, velocity: list[float]) -> None:
    mag = float(np.linalg.norm(velocity))
    bar_w = int(min(mag * 120, 220))
    color = (0, 200, 255) if mag > 0.05 else (60, 60, 60)
    bx, by = frame.shape[1] - 240, frame.shape[0] - 30
    cv2.rectangle(frame, (bx, by), (bx + 220, by + 14), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 14), color, -1)
    cv2.putText(
        frame,
        f"motion {mag:.3f}",
        (bx, by - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def validate_classifier_input(name: str, classifier: LoadedClassifier | None) -> LoadedClassifier | None:
    if classifier is None:
        return None
    if classifier.input_size == LEGACY_INPUT_SIZE:
        print(f"{name} model uses legacy 126-feature format. Compatibility mode enabled.")
        return classifier
    if classifier.input_size != EXPECTED_INPUT_SIZE:
        print(
            f"{name} model input mismatch: checkpoint expects {classifier.input_size}, "
            f"runtime produces {EXPECTED_INPUT_SIZE}."
        )
        return None
    return classifier


def main() -> None:
    args = parse_args()

    wants_letters = args.mode in ("letters", "hybrid")
    wants_words = args.mode in ("words", "hybrid")

    word_model = None
    if wants_words:
        word_model_path = first_existing(
            [
                args.word_model_path,
                os.path.join(MODEL_DIR, "asl_classifier.pt"),
            ]
        )
        word_label_map_path = first_existing(
            [
                args.word_label_map_path,
                os.path.join(MODEL_DIR, "asl_label_map.json"),
            ]
        )
        word_model = (
            try_load(word_model_path, word_label_map_path)
            if word_model_path is not None and word_label_map_path is not None
            else None
        )
        word_model = validate_classifier_input("Word", word_model)

        if word_model is None:
            checked = [
                args.word_model_path,
                args.word_label_map_path,
                os.path.join(MODEL_DIR, "asl_classifier.pt"),
                os.path.join(MODEL_DIR, "asl_label_map.json"),
            ]
            checked_text = "\n".join([f" - {path}" for path in checked])
            raise RuntimeError(
                "No usable word model found.\n"
                f"Expected feature size is {EXPECTED_INPUT_SIZE} (holistic position + velocity).\n"
                "Checkpoint may be from old feature pipeline.\n"
                f"Checked paths:\n{checked_text}"
            )

    alpha_model = None
    if wants_letters:
        alpha_model_path = first_existing(
            [
                args.alpha_model_path,
                os.path.join(MODEL_DIR, "asl_classifier.pt"),
            ]
        )
        alpha_label_map_path = first_existing(
            [
                args.alpha_label_map_path,
                os.path.join(MODEL_DIR, "asl_label_map.json"),
            ]
        )

        alpha_model = (
            try_load(alpha_model_path, alpha_label_map_path)
            if alpha_model_path is not None and alpha_label_map_path is not None
            else None
        )
        alpha_model = validate_classifier_input("Alphabet", alpha_model)

        if alpha_model is None:
            checked = [
                args.alpha_model_path,
                args.alpha_label_map_path,
                os.path.join(MODEL_DIR, "asl_classifier.pt"),
                os.path.join(MODEL_DIR, "asl_label_map.json"),
            ]
            checked_text = "\n".join([f" - {path}" for path in checked])
            raise RuntimeError(
                "No usable alphabet model found.\n"
                f"Expected feature size is {EXPECTED_INPUT_SIZE} (holistic position + velocity).\n"
                "Checkpoint may be from old feature pipeline.\n"
                f"Checked paths:\n{checked_text}"
            )

    voice = VoiceEngine()
    voice_enabled = args.voice and voice.enabled
    if args.voice and not voice.enabled:
        print("Voice mode requested but API keys are missing. Continuing without speech.")

    spelled_text = ""
    pending_char = ""
    pending_char_frames = 0
    pending_char_started_at = 0.0
    pending_char_last_seen_at = 0.0
    repeat_unlocked = True
    last_confident_spell_time = 0.0

    speaking = False

    phrase_buffer: deque[str] = deque(maxlen=max(1, args.buffer_max))
    last_word_token = ""
    last_word_token_time = 0.0
    pending_word = ""
    pending_word_frames = 0
    pending_word_started_at = 0.0
    pending_word_last_seen_at = 0.0
    last_confident_word_time = 0.0
    word_repeat_unlocked = True

    prev_pos = [0.0] * FEATURE_SIZE
    prev_legacy_pos = [0.0] * 63
    print(f"Starting runtime mode={args.mode}. Press Q to quit.")

    with create_holistic_model() as holistic:
        cap = cv2.VideoCapture(resolve_source(str(args.source)))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            hand_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)

            position = extract_position_features(results)
            velocity = compute_velocity(position, prev_pos)
            prev_pos = position
            features = position + velocity

            legacy_position = extract_legacy_hand_position_features(results)
            legacy_velocity = compute_velocity(legacy_position, prev_legacy_pos)
            prev_legacy_pos = legacy_position
            legacy_features = legacy_position + legacy_velocity

            draw_overlays(frame, results)
            draw_motion_meter(frame, velocity)

            word_pred = None
            if wants_words and word_model is not None and hand_present:
                word_features = legacy_features if word_model.input_size == LEGACY_INPUT_SIZE else features
                word_pred = predict_label(word_model, word_features)

            alpha_pred = None
            if wants_letters and alpha_model is not None and hand_present:
                alpha_features = legacy_features if alpha_model.input_size == LEGACY_INPUT_SIZE else features
                alpha_pred = predict_label(alpha_model, alpha_features)

            chosen_label = None
            chosen_conf = 0.0
            chosen_mode = "none"

            if args.mode == "words":
                if word_pred is not None and word_pred[1] >= args.word_threshold:
                    chosen_label, chosen_conf = word_pred
                    chosen_mode = "WORD"
                elif word_pred is not None:
                    chosen_label, chosen_conf = word_pred
                    chosen_mode = "LOW-WORD"
            elif args.mode == "letters":
                if alpha_pred is not None and alpha_pred[1] >= args.letter_threshold:
                    chosen_label, chosen_conf = alpha_pred
                    chosen_mode = "LETTER"
                elif alpha_pred is not None:
                    chosen_label, chosen_conf = alpha_pred
                    chosen_mode = "LOW-LETTER"
            else:
                if word_pred is not None and word_pred[1] >= args.word_threshold:
                    chosen_label, chosen_conf = word_pred
                    chosen_mode = "WORD"
                elif alpha_pred is not None and alpha_pred[1] >= args.letter_threshold:
                    chosen_label, chosen_conf = alpha_pred
                    chosen_mode = "LETTER"
                elif alpha_pred is not None:
                    chosen_label, chosen_conf = alpha_pred
                    chosen_mode = "LOW-LETTER"
                elif word_pred is not None:
                    chosen_label, chosen_conf = word_pred
                    chosen_mode = "LOW-WORD"

            if chosen_label is not None:
                cv2.putText(
                    frame,
                    f"{chosen_mode}: {chosen_label}",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"conf {chosen_conf:.2f}",
                    (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            now = time.time()
            if not wants_letters:
                pending_char = ""
                pending_char_frames = 0
                pending_char_started_at = 0.0
                pending_char_last_seen_at = 0.0
            elif not hand_present:
                pending_char = ""
                pending_char_frames = 0
                pending_char_started_at = 0.0
                pending_char_last_seen_at = 0.0
            elif alpha_pred is not None and alpha_pred[1] >= args.spell_threshold:
                current_char = str(alpha_pred[0]).lower()
                if len(current_char) == 1 and current_char.isalpha():
                    last_confident_spell_time = now
                    if current_char == pending_char:
                        pending_char_frames += 1
                        pending_char_last_seen_at = now
                    else:
                        pending_char = current_char
                        pending_char_frames = 1
                        pending_char_started_at = now
                        pending_char_last_seen_at = now

                    stable_for_frames = pending_char_frames >= max(1, args.spell_stable_frames)
                    stable_for_time = (now - pending_char_started_at) >= max(0.0, args.spell_commit_seconds)
                    if stable_for_frames and stable_for_time:
                        if repeat_unlocked or current_char != (spelled_text[-1:] if spelled_text else ""):
                            spelled_text += current_char
                            repeat_unlocked = False
                            pending_char = ""
                            pending_char_frames = 0
                            pending_char_started_at = 0.0
                            pending_char_last_seen_at = 0.0
                            print(f"Text: {spelled_text}")
            else:
                if pending_char and (now - pending_char_last_seen_at) >= max(0.0, args.spell_reset_seconds):
                    pending_char = ""
                    pending_char_frames = 0
                    pending_char_started_at = 0.0
                    pending_char_last_seen_at = 0.0

            if (now - last_confident_spell_time) >= args.spell_gap_seconds:
                repeat_unlocked = True

            if wants_words and word_pred is not None and word_pred[1] >= args.append_threshold:
                current_word = str(word_pred[0]).strip().lower()
                last_confident_word_time = now

                if current_word == pending_word:
                    pending_word_frames += 1
                    pending_word_last_seen_at = now
                else:
                    pending_word = current_word
                    pending_word_frames = 1
                    pending_word_started_at = now
                    pending_word_last_seen_at = now

                stable_for_frames = pending_word_frames >= max(1, args.word_stable_frames)
                stable_for_time = (now - pending_word_started_at) >= max(0.0, args.word_commit_seconds)
                cooldown_ok = (now - last_word_token_time) >= max(0.0, args.append_cooldown)

                if stable_for_frames and stable_for_time:
                    if word_repeat_unlocked or current_word != last_word_token:
                        if current_word != last_word_token or cooldown_ok:
                            phrase_buffer.append(current_word)
                            last_word_token = current_word
                            last_word_token_time = now
                            word_repeat_unlocked = False
                            pending_word = ""
                            pending_word_frames = 0
                            pending_word_started_at = 0.0
                            pending_word_last_seen_at = 0.0
            else:
                if pending_word and (now - pending_word_last_seen_at) >= max(0.0, args.word_reset_seconds):
                    pending_word = ""
                    pending_word_frames = 0
                    pending_word_started_at = 0.0
                    pending_word_last_seen_at = 0.0

            if wants_words and (now - last_confident_word_time) >= max(0.0, args.word_gap_seconds):
                word_repeat_unlocked = True

            if word_pred is not None:
                cv2.putText(
                    frame,
                    f"word: {word_pred[0]} ({word_pred[1]:.2f})",
                    (30, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 220, 120),
                    1,
                    cv2.LINE_AA,
                )

            if alpha_pred is not None:
                alpha_tag = "letter" if alpha_pred[1] >= args.letter_threshold else "low-letter"
                cv2.putText(
                    frame,
                    f"{alpha_tag}: {alpha_pred[0]} ({alpha_pred[1]:.2f})",
                    (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 180, 120),
                    1,
                    cv2.LINE_AA,
                )

            if wants_words:
                phrase_text = " ".join(list(phrase_buffer)) if phrase_buffer else "(empty)"
                cv2.putText(
                    frame,
                    f"phrase: {phrase_text}",
                    (30, frame.shape[0] - 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                pending_word_text = "-"
                if pending_word:
                    elapsed = max(0.0, now - pending_word_started_at)
                    remaining = max(0.0, args.word_commit_seconds - elapsed)
                    pending_word_text = f"{pending_word} ({remaining:.2f}s)"
                cv2.putText(
                    frame,
                    f"word-candidate: {pending_word_text}",
                    (30, frame.shape[0] - 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 240, 200),
                    1,
                    cv2.LINE_AA,
                )

            if wants_letters:
                text_line = spelled_text if spelled_text else "(none)"
                cv2.putText(
                    frame,
                    f"text: {text_line}",
                    (30, frame.shape[0] - 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 230, 120),
                    1,
                    cv2.LINE_AA,
                )

            if wants_letters:
                candidate_text = "-"
                if pending_char:
                    elapsed = max(0.0, now - pending_char_started_at)
                    remaining = max(0.0, args.spell_commit_seconds - elapsed)
                    candidate_text = f"{pending_char} ({remaining:.2f}s)"
                cv2.putText(
                    frame,
                    f"candidate: {candidate_text}",
                    (30, frame.shape[0] - 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

            voice_state = "on" if voice_enabled else "off"
            key_hint = "V speak, Q quit"
            if args.mode == "letters":
                key_hint = "V speak, X clear text, BACKSPACE delete, SPACE gap, Q quit"
            elif args.mode == "words":
                key_hint = "V speak, C clear phrase, BACKSPACE delete word, Q quit"
            else:
                key_hint = "V speak, C clear phrase, X clear text, BACKSPACE delete (text/word), SPACE gap, Q quit"
            status_text = f"mode:{args.mode} voice:{voice_state} keys: {key_hint}"
            if speaking:
                status_text = "voice: speaking..."
            cv2.putText(
                frame,
                status_text,
                (30, frame.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 200, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Letters Runtime", frame)
            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key).lower() if 32 <= key <= 126 else ""
            if key_char == "q":
                break
            if key_char == "x" and wants_letters:
                spelled_text = ""
                pending_char = ""
                pending_char_frames = 0
                pending_char_started_at = 0.0
                pending_char_last_seen_at = 0.0
                repeat_unlocked = True
                print("Text buffer cleared.")
            if key in (8, 127) and wants_letters:
                if spelled_text:
                    spelled_text = spelled_text[:-1]
                    print(f"Text: {spelled_text}")
                elif wants_words and phrase_buffer:
                    removed = phrase_buffer.pop()
                    print(f"Removed word: {removed}")
            elif key in (8, 127) and wants_words and phrase_buffer:
                removed = phrase_buffer.pop()
                print(f"Removed word: {removed}")
            if key == ord(" ") and wants_letters:
                if spelled_text and not spelled_text.endswith(" "):
                    spelled_text += " "
                    repeat_unlocked = True
                    print(f"Text: {spelled_text}")
            if key_char == "c" and wants_words:
                phrase_buffer.clear()
                print("Phrase buffer cleared.")
            if key_char == "v":
                if not voice_enabled:
                    print("Voice mode is off. Restart with --voice to enable speech.")
                elif speaking:
                    print("Already speaking. Please wait.")
                else:
                    tokens_to_speak: list[str] = []
                    if wants_letters and spelled_text.strip():
                        words = [w for w in spelled_text.strip().split() if w]
                        tokens_to_speak.extend([" ".join(list(word.upper())) for word in words])
                    if wants_words and phrase_buffer:
                        tokens_to_speak.extend(list(phrase_buffer))
                    if not tokens_to_speak:
                        print("Nothing to speak yet.")
                        continue

                    def _speak_async(tokens: list[str]) -> None:
                        nonlocal speaking
                        speaking = True
                        try:
                            translation, emotion, spoken = voice.speak_from_sign_tokens(tokens)
                            if spoken:
                                print(f"Spoken: {translation} [{emotion}]")
                            else:
                                print("Speech generation failed.")
                        finally:
                            speaking = False

                    threading.Thread(target=_speak_async, args=(tokens_to_speak,), daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
