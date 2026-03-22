import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from dual_runtime import (
    LEGACY_INPUT_SIZE,
    MODEL_DIR,
    EXPECTED_INPUT_SIZE,
    compute_velocity,
    create_holistic_model,
    extract_legacy_hand_position_features,
    extract_position_features,
    feature_size,
    first_existing,
    load_classifier,
    predict_label,
    resolve_source,
    validate_classifier_input,
)
from voice_engine import VoiceEngine


class RuntimeStartRequest(BaseModel):
    mode: str = Field(default="hybrid", pattern="^(letters|words|hybrid)$")
    source: str = "0"
    voice: bool = False
    word_threshold: float = 0.70
    letter_threshold: float = 0.70
    append_threshold: float = 0.85
    append_cooldown: float = 1.2
    word_stable_frames: int = 8
    word_commit_seconds: float = 0.65
    word_reset_seconds: float = 0.35
    word_gap_seconds: float = 0.7
    buffer_max: int = 12
    spell_threshold: float = 0.75
    spell_stable_frames: int = 5
    spell_gap_seconds: float = 0.40
    spell_commit_seconds: float = 0.70
    spell_reset_seconds: float = 0.25


class RuntimeCommandRequest(BaseModel):
    action: str = Field(pattern="^(clear_phrase|clear_text|clear_transcript|speak)$")


class RuntimeState(BaseModel):
    running: bool
    mode: str
    seq: int
    source: str
    chosen_label: str | None
    chosen_conf: float
    chosen_mode: str
    word_prediction: dict | None
    letter_prediction: dict | None
    phrase: str
    text: str
    transcript: str
    pending_word: str
    pending_char: str
    speaking: bool
    voice_enabled: bool
    error: str | None
    updated_at: float


@dataclass
class WorkerState:
    running: bool = False
    mode: str = "hybrid"
    seq: int = 0
    source: str = "0"
    chosen_label: str | None = None
    chosen_conf: float = 0.0
    chosen_mode: str = "none"
    word_prediction: dict | None = None
    letter_prediction: dict | None = None
    phrase: str = ""
    text: str = ""
    transcript: str = ""
    pending_word: str = ""
    pending_char: str = ""
    speaking: bool = False
    voice_enabled: bool = False
    error: str | None = None
    updated_at: float = field(default_factory=time.time)


class RuntimeWorker:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state = WorkerState()
        self._config = RuntimeStartRequest()
        self._command_queue: deque[str] = deque()
        self._frame_jpeg: bytes | None = None
        self._voice_engine = VoiceEngine()

    def start(self, config: RuntimeStartRequest) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Runtime is already running.")
            self._config = config
            self._stop_event.clear()
            self._command_queue.clear()
            self._frame_jpeg = None
            self._set_state_locked(
                running=True,
                mode=config.mode,
                source=config.source,
                error=None,
                chosen_label=None,
                chosen_conf=0.0,
                chosen_mode="none",
                word_prediction=None,
                letter_prediction=None,
                phrase="",
                text="",
                transcript="",
                pending_word="",
                pending_char="",
                speaking=False,
                voice_enabled=bool(config.voice and self._voice_engine.enabled),
            )
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        return {"status": "started"}

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            self._state.running = False
            self._state.speaking = False
            self._state.seq += 1
            self._state.updated_at = time.time()
            self._command_queue.clear()
        return {"status": "stopped"}

    def snapshot(self) -> RuntimeState:
        with self._lock:
            return RuntimeState(**self._state.__dict__)

    def queue_command(self, action: str) -> None:
        with self._lock:
            if not self._state.running:
                raise RuntimeError("Runtime is not running.")
            self._command_queue.append(action)

    def latest_frame(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def _set_state(self, **kwargs) -> None:
        with self._lock:
            self._set_state_locked(**kwargs)

    def _set_state_locked(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._state, key, value)
        self._state.seq += 1
        self._state.updated_at = time.time()

    def _drain_commands(self) -> list[str]:
        with self._lock:
            commands = list(self._command_queue)
            self._command_queue.clear()
        return commands

    def _set_latest_frame(self, frame) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            return
        with self._lock:
            self._frame_jpeg = encoded.tobytes()

    def _load_models(self, config: RuntimeStartRequest):
        wants_letters = config.mode in ("letters", "hybrid")
        wants_words = config.mode in ("words", "hybrid")

        word_model = None
        if wants_words:
            word_model_path = first_existing(
                [
                    f"{MODEL_DIR}/asl_words_classifier.pt",
                    f"{MODEL_DIR}/asl_classifier.pt",
                ]
            )
            word_label_map_path = first_existing(
                [
                    f"{MODEL_DIR}/asl_words_label_map.json",
                    f"{MODEL_DIR}/asl_label_map.json",
                ]
            )
            if word_model_path is None or word_label_map_path is None:
                raise RuntimeError("No word model files found in models folder.")
            word_model = load_classifier(word_model_path, word_label_map_path)
            word_model = validate_classifier_input("Word", word_model)
            if word_model is None:
                raise RuntimeError(
                    f"Word model is incompatible. Expected input size {EXPECTED_INPUT_SIZE} or {LEGACY_INPUT_SIZE}."
                )

        alpha_model = None
        if wants_letters:
            alpha_model_path = first_existing(
                [
                    f"{MODEL_DIR}/asl_alphabet_classifier.pt",
                    f"{MODEL_DIR}/asl_classifier.pt",
                ]
            )
            alpha_label_map_path = first_existing(
                [
                    f"{MODEL_DIR}/asl_alphabet_label_map.json",
                    f"{MODEL_DIR}/asl_label_map.json",
                ]
            )
            if alpha_model_path is None or alpha_label_map_path is None:
                raise RuntimeError("No alphabet model files found in models folder.")
            alpha_model = load_classifier(alpha_model_path, alpha_label_map_path)
            alpha_model = validate_classifier_input("Alphabet", alpha_model)
            if alpha_model is None:
                raise RuntimeError(
                    f"Alphabet model is incompatible. Expected input size {EXPECTED_INPUT_SIZE} or {LEGACY_INPUT_SIZE}."
                )

        return wants_letters, wants_words, alpha_model, word_model

    def _speak_async(self, tokens: list[str]) -> None:
        self._set_state(speaking=True, error=None)
        try:
            _translation, _emotion, spoken = self._voice_engine.speak_from_sign_tokens(tokens)
            if not spoken:
                self._set_state(error="Speech generation failed.")
        except Exception as exc:
            self._set_state(error=f"Speak failed: {exc}")
        finally:
            self._set_state(speaking=False)

    @staticmethod
    def _build_transcript_text(entries: list[tuple[str, str]]) -> str:
        parts: list[str] = []
        letter_run: list[str] = []

        def flush_letters() -> None:
            nonlocal letter_run
            if letter_run:
                parts.append("".join(letter_run))
                letter_run = []

        for kind, value in entries:
            if kind == "letter":
                letter_run.append(value)
            else:
                flush_letters()
                parts.append(value)
        flush_letters()
        return " ".join([p for p in parts if p]).strip()

    @staticmethod
    def _build_speak_tokens(entries: list[tuple[str, str]]) -> list[str]:
        tokens: list[str] = []
        letter_run: list[str] = []

        def flush_letters() -> None:
            nonlocal letter_run
            if letter_run:
                tokens.append(" ".join([ch.upper() for ch in letter_run]))
                letter_run = []

        for kind, value in entries:
            if kind == "letter":
                letter_run.append(value)
            else:
                flush_letters()
                tokens.append(value)
        flush_letters()
        return [token for token in tokens if token]

    def _run_loop(self) -> None:
        config = self._config
        voice_enabled = bool(config.voice and self._voice_engine.enabled)
        prev_pos = [0.0] * feature_size()
        prev_legacy_pos = [0.0] * 63

        transcript_entries: list[tuple[str, str]] = []
        text_preview: deque[str] = deque(maxlen=18)
        phrase_preview: deque[str] = deque(maxlen=max(3, min(8, config.buffer_max)))
        runtime_error: str | None = None

        pending_char = ""
        pending_char_frames = 0
        pending_char_started_at = 0.0
        pending_char_last_seen_at = 0.0
        repeat_unlocked = True
        last_confident_spell_time = 0.0

        pending_word = ""
        pending_word_frames = 0
        pending_word_started_at = 0.0
        pending_word_last_seen_at = 0.0
        last_confident_word_time = 0.0
        word_repeat_unlocked = True
        last_word_token = ""
        last_word_token_time = 0.0

        cap = None
        try:
            wants_letters, wants_words, alpha_model, word_model = self._load_models(config)
            cap = cv2.VideoCapture(resolve_source(str(config.source)))
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open camera source: {config.source}")

            with create_holistic_model() as holistic:
                while cap.isOpened() and not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame = cv2.flip(frame, 1)
                    self._set_latest_frame(frame)

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

                    word_pred = None
                    if wants_words and word_model is not None and hand_present:
                        word_features = legacy_features if word_model.input_size == LEGACY_INPUT_SIZE else features
                        wp = predict_label(word_model, word_features)
                        if wp is not None:
                            w_label, w_conf = wp
                            word_pred = {"label": str(w_label), "confidence": float(w_conf)}

                    alpha_pred = None
                    if wants_letters and alpha_model is not None and hand_present:
                        alpha_features = legacy_features if alpha_model.input_size == LEGACY_INPUT_SIZE else features
                        ap = predict_label(alpha_model, alpha_features)
                        if ap is not None:
                            a_label, a_conf = ap
                            alpha_pred = {"label": str(a_label), "confidence": float(a_conf)}

                    chosen_label = None
                    chosen_conf = 0.0
                    chosen_mode = "none"
                    if config.mode == "words":
                        if word_pred and word_pred["confidence"] >= config.word_threshold:
                            chosen_label, chosen_conf, chosen_mode = word_pred["label"], word_pred["confidence"], "WORD"
                        elif word_pred:
                            chosen_label, chosen_conf, chosen_mode = word_pred["label"], word_pred["confidence"], "LOW-WORD"
                    elif config.mode == "letters":
                        if alpha_pred and alpha_pred["confidence"] >= config.letter_threshold:
                            chosen_label, chosen_conf, chosen_mode = alpha_pred["label"], alpha_pred["confidence"], "LETTER"
                        elif alpha_pred:
                            chosen_label, chosen_conf, chosen_mode = alpha_pred["label"], alpha_pred["confidence"], "LOW-LETTER"
                    else:
                        if word_pred and word_pred["confidence"] >= config.word_threshold:
                            chosen_label, chosen_conf, chosen_mode = word_pred["label"], word_pred["confidence"], "WORD"
                        elif alpha_pred and alpha_pred["confidence"] >= config.letter_threshold:
                            chosen_label, chosen_conf, chosen_mode = alpha_pred["label"], alpha_pred["confidence"], "LETTER"
                        elif alpha_pred:
                            chosen_label, chosen_conf, chosen_mode = alpha_pred["label"], alpha_pred["confidence"], "LOW-LETTER"
                        elif word_pred:
                            chosen_label, chosen_conf, chosen_mode = word_pred["label"], word_pred["confidence"], "LOW-WORD"

                    now = time.time()
                    if not wants_letters or not hand_present:
                        pending_char = ""
                        pending_char_frames = 0
                        pending_char_started_at = 0.0
                        pending_char_last_seen_at = 0.0
                    elif alpha_pred is not None and alpha_pred["confidence"] >= config.spell_threshold:
                        current_char = alpha_pred["label"].lower()
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

                            stable_for_frames = pending_char_frames >= max(1, config.spell_stable_frames)
                            stable_for_time = (now - pending_char_started_at) >= max(0.0, config.spell_commit_seconds)
                            if stable_for_frames and stable_for_time:
                                if repeat_unlocked or current_char != (text_preview[-1] if text_preview else ""):
                                    text_preview.append(current_char)
                                    transcript_entries.append(("letter", current_char))
                                    repeat_unlocked = False
                                    pending_char = ""
                                    pending_char_frames = 0
                    else:
                        if pending_char and (now - pending_char_last_seen_at) >= max(0.0, config.spell_reset_seconds):
                            pending_char = ""
                            pending_char_frames = 0

                    if (now - last_confident_spell_time) >= config.spell_gap_seconds:
                        repeat_unlocked = True

                    if wants_words and word_pred is not None and word_pred["confidence"] >= config.append_threshold:
                        current_word = word_pred["label"].strip().lower()
                        last_confident_word_time = now

                        if current_word == pending_word:
                            pending_word_frames += 1
                            pending_word_last_seen_at = now
                        else:
                            pending_word = current_word
                            pending_word_frames = 1
                            pending_word_started_at = now
                            pending_word_last_seen_at = now

                        stable_for_frames = pending_word_frames >= max(1, config.word_stable_frames)
                        stable_for_time = (now - pending_word_started_at) >= max(0.0, config.word_commit_seconds)
                        cooldown_ok = (now - last_word_token_time) >= max(0.0, config.append_cooldown)

                        if stable_for_frames and stable_for_time:
                            if word_repeat_unlocked or current_word != last_word_token:
                                if current_word != last_word_token or cooldown_ok:
                                    phrase_preview.append(current_word)
                                    transcript_entries.append(("word", current_word))
                                    last_word_token = current_word
                                    last_word_token_time = now
                                    word_repeat_unlocked = False
                                    pending_word = ""
                                    pending_word_frames = 0
                    else:
                        if pending_word and (now - pending_word_last_seen_at) >= max(0.0, config.word_reset_seconds):
                            pending_word = ""
                            pending_word_frames = 0

                    if wants_words and (now - last_confident_word_time) >= max(0.0, config.word_gap_seconds):
                        word_repeat_unlocked = True

                    for command in self._drain_commands():
                        if command == "clear_phrase":
                            phrase_preview.clear()
                        elif command == "clear_text":
                            text_preview.clear()
                        elif command == "clear_transcript":
                            transcript_entries.clear()
                        elif command == "speak":
                            if not voice_enabled:
                                runtime_error = "Voice is disabled for this session. Restart with voice enabled."
                            elif self.snapshot().speaking:
                                runtime_error = "Already speaking."
                            else:
                                tokens_to_speak = self._build_speak_tokens(transcript_entries)
                                if not tokens_to_speak:
                                    runtime_error = "Nothing to speak yet."
                                else:
                                    runtime_error = None
                                    threading.Thread(target=self._speak_async, args=(tokens_to_speak,), daemon=True).start()

                    phrase_text = " ".join(list(phrase_preview))
                    spelled_text = "".join(list(text_preview))
                    transcript_text = self._build_transcript_text(transcript_entries)

                    self._set_state(
                        running=True,
                        mode=config.mode,
                        source=str(config.source),
                        chosen_label=chosen_label,
                        chosen_conf=float(chosen_conf),
                        chosen_mode=chosen_mode,
                        word_prediction=word_pred,
                        letter_prediction=alpha_pred,
                        phrase=phrase_text,
                        text=spelled_text,
                        transcript=transcript_text,
                        pending_word=pending_word,
                        pending_char=pending_char,
                        voice_enabled=voice_enabled,
                        error=runtime_error,
                    )

            self._set_state(running=False)
        except Exception as exc:
            self._set_state(running=False, error=str(exc))
        finally:
            if cap is not None:
                cap.release()


worker = RuntimeWorker()
app = FastAPI(title="Sign-Speak Runtime API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "runtime": worker.snapshot().model_dump()})


@app.post("/start")
def start_runtime(payload: RuntimeStartRequest) -> JSONResponse:
    try:
        result = worker.start(payload)
        return JSONResponse(result)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/stop")
def stop_runtime() -> JSONResponse:
    return JSONResponse(worker.stop())


@app.post("/command")
def command_runtime(payload: RuntimeCommandRequest) -> JSONResponse:
    try:
        worker.queue_command(payload.action)
        return JSONResponse({"queued": payload.action})
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/state")
def get_state() -> RuntimeState:
    return worker.snapshot()


@app.get("/frame")
def get_frame() -> Response:
    frame = worker.latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    return Response(content=frame, media_type="image/jpeg")


@app.websocket("/stream")
async def stream_state(socket: WebSocket) -> None:
    await socket.accept()
    last_seq = -1
    try:
        while True:
            state = worker.snapshot().model_dump()
            if state["seq"] != last_seq:
                await socket.send_json(state)
                last_seq = state["seq"]
            await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        return
