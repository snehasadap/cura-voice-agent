import asyncio
import contextlib
import random
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles

from assemblyai_stt import AssemblyAISTT
from cartesia_prompts import CARTESIA_TTS_SYSTEM_PROMPT
from cartesia_tts import CartesiaTTS
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from utils import merge_async_iters

load_dotenv()

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory appointment store — persists for the lifetime of the server
# ---------------------------------------------------------------------------
_ALL_SLOTS = ["9:00 AM", "10:30 AM", "1:00 PM", "2:30 PM", "4:00 PM"]

# confirmation_code -> appointment dict
_appointments: dict[str, dict] = {}


def _booked_slots(specialty: str, date: str) -> set[str]:
    return {
        a["time"]
        for a in _appointments.values()
        if a["specialty"].lower() == specialty.lower()
        and a["date"].lower() == date.lower()
    }


def check_availability(specialty: str, preferred_date: str) -> str:
    """Check available appointment slots for a given specialty and date."""
    available = [s for s in _ALL_SLOTS if s not in _booked_slots(specialty, preferred_date)]
    if not available:
        return f"No available slots for {specialty} on {preferred_date}. Please try another date."
    return f"Available slots for {specialty} on {preferred_date}: {', '.join(available)}."


def book_appointment(patient_name: str, specialty: str, date: str, time: str) -> str:
    """Book an appointment for the patient."""
    if time in _booked_slots(specialty, date):
        return f"{time} on {date} is already taken for {specialty}. Please choose another slot."
    code = f"HC{random.randint(10000, 99999)}"
    _appointments[code] = {
        "patient_name": patient_name,
        "specialty": specialty,
        "date": date,
        "time": time,
        "confirmation_code": code,
    }
    return (
        f"Appointment booked for {patient_name} with {specialty} on {date} at {time}. "
        f"Confirmation code: {code}."
    )


def list_appointments(patient_name: str) -> str:
    """List all appointments for a patient."""
    appts = [a for a in _appointments.values() if a["patient_name"].lower() == patient_name.lower()]
    if not appts:
        return f"No appointments found for {patient_name}."
    lines = [
        f"{a['specialty']} on {a['date']} at {a['time']} — confirmation {a['confirmation_code']}"
        for a in appts
    ]
    return f"Appointments for {patient_name}: " + "; ".join(lines) + "."


def cancel_appointment(confirmation_code: str) -> str:
    """Cancel an appointment by confirmation code."""
    appt = _appointments.pop(confirmation_code.upper(), None)
    if not appt:
        return f"No appointment found with confirmation code {confirmation_code}."
    return (
        f"Appointment {confirmation_code} for {appt['patient_name']} "
        f"({appt['specialty']} on {appt['date']} at {appt['time']}) has been cancelled."
    )


system_prompt = f"""
You are a friendly and professional healthcare receptionist. Your goal is to help patients book, view, and cancel medical appointments.
Be concise, empathetic, and clear.

Available specialties: general practice, cardiology, dermatology, orthopedics, pediatrics, neurology.
Available days: Monday through Friday.
Available hours: 9:00 AM to 5:00 PM in 90-minute slots.

To book an appointment collect: patient full name, specialty or reason for visit, preferred date and time.
Use check_availability to show open slots before booking.
Use book_appointment once the patient confirms a slot.
Use list_appointments when a patient wants to see their upcoming appointments.
Use cancel_appointment when a patient wants to cancel using their confirmation code.

{CARTESIA_TTS_SYSTEM_PROMPT}
"""

agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[check_availability, book_appointment, list_appointments, cancel_appointment],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)

    This function takes a stream of audio chunks and sends them to AssemblyAI for STT.

    It uses a producer-consumer pattern where:
    - Producer: A background task reads audio chunks from audio_stream and sends
      them to AssemblyAI via WebSocket. This runs concurrently with the consumer,
      allowing transcription to begin before all audio has arrived.
    - Consumer: The main coroutine receives transcription events from AssemblyAI
      and yields them downstream. Events include both partial results (stt_chunk)
      and final transcripts (stt_output).

    Args:
        audio_stream: Async iterator of PCM audio bytes (16-bit, mono, 16kHz)

    Yields:
        STT events (stt_chunk for partials, stt_output for final transcripts)
    """
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio():
        """
        Background task that pumps audio chunks to AssemblyAI.

        This runs concurrently with the main coroutine, continuously reading
        audio chunks from the input stream and forwarding them to AssemblyAI.
        When the input stream ends, it signals completion by closing the
        WebSocket connection.
        """
        try:
            # Stream each audio chunk to AssemblyAI as it arrives
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            # Signal to AssemblyAI that audio streaming is complete
            await stt.close()

    # Launch the audio sending task in the background
    # This allows us to simultaneously receive transcripts in the main coroutine
    send_task = asyncio.create_task(send_audio())

    try:
        # Consumer loop: receive and yield transcription events as they arrive
        # from AssemblyAI. The receive_events() method listens on the WebSocket
        # for transcript events and yields them as they become available.
        async for event in stt.receive_events():
            yield event
    finally:
        # Cleanup: ensure the background task is cancelled and awaited
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        # Ensure the WebSocket connection is closed
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    thread_id = str(uuid4())
    queue: asyncio.Queue[VoiceAgentEvent | object] = asyncio.Queue()
    _sentinel = object()

    accumulated: list[str] = []
    latest_transcript: str | None = None
    pending_task: asyncio.Task | None = None

    async def run_agent(transcript: str) -> None:
        stream = agent.astream(
            {"messages": [HumanMessage(content=transcript)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        )
        async for message, _ in stream:
            if isinstance(message, AIMessage) and message.text:
                await queue.put(AgentChunkEvent.create(message.text))
            if isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    await queue.put(ToolCallEvent.create(
                        id=tc.get("id", str(uuid4())),
                        name=tc.get("name", "unknown"),
                        args=tc.get("args", {}),
                    ))
            if isinstance(message, ToolMessage):
                await queue.put(ToolResultEvent.create(
                    tool_call_id=getattr(message, "tool_call_id", ""),
                    name=getattr(message, "name", "unknown"),
                    result=str(message.content) if message.content else "",
                ))
        await queue.put(AgentEndEvent.create())

    async def reset_debounce() -> None:
        nonlocal pending_task
        if pending_task and not pending_task.done():
            pending_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_task

        async def debounced() -> None:
            nonlocal latest_transcript
            await asyncio.sleep(0.25)
            transcript = latest_transcript
            accumulated.clear()
            latest_transcript = None
            if transcript:
                await run_agent(transcript)

        pending_task = asyncio.create_task(debounced())

    async def producer() -> None:
        nonlocal latest_transcript
        async for event in event_stream:
            await queue.put(event)
            if event.type == "stt_output":
                # Accumulate fragments — utterances split across turns get joined
                accumulated.append(event.transcript)
                latest_transcript = " ".join(accumulated)
            if event.type in ("stt_chunk", "stt_output"):
                await reset_debounce()
        if pending_task and not pending_task.done():
            await pending_task
        await queue.put(_sentinel)

    task = asyncio.create_task(producer())
    try:
        while True:
            item = await queue.get()
            if item is _sentinel:
                break
            yield item  # type: ignore[misc]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Audio)

    This function takes a stream of upstream voice agent events and processes them.
    When agent_chunk events arrive, it sends the text to Cartesia for TTS synthesis.
    Audio is streamed back as tts_chunk events as it's generated.
    All upstream events are passed through unchanged.

    It uses merge_async_iters to combine two concurrent streams:
    - process_upstream(): Iterates through incoming events, yields them for
      passthrough, and sends agent text chunks to Cartesia for synthesis.
    - tts.receive_events(): Yields audio chunks from Cartesia as they are
      synthesized.

    The merge utility runs both iterators concurrently, yielding items from
    either stream as they become available. This allows audio generation to
    begin before the agent has finished generating all text, minimizing latency.

    Args:
        event_stream: An async iterator of upstream voice agent events

    Yields:
        All upstream events plus tts_chunk events for synthesized audio
    """
    tts = CartesiaTTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        """
        Process upstream events, yielding them while sending text to Cartesia.

        This async generator serves two purposes:
        1. Pass through all upstream events (stt_chunk, stt_output, agent_chunk)
           so downstream consumers can observe the full event stream.
        2. Buffer agent_chunk text and send to Cartesia when agent_end arrives.
           This ensures the full response is sent at once for better TTS quality.
        """
        buffer: list[str] = []
        async for event in event_stream:
            # Pass through all events to downstream consumers
            yield event
            # Buffer agent text chunks
            if event.type == "agent_chunk":
                buffer.append(event.text)
            # Send all buffered text to Cartesia when agent finishes
            if event.type == "agent_end":
                await tts.send_text("".join(buffer))
                buffer = []

    try:
        # Merge the processed upstream events with TTS audio events
        # Both streams run concurrently, yielding events as they arrive
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        # Cleanup: close the WebSocket connection to Cartesia
        await tts.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        try:
            while True:
                data = await websocket.receive_bytes()
                yield data
        except WebSocketDisconnect:
            return

    output_stream = _tts_stream(_agent_stream(_stt_stream(websocket_audio_stream())))

    try:
        async for event in output_stream:
            await websocket.send_json(event_to_dict(event))
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
