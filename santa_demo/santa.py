import asyncio
import queue
import sys
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner, RealtimeSession, RealtimeSessionEvent
from dotenv import load_dotenv
import os
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Audio configuration
CHUNK_LENGTH_S = 0.05  # 50ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1

@function_tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = RealtimeAgent(
    name="Santa Claus",
    instructions=(
        "You are Santa Claus. Speak in a warm, jolly, and friendly tone. "
        "Always speak in English. Greet children and answer questions in a cheerful and festive way. "
        "Keep your replies short and magical."
    ),
    tools=[get_weather],
)

def _truncate_str(s: str, max_length: int) -> str:
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s

class NoUIDemo:
    def __init__(self) -> None:
        self.session: RealtimeSession | None = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_player: sd.OutputStream | None = None
        self.recording = False

        self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=10)
        self.interrupt_event = threading.Event()
        self.current_audio_chunk: np.ndarray | None = None
        self.chunk_position = 0

        self.is_playing_audio = threading.Event()

    def _output_callback(self, outdata, frames: int, time, status) -> None:
        if status:
            print(f"Output callback status: {status}")

        if self.interrupt_event.is_set():
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            self.current_audio_chunk = None
            self.chunk_position = 0
            self.interrupt_event.clear()
            self.is_playing_audio.clear()
            outdata.fill(0)
            return

        outdata.fill(0)
        samples_filled = 0
        audio_playing = False

        while samples_filled < len(outdata):
            if self.current_audio_chunk is None:
                try:
                    self.current_audio_chunk = self.output_queue.get_nowait()
                    self.chunk_position = 0
                except queue.Empty:
                    break

            remaining_output = len(outdata) - samples_filled
            remaining_chunk = len(self.current_audio_chunk) - self.chunk_position
            samples_to_copy = min(remaining_output, remaining_chunk)

            if samples_to_copy > 0:
                chunk_data = self.current_audio_chunk[
                    self.chunk_position : self.chunk_position + samples_to_copy
                ]
                outdata[samples_filled : samples_filled + samples_to_copy, 0] = chunk_data
                samples_filled += samples_to_copy
                self.chunk_position += samples_to_copy
                audio_playing = True

                if self.chunk_position >= len(self.current_audio_chunk):
                    self.current_audio_chunk = None
                    self.chunk_position = 0

        if audio_playing:
            self.is_playing_audio.set()
        else:
            self.is_playing_audio.clear()

    async def run(self) -> None:
        print("Connecting, may take a few seconds...")

        chunk_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)
        self.audio_player = sd.OutputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
            callback=self._output_callback,
            blocksize=chunk_size,
        )
        self.audio_player.start()

        try:
            runner = RealtimeRunner(agent)
            async with await runner.run() as session:
                self.session = session
                print("Connected. Starting audio recording...")

                await self.start_audio_recording()
                print("Audio recording started. You can start speaking - expect lots of logs!")

                async for event in session:
                    await self._on_event(event)

        finally:
            if self.audio_player and self.audio_player.active:
                self.audio_player.stop()
            if self.audio_player:
                self.audio_player.close()

        print("Session ended")

    async def start_audio_recording(self) -> None:
        self.audio_stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
        )

        self.audio_stream.start()
        self.recording = True
        asyncio.create_task(self.capture_audio())

    async def capture_audio(self) -> None:
        if not self.audio_stream or not self.session:
            return

        read_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)

        try:
            while self.recording:
                if self.audio_stream.read_available < read_size:
                    await asyncio.sleep(0.01)
                    continue

                data, _ = self.audio_stream.read(read_size)

                if self.is_playing_audio.is_set():
                    continue  # 🧠 忽略自己的聲音播放過程中的輸入

                audio_bytes = data.tobytes()
                await self.session.send_audio(audio_bytes)
                await asyncio.sleep(0)

        except Exception as e:
            print(f"Audio capture error: {e}")
        finally:
            if self.audio_stream and self.audio_stream.active:
                self.audio_stream.stop()
            if self.audio_stream:
                self.audio_stream.close()

    async def _on_event(self, event: RealtimeSessionEvent) -> None:
        try:
            if event.type == "agent_start":
                print(f"Agent started: {event.agent.name}")
            elif event.type == "agent_end":
                print(f"Agent ended: {event.agent.name}")
            elif event.type == "handoff":
                print(f"Handoff from {event.from_agent.name} to {event.to_agent.name}")
            elif event.type == "tool_start":
                print(f"Tool started: {event.tool.name}")
            elif event.type == "tool_end":
                print(f"Tool ended: {event.tool.name}; output: {event.output}")
            elif event.type == "audio_end":
                print("Audio ended")
            elif event.type == "audio":
                np_audio = np.frombuffer(event.audio.data, dtype=np.int16)
                
                try:
                    self.output_queue.put_nowait(np_audio)
                except queue.Full:
                    if self.output_queue.qsize() > 8:
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put_nowait(np_audio)
                        except queue.Empty:
                            pass
            elif event.type == "audio_interrupted":
                print("Audio interrupted")
                self.interrupt_event.set()
            elif event.type == "error":
                print(f"Error: {event.error}")
            elif event.type == "history_updated":
                pass
            elif event.type == "history_added":
                pass
            elif event.type == "raw_model_event":
                print(f"Raw model event: {_truncate_str(str(event.data), 50)}")
            else:
                print(f"Unknown event type: {event.type}")
        except Exception as e:
            print(f"Error processing event: {_truncate_str(str(e), 50)}")

if __name__ == "__main__":
    demo = NoUIDemo()
    try:
        asyncio.run(demo.run())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
