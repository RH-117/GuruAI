import os
import asyncio
from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()

async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit WebRTC Room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    print("Guru AI worker natively connected. Initializing LiveKit 1.0 AgentSession...")

    # Groq is 100% compatible with the OpenAI API protocol.
    groq_llm = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-8b-8192", 
    )

    # The Holy Grail: LiveKit 1.0 Synchronous Voice Orchestrator
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            api_key=os.environ.get("DEEPGRAM_API_KEY"),
            model="nova-2-general",
        ),
        llm=groq_llm,
        tts=cartesia.TTS(
            api_key=os.environ.get("CARTESIA_API_KEY"),
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02", # Katie Voice
        )
    )

    # Start the continuous WebRTC streaming loop targeting this local room
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "You are Guru AI, a calm, empathetic spiritual companion. "
                "Provide extremely brief (1-3 sentences), warm, and deeply comforting guidance. "
                "Speak naturally as if in a live voice conversation—no markdown, no lists."
            )
        )
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
