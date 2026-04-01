import os
import asyncio
from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()

async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit WebRTC Room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Configure Guru AI Personality
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are Guru AI, a calm, empathetic spiritual companion. "
            "Provide extremely brief (1-3 sentences), warm, and deeply comforting guidance. "
            "Speak naturally as if in a live voice conversation—no markdown, no lists."
        ),
    )

    # Groq is 100% compatible with the OpenAI API protocol.
    # We substitute the base API URL to point to Groq's high-speed LPU clusters.
    groq_llm = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-8b-8192", 
    )

    # The Holy Grail: Synchronous Voice Pipeline Orchestrator
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            api_key=os.environ.get("DEEPGRAM_API_KEY"),
            model="nova-2-general",
        ),
        llm=groq_llm,
        tts=cartesia.TTS(
            api_key=os.environ.get("CARTESIA_API_KEY"),
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02", # The beloved 'Katie' Voice ID
        ),
        chat_ctx=initial_ctx,
    )

    # Start the continuous WebRTC streaming loop
    agent.start(ctx.room)

    # Greet the user automatically upon successful connection
    await asyncio.sleep(1)
    await agent.say("I am here with you. How is your spirit today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
