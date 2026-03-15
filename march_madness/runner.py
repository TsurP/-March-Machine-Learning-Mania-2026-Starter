"""High-level runner for the March Madness agent pipeline."""

from __future__ import annotations

import asyncio
from typing import Optional

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part

from .agents import build_pipeline_agent
from .config import get_google_api_key


async def run_pipeline_once(
    app_name: str = "march_madness_2026",
    user_id: str = "kaggle_user",
    session_id: str = "pipeline_run_1",
) -> None:
    """Execute the full agent pipeline once and stream agent outputs."""
    # Ensure API key is configured
    get_google_api_key()

    pipeline = build_pipeline_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=pipeline, app_name=app_name, session_service=session_service)

    # Create session
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Kick off the pipeline with a user message
    user_message = Content(
        role="user",
        parts=[
            Part(
                text=(
                    "Run the full March Madness prediction pipeline. "
                    "Load data, compute features, train the model, "
                    "and generate the submission file."
                )
            )
        ],
    )

    print("🚀 Starting pipeline...\n")
    print("=" * 60)

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            author = event.author or "Pipeline"
            text = event.content.parts[0].text or ""
            if text.strip():
                print(f"\n🤖 [{author}]")
                print("-" * 40)
                print(text.strip())
                print("=" * 60)

    print("\n✅ Pipeline complete!")


def run_sync(
    app_name: str = "march_madness_2026",
    user_id: str = "kaggle_user",
    session_id: str = "pipeline_run_1",
) -> None:
    """Synchronous wrapper to run the pipeline once."""
    asyncio.run(run_pipeline_once(app_name=app_name, user_id=user_id, session_id=session_id))

