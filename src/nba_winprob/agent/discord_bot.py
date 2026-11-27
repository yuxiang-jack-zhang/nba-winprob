"""The catch-up agent on Discord: push updates to one channel, and chat in it.

Runs beside ``live run`` as its own process and only reads Postgres (read-only
connections), so a Discord or Claude outage cannot touch the trader. Every message in the
channel is a question; the agent answers with the channel's conversation so far, and
"reset" starts a fresh one. Every ``POLL_S`` seconds it posts what the notifier found, and
a digest from the agent when one is due.

The logic lives in ``notify.py`` and ``catchup.py``, which are tested; this module is the
thin asyncio glue between them and Discord.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anthropic
import discord
import psycopg
from discord.ext import tasks

from nba_winprob.agent import tools
from nba_winprob.agent.catchup import Answer, ask
from nba_winprob.agent.notify import Notifier, chunks
from nba_winprob.config import Settings
from nba_winprob.db import store

log = logging.getLogger(__name__)

POLL_S = 15
MAX_HISTORY = 40  # messages; a longer conversation starts over rather than grow forever
DIGEST_QUESTION = "Digest for Jack: P&L, open positions, and how the live games stand."


class CatchupBot(discord.Client):
    def __init__(self, settings: Settings, channel_id: int) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # must also be switched on in the developer portal
        super().__init__(intents=intents)
        self.channel_id = channel_id
        self.api = anthropic.Anthropic(api_key=settings.anthropic_api_key or None).beta.messages
        # One connection per thread that uses it: chat runs in a worker, pushes in another.
        self.chat_conn, self.watch_conn = store.connect(settings), store.connect(settings)
        for conn in (self.chat_conn, self.watch_conn):
            tools.make_read_only(conn)
        self.notifier = Notifier(self.watch_conn, now=time.time())
        self.history: list[dict[str, Any]] = []
        self.lock = asyncio.Lock()  # one question at a time: they share the history

    async def setup_hook(self) -> None:
        self.push.start()

    async def on_ready(self) -> None:
        log.info("logged in as %s; channel %d", self.user, self.channel_id)

    async def close(self) -> None:
        self.push.cancel()
        self.chat_conn.close()
        self.watch_conn.close()
        await super().close()

    def _channel(self) -> discord.abc.Messageable:
        channel = self.get_channel(self.channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            raise RuntimeError(f"channel {self.channel_id} not found, or not a text channel")
        return channel

    @tasks.loop(seconds=POLL_S)
    async def push(self) -> None:
        try:
            messages, digest_due = await asyncio.to_thread(self.notifier.poll, time.time())
            channel = self._channel()
            for message in messages:
                for part in chunks(message):
                    await channel.send(part)
            if digest_due:
                answer = await self._ask(DIGEST_QUESTION, fresh=True)
                for part in chunks(f"Digest: {answer.text}"):
                    await channel.send(part)
        except Exception:  # an unhandled error would stop the loop for good
            log.exception("push failed; retrying next poll")

    @push.before_loop
    async def _wait(self) -> None:
        await self.wait_until_ready()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.channel.id != self.channel_id:
            return
        text = message.content.strip()
        if text.lower() == "reset":
            self.history.clear()
            await message.channel.send("Starting a fresh conversation.")
            return
        async with message.channel.typing():
            answer = await self._ask(text)
        for part in chunks(answer.text) or ["(no answer)"]:
            await message.channel.send(part)

    async def _ask(self, question: str, fresh: bool = False) -> Answer:
        async with self.lock:
            run_id = await asyncio.to_thread(tools.latest_run_id, self.chat_conn)
            if run_id is None:
                return Answer("No trading runs yet: start one with `nba-winprob live run`.", "none")
            if len(self.history) > MAX_HISTORY:
                self.history.clear()
            history = [] if fresh else self.history  # digests don't crowd the chat
            try:
                return await asyncio.to_thread(
                    ask, self.api, self.chat_conn, run_id, question, history=history
                )
            except (anthropic.APIError, psycopg.Error) as err:
                # A failed exchange can leave a tool call unanswered, which the API would
                # reject on every later request: start over instead.
                history.clear()
                log.exception("question failed")
                return Answer(f"Couldn't answer that ({type(err).__name__}); try again.", "error")


def run(settings: Settings) -> None:
    if not settings.discord_bot_token or settings.discord_channel_id is None:
        raise ValueError("set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in .env")
    bot = CatchupBot(settings, settings.discord_channel_id)
    bot.run(settings.discord_bot_token, log_handler=None)  # logging is configured by the CLI
