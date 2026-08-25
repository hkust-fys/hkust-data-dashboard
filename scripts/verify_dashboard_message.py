"""One-off verification: inspect the live dashboard message via Discord REST."""

import asyncio
import json
import os
import sys

import aiohttp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("DISCORD_TOKEN", "").strip()
    channel_id = os.getenv("ANNOUNCE_CHANNEL_ID", "").strip()
    headers = {"Authorization": f"Bot {token}"}
    async with aiohttp.ClientSession() as session:
        # Channel: pin down the dashboard message by scanning recent messages.
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=50"
        async with session.get(url, headers=headers) as resp:
            print("GET messages ->", resp.status)
            messages = await resp.json()
        dashboards = [
            m
            for m in messages
            if m.get("author", {}).get("bot")
            and m.get("content") == "HKUST Campus Dashboard"
        ]
        if not dashboards:
            print("NO_DASHBOARD_MESSAGE_FOUND in last", len(messages), "messages")
            return
        msg = dashboards[0]
        print("message id:", msg["id"], "| edited_at:", msg.get("edited_timestamp"))
        print("components:", json.dumps(msg.get("components"), indent=1)[:800])
        print("embed count:", len(msg.get("embeds", [])))
        for i, embed in enumerate(msg["embeds"]):
            title = embed.get("title") or "(no title)"
            image = (embed.get("image") or {}).get("url", "")
            print(f"  embed[{i}] {title} | image={image}")
        print("attachments:", [(a["filename"], a["size"]) for a in msg.get("attachments", [])])
        camera_titles = [e.get("title", "") for e in msg["embeds"] if e.get("title", "").startswith("📷")]
        print("camera_embeds_remaining:", len(camera_titles))


if __name__ == "__main__":
    asyncio.run(main())
