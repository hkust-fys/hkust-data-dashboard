"""One-off verification: download the live map PNG and pixel-check markers."""

import asyncio
import io
import os
import sys

import aiohttp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPERATOR_COLORS = {
    "KMB red": (225, 29, 72),
    "GMB green": (22, 163, 74),
    "CTB yellow": (250, 204, 21),
    "public stop purple": (139, 92, 246),
    "gate/shuttle blue": (37, 99, 235),
}


async def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("DISCORD_TOKEN", "").strip()
    channel_id = os.getenv("ANNOUNCE_CHANNEL_ID", "").strip()
    headers = {"Authorization": f"Bot {token}"}
    async with aiohttp.ClientSession() as session:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=20"
        async with session.get(url, headers=headers) as resp:
            messages = await resp.json()
        dashboards = [
            m
            for m in messages
            if m.get("author", {}).get("bot")
            and m.get("content") == "HKUST Campus Dashboard"
        ]
        if not dashboards:
            print("NO_DASHBOARD_MESSAGE")
            return
        map_embed = next(
            (
                e
                for e in dashboards[0]["embeds"]
                if "Traffic map" in (e.get("title") or "")
            ),
            None,
        )
        if map_embed is None:
            print("NO_MAP_EMBED")
            return
        image_url = (map_embed.get("image") or {}).get("url", "")
        print("map image url host:", image_url.split("/")[2])
        async with session.get(image_url) as img_resp:
            print("GET map png ->", img_resp.status)
            data = await img_resp.read()
        print("png bytes:", len(data), "| magic:", data[:4].hex())

        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        print("dimensions:", image.size)
        counts = {name: 0 for name in OPERATOR_COLORS}
        tolerance = 10
        for _count, pixel in image.getcolors(maxcolors=2_000_000):
            for name, color in OPERATOR_COLORS.items():
                if all(abs(pixel[i] - color[i]) <= tolerance for i in range(3)):
                    counts[name] += _count
                    break
        print("marker-color pixel counts:", counts)
        ok = all(counts[name] > 0 for name in OPERATOR_COLORS)
        print("ALL_MARKERS_PRESENT" if ok else "SOME_MARKER_COLORS_MISSING")


if __name__ == "__main__":
    asyncio.run(main())
