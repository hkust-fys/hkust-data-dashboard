# bot.py
import discord
from discord import app_commands
from discord.ext import commands, tasks

import os
import warnings
import asyncio
import datetime
from dotenv import load_dotenv

# change working directory to wherever bot.py is in
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# load bot token
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Load announcement channel ID
announce_channel_id = int(os.getenv('ANNOUNCE_CHANNEL_ID'))

# define bot
intents = discord.Intents.default()
bot = commands.Bot(
    command_prefix = "d.",
    intents=intents,
    activity=discord.Activity(type=discord.ActivityType.watching, name="campus data"),
    help_command=None
)

# Fetch campus data every 5 minutes
@tasks.loop(minutes=5)
async def fetch_campus_data() -> None:
    """
    Fetch campus data, and display them in the announcement channel.
    """

    # Check announcement channel status
    try:
        announce_channel = await bot.fetch_channel(announce_channel_id)
    except:
        warnings.warn("Failed to connect to announcement channel, retrying in next loop...")
        return

    # TODO: Fetch campus data from API

    # Compose embed using collected data
    embed_data = discord.Embed(
        title="☀️ Campus data dashboard",
        color=0xe0af68,
        timestamp=datetime.datetime.now()
    )

    # Bus queue
    bus_queue_field = f"```\n"
    bus_queue_field += f"{'North':<10}| 42\n"
    bus_queue_field += f"{'South':<10}| 91\n"
    bus_queue_field += "```"

    embed_data.add_field(
        name="🚏 Bus queue",
        value=bus_queue_field,
        inline=False
    )

    # People count
    ppl_count_field = f"```\n"
    ppl_count_field += f"{'LG1 Rest.':<10}| 443\n"
    ppl_count_field += f"{'LG5 Rest.':<10}| 118\n"
    ppl_count_field += f"{'LG7 Rest.':<10}| 721\n"
    ppl_count_field += f"{'LC':<10}| 512\n"
    ppl_count_field += f"{'Atrium':<10}| 203\n"
    ppl_count_field += "```"

    embed_data.add_field(
        name="👩 People count",
        value=ppl_count_field,
        inline=False
    )

    # Food waste
    food_waste_field = f"```\n"
    food_waste_field += f"{'Total':<10}| 73t\n\n"
    food_waste_field += f"🐷 Happypig375 reminder: If you don't finish your food, I will finish you!\n"
    food_waste_field += "```"

    embed_data.add_field(
        name="🗑️ Food waste",
        value=food_waste_field,
        inline=False
    )

    # Sensor data
    sensor_data_field = f"```\n"
    sensor_data_field += f"{'CO2':<10}| 0.04%\n"
    sensor_data_field += f"{'Temp.':<10}| 26°C\n"
    sensor_data_field += f"{'Humidity':<10}| 71%\n"
    sensor_data_field += "```"

    embed_data.add_field(
        name="🎐 Sensor data",
        value=sensor_data_field,
        inline=False
    )

    # Footer
    embed_data.set_footer(text="🕒 Last updated")

    # Send update to announcement channel
    messages = [m async for m in announce_channel.history(limit=1)]
    if len(messages) == 0:
        await announce_channel.send(embed=embed_data)
    else:
        await messages[0].edit(embed=embed_data)

# On ready event
# Display bot guilds
@bot.event
async def on_ready():
    for guild in bot.guilds:
        print(
            f'{bot.user} is connected to the following guild(s):\n'
            f'{guild.name}(id: {guild.id})'
        )

    # Start fetching data
    await fetch_campus_data.start()

# (text) command not found error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, discord.ext.commands.CommandNotFound):
        return
    raise error

# Launch bot
bot.run(TOKEN)