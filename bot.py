# bot.py
# Run on 09 or 39th second of a minute to minimize bus ETA delay
import discord
from discord import app_commands
from discord.ext import commands, tasks

import os
import warnings
import asyncio
import requests
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

# Bus stop IDs
kmb_ids = {
    "S_91_Dia. Hill": "B002CEF0DBC568F5",
    "S_91M_Dia. Hill": "B002CEF0DBC568F5",
    "S_91P_Choi Hung": "E9018F8A7E096544",
    "S_291P_Mong Kok": "E9018F8A7E096544",
    "N_91_CWB": "3592A0182BF020C7",
    "N_91M_Po Lam": "B3E60EE895DBBF06",
}
ctb_ids = {
    "O_792M_Sai Kung": "003130",
    "I_792M_TKO": "003130"
}

# define bot
intents = discord.Intents.default()
bot = commands.Bot(
    command_prefix = "d.",
    intents=intents,
    activity=discord.Activity(type=discord.ActivityType.watching, name="campus data"),
    help_command=None
)

# Fetch campus data every 5 minutes
@tasks.loop(seconds=30)
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

    # Fetch transit ETAs from API
    n_etas = {}
    s_etas = {}
    try:
        for k, v in kmb_ids.items():
            stop = k.split("_")[0]
            route = k.split("_")[1]
            dest = k.split("_")[2]

            # eta_entry = [str(round((datetime.datetime.fromisoformat(str(x['eta'])) - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60)) for x in requests.request("GET", f"https://data.etabus.gov.hk/v1/transport/kmb/stop-eta/{v}").json()['data'] if x['route'] == route]

            # Get list of ETAs at stop with ID v, filter to route in k and extract ISO arrival times
            eta_entry = [x['eta'] for x in requests.request("GET", f"https://data.etabus.gov.hk/v1/transport/kmb/stop-eta/{v}").json()['data'] if x['route'] == route and x['service_type'] == 1]

            # Calculate minute difference from ISO arrival time
            eta_entry = [str(round((datetime.datetime.fromisoformat(str(i)) - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60)) for i in eta_entry if i != None]

            # Highlight ETA if < 5 minutes away
            eta_entry = [f"\u001b[0;41;37m{i}\u001b[0m" if int(i) <= 5 else i for i in eta_entry]

            if stop == "S":
                s_etas[f"{route:<4} {dest}"] = eta_entry
            else:
                n_etas[f"{route:<4} {dest}"] = eta_entry
    except Exception as e:
        warnings.warn(f"Failed to connect to KMB ETA API\nRetrying in next loop...")
        print(e)
    
    try:
        for k, v in ctb_ids.items():
            dir = k.split("_")[0]
            route = k.split("_")[1]
            dest = k.split("_")[2]

            # Get list of ETAs at stop with ID v and route route, filter to direction in dir and extract ISO arrival times
            eta_entry = [x['eta'] for x in requests.request("GET", f"https://rt.data.gov.hk/v2/transport/citybus/eta/CTB/{v}/{route}").json()['data'] if x['dir'] == dir]

            # Calculate minute difference from ISO arrival time
            eta_entry = [str(round((datetime.datetime.fromisoformat(str(i)) - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60)) for i in eta_entry if i != None]

            # Highlight ETA if < 5 minutes away
            eta_entry = [f"\u001b[0;41;37m{i}\u001b[0m" if int(i) <= 5 else i for i in eta_entry]

            n_etas[f"{route:<4} {dest}"] = eta_entry
    except Exception as e:
        warnings.warn(f"Failed to connect to CTB ETA API\nRetrying in next loop...")
        print(e)

    # Compose embed using collected data
    embed_data = discord.Embed(
        title="Campus Dashboard",
        color=0xe0af68,
        timestamp=datetime.datetime.now()
    )

    # Bus queue
    bus_queue_field = f"```ansi\n"
    bus_queue_field += f"North (42 in queue)\n"
    bus_queue_field += f"{'Route':<15}| ETA (mins)\n"
    for route, times in n_etas.items():
        bus_queue_field += f"{route:<15}| {', '.join(times)}\n"

    bus_queue_field += "\n"

    bus_queue_field += f"South (91 in queue)\n"
    bus_queue_field += f"{'Route':<15}| ETA (mins)\n"
    for route, times in s_etas.items():
        bus_queue_field += f"{route:<15}| {', '.join(times)}\n"

    bus_queue_field += "```"

    embed_data.add_field(
        name="🚏 Bus times",
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