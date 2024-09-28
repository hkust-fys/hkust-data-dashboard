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
import traceback
import dateutil.parser as dp
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

# Load API Keys
bus_queue_key = os.getenv('BUS_QUEUE_KEY')
ssc_key = os.getenv('SSC_KEY')
ppl_count_key = os.getenv('PPL_COUNT_KEY')

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

gmb_ids = {
    20013010: [
        {
            "gate": "S",
            "no": "11",
            "dest": "Choi Hung",
            "route": 2004791,
            "seq": 1
        }
    ],
    20012472: [
        {
            "gate": "N",
            "no": "11",
            "dest": "Hang Hau",
            "route": 2004791,
            "seq": 2
        }
    ],
    20013011: [
        {
            "gate": "S",
            "no": "11B",
            "dest": "Choi Hung",
            "route": 2004828,
            "seq": 1
        },
        {
            "gate": "S",
            "no": "11S",
            "dest": "Choi Hung",
            "route": 2004826,
            "seq": 1
        },
    ],
    20012474: [
        {
            "gate": "N",
            "no": "11M",
            "dest": "Hang Hau",
            "route": 2004825,
            "seq": 2
        },
        {
            "gate": "N",
            "no": "11S",
            "dest": "Po Lam",
            "route": 2004826,
            "seq": 2
        },
        {
            "gate": "N",
            "no": "12",
            "dest": "Po Lam",
            "route": 2004764,
            "seq": 1
        },
        {
            "gate": "N",
            "no": "12",
            "dest": "Sai Kung",
            "route": 2004764,
            "seq": 2
        }
    ]
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
    # Fetch bus queue from API
    bus_queue_hdr = {
        'Cache-Control': 'no-cache',
        'X-Apim-Subscription-Key': bus_queue_key
    }

    try:
        bus_queue = requests.request(
            "GET", 
            "https://hkust.azure-api.net/bus-queue-data/_search?sort=@timestamp:desc&size=1", 
            headers=bus_queue_hdr
        ).json()

        bus_queue_length_north = bus_queue['hits']['hits'][0]['_source']['north_waiting']
        bus_queue_length_south = bus_queue['hits']['hits'][0]['_source']['south_waiting']

        # Get update time of bus queue
        bus_queue_timestamp = bus_queue['hits']['hits'][0]['_source']['@timestamp']
        bus_queue_dt = dp.parse(bus_queue_timestamp)
        bus_queue_unix = int(bus_queue_dt.timestamp())
    except Exception as e:
        warnings.warn(f"Failed to connect to HKUST bus-queue API\nRetrying in next loop...")
        print(traceback.format_exc())

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
            eta_entry_raw = [x for x in requests.request("GET", f"https://data.etabus.gov.hk/v1/transport/kmb/stop-eta/{v}").json()['data'] if x['route'] == route and x['service_type'] == 1]

            eta_entry = []
            for x in eta_entry_raw:
                time = x['eta']
                if time != None:
                    time = str(round((datetime.datetime.fromisoformat(str(time)) - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60))
                else:
                    time = ""

                rmk = x['rmk_en']
                if rmk == "Scheduled Bus":
                    rmk = "*"
                elif "delayed" in rmk.lower():
                    rmk = "!"
                elif rmk != "":
                    rmk = ""
                
                eta_entry_elem = ""
                if time and int(time) <= 5:
                    eta_entry_elem = f"\u001b[0;41;37m{time}\u001b[0m"
                else:
                    eta_entry_elem = time
                eta_entry_elem += rmk

                eta_entry.append(eta_entry_elem)

            # Calculate minute difference from ISO arrival time
            # eta_entry = [str(round((datetime.datetime.fromisoformat(str(i)) - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60)) for i in eta_entry if i != None]

            # Highlight ETA if < 5 minutes away
            # eta_entry = [f"\u001b[0;41;37m{i}\u001b[0m" if int(i) <= 5 else i for i in eta_entry]

            if stop == "S":
                s_etas[f"🟥{route:<4} {dest}"] = eta_entry
            else:
                n_etas[f"🟥{route:<4} {dest}"] = eta_entry
    except Exception as e:
        warnings.warn(f"Failed to connect to KMB ETA API\nRetrying in next loop...")
        print(traceback.format_exc())
    
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

            n_etas[f"🟨{route:<4} {dest}"] = eta_entry
    except Exception as e:
        warnings.warn(f"Failed to connect to CTB ETA API\nRetrying in next loop...")
        print(traceback.format_exc())
    
    try:
        for stop_id, routes in gmb_ids.items():
            # Get list of ETAs at stop
            eta_stop = requests.request("GET", f"https://data.etagmb.gov.hk/eta/stop/{stop_id}").json()['data']

            # Get ETAs of each route at stop
            for route in routes:
                eta_route = [r['eta'] for r in eta_stop if r['route_id'] == route['route'] and r['route_seq'] == route['seq']][0]

                eta_entry = []
                for eta_route_elem in eta_route:
                    eta_entry_elem = str(eta_route_elem['diff'])
                    if eta_route_elem['diff'] <= 5:
                        eta_entry_elem = f"\u001b[0;41;37m{eta_route_elem['diff']}\u001b[0m"

                    if eta_route_elem['remarks_en']:
                        if "Delayed" in eta_route_elem['remarks_en']:
                            eta_entry_elem += "!"
                        elif "Scheduled" in eta_route_elem['remarks_en']:
                            eta_entry_elem += "*"
                    
                    eta_entry.append(eta_entry_elem)
                
                if eta_entry != []:
                    if route['gate'] == "S":
                        s_etas[f"🟩{route['no']:<4} {route['dest']}"] = eta_entry
                    else:
                        n_etas[f"🟩{route['no']:<4} {route['dest']}"] = eta_entry
    except Exception as e:
        warnings.warn(f"Failed to connect to GMB ETA API\nRetrying in next loop...")
        print(traceback.format_exc())
    
    # Fetch people count from API
    ppl_count_hdr = {
        'Cache-Control': 'no-cache',
        'X-Apim-Subscription-Key': ppl_count_key
    }

    try:
        ppl_count_raw = requests.request(
            "GET",
            "https://hkust.azure-api.net/people-count-pulse/_search?sort=@timestamp:desc&size=50",
            headers=ppl_count_hdr
        ).json()

        ppl_count = {}
        for h in ppl_count_raw['hits']['hits']:
            ppl_count[h['_source']['location']] = h['_source']['count']
        
        # Get update time of people count
        ppl_count_timestamp = ppl_count_raw['hits']['hits'][0]['_source']['@timestamp']
        ppl_count_dt = dp.parse(ppl_count_timestamp)
        ppl_count_unix = int(ppl_count_dt.timestamp())
        
        # Filter locations
        ppl_count_locs = ["LG1 Canteen", "McDonalds", "LG7 Canteen", "Chinese Restaurant", "LSK Canteen", "Seafront Cafeteria", "Starbucks", "North Gate Bus Stop", "South Gate Bus Stop", "Staff Bus Stop", "Lee Shau Kee Library 1/F", "Lee Shau Kee Library G/F", "Lee Shau Kee Library LG1", "Lee Shau Kee Library LG3", "Lee Shau Kee Library LG4", "wholeCampus"]
        ppl_count = {loc: ppl_count[loc] for loc in ppl_count_locs}
    except Exception as e:
        warnings.warn(f"Failed to connect to HKUST people-count-pulse API\nRetrying in next loop...")
        print(traceback.format_exc())
    
    # Fetch food waste count from API
    ssc_hdr = {
        'Cache-Control': 'no-cache',
        'X-Apim-Subscription-Key': ssc_key
    }

    try:
        ssc_raw = requests.request(
            "GET",
            "https://hkust.azure-api.net/ssc/ssc/food_waste/_search?sort=@timestamp:desc&size=100",
            headers=ssc_hdr
        ).json()

        ssc = {}
        for h in ssc_raw['hits']['hits']:
            ssc[h['_source']['location']] = h['_source']['weight']
        
        # Get update time of food waste
        ssc_timestamp = ssc_raw['hits']['hits'][0]['_source']['@timestamp']
        ssc_dt = dp.parse(ssc_timestamp)
        ssc_unix = int(ssc_dt.timestamp())
    except Exception as e:
        warnings.warn(f"Failed to connect to HKUST ssc API\nRetrying in next loop...")
        print(traceback.format_exc())

    # Compose embed using collected data
    embed_data = discord.Embed(
        title="Campus Dashboard",
        color=0xe0af68,
        timestamp=datetime.datetime.now()
    )

    # Bus queue
    bus_queue_field = f"```ansi\n"

    bus_queue_field += "🟥 KMB | 🟨 Citybus | 🟩 Minibus (inaccurate)\n"
    bus_queue_field += "* Scheduled departure, not real-time\n"
    bus_queue_field += "! Delayed\n\n"

    # bus_queue_field += f"North ({bus_queue_length_north} in queue)\n"
    bus_queue_field += f"North ({ppl_count['North Gate Bus Stop']} in queue)\n"  # Use people count instead
    bus_queue_field += f"{'🚍Route':<16}| ETA (mins)\n"
    for route, times in n_etas.items():
        bus_queue_field += f"{route:<16}| {', '.join(times)}\n"

    bus_queue_field += "\n"

    # bus_queue_field += f"South ({bus_queue_length_south} in queue)\n"
    bus_queue_field += f"South ({ppl_count['South Gate Bus Stop']} in queue)\n"  # Use people count instead
    bus_queue_field += f"{'🚍Route':<16}| ETA (mins)\n"
    for route, times in s_etas.items():
        bus_queue_field += f"{route:<16}| {', '.join(times)}\n"

    bus_queue_field += "```"

    embed_data.add_field(
        name=f"🚌 Bus stops (queue length updated <t:{bus_queue_unix}:R>)",
        value=bus_queue_field,
        inline=False
    )

    # People count
    ppl_count_field = f"```\n"
    for k, v in ppl_count.items():
        ppl_count_field += f"{k:<25}| {v}\n"
    ppl_count_field += "```"

    embed_data.add_field(
        name=f"👩 People count (updated <t:{ppl_count_unix}:R>)",
        value=ppl_count_field,
        inline=False
    )

    # Food waste
    food_waste_field = f"```\n"
    food_waste_field += f"{'Location':<25}| Weight (kg)\n"
    
    for k, v in ssc.items():
        food_waste_field += f"{k:<25}| {int(v)}\n"

    food_waste_field += "\n"
    food_waste_field += f"🐷 Happypig375 reminder: If you don't finish your food, I will finish you!\n"
    food_waste_field += "```"

    embed_data.add_field(
        name=f"🗑️ Food waste (updated <t:{ssc_unix}:R>)",
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
    embed_data.set_footer(text="🕒")

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