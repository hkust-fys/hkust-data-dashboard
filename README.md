# hkust-data-dashboard

Display real-time campus data in HKUST FYS Discord server.

## Usage

### 🌐 Download the bot

1.
_For Colab development_ Open the source code for the Bot [here](https://github.com/hkust-fys/hkust-data-dashboard/blob/main/bot.py)

_For Discord Bot production_ Clone this repository:

```txt
git clone https://github.com/hkust-fys/hkust-data-dashboard.git
```

2. _For Colab development_ Copy the whole script to a code cell on [Google Colab](https://colab.research.google.com).

_For Discord Bot production_ Set up virtual environment for the bot:

```txt
cd ~/hkust-data-dashboard
python -m venv ./.venv
```

3. Run the script once for setting up dependencies and the `.env` file. This message will be printed to the console:
```
Dependencies and .env file set up. Please edit the .env file with valid input.
```

These packages will be installed or updated:

- [discord.py](https://github.com/Rapptz/discord.py)
- [python-dotenv](https://pypi.org/project/python-dotenv)

### 🪪 Get an account for the bot

4.
_For Colab development_ Create a webhook in your debug channel and copy its Webhook URL.

> [Here](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks) is a tutorial on how to create a webhook on Discord.

_For Discord Bot production_ Create a bot user in the [Discord Developer Portal](https://discord.com/developers/applications) and copy its token.

> [Here](https://realpython.com/how-to-make-a-discord-bot-python/#how-to-make-a-discord-bot-in-the-developer-portal) is a tutorial on how to create a bot on Discord.

5. Edit the `.env` in the bot's directory:

- `BUS_QUEUE_KEY`, `SSC_KEY`, `PPL_COUNT_KEY` are all from HKUST. Ask us to lend you our token.
- `DEV_WEBHOOK` is for Google Colab development purposes. Edit it to a valid Discord webhook URL or comment it out for Discord Bot production.
- `DISCORD_TOKEN` and `ANNOUNCE_CHANNEL_ID` will only be read if `DEV_WEBHOOK` is commented out. 
`DISCORD_TOKEN` is the Discord Bot token. `ANNOUNCE_CHANNEL_ID` is the ID of the channel to display the data in.
> Find your user and server IDs [here](https://support.discord.com/hc/en-us/articles/206346498-Where-can-I-find-my-User-Server-Message-ID).

### Run the bot

6.
_For Colab development_ Ctrl+Enter anywhere in the code cell you use will run the entire script.

_For Discord Bot production)
Run `bot.py`

```txt
python3 bot.py
```

### 🏁 And you're all set!
