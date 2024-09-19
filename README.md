# hkust-data-dashboard

Display real-time campus data in HKUST FYS Discord server.

## Usage

### Dependencies

- [discord.py](https://github.com/Rapptz/discord.py)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

### 🌐 Download the bot

1. Clone this repository:

```txt
git clone https://github.com/hkust-fys/hkust-data-dashboard.git
```

2. Set up virtual environment for the bot:

```txt
cd ~/hkust-data-dashboard
python -m venv ./.venv
```

3. Install the dependencies

```txt
pip install discord.py python-dotenv
```

### 🪪 Get an account for the bot

1. Create a bot user in the [Discord Developer Portal](https://discord.com/developers/applications) and copy its token.

    > [Here](https://realpython.com/how-to-make-a-discord-bot-python/#how-to-make-a-discord-bot-in-the-developer-portal) is a tutorial on how to create a bot on Discord.

2. Create a file named `.env` in the bot's directory with the following contents:

```env
# .env
DISCORD_TOKEN=<your-bot-token>
ANNOUNCE_CHANNEL_ID=<your-test-server-id>
```

Where `ADMIN_ID` is your Discord ID, and `ANNOUNCE_CHANNEL_ID` is the ID of the channel to display the data in.
  > Find your user and server IDs [here](https://support.discord.com/hc/en-us/articles/206346498-Where-can-I-find-my-User-Server-Message-ID).

### Run the bot

1. Run `bot.py`

```txt
python3 bot.py
```

### 🏁 And you're all set!
