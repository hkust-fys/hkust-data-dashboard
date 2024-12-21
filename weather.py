import aiohttp

async def get_weather_data():
    url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
    weather_field = "```\n"  
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return "Error: Unable to fetch weather data"
                
                data = await response.json()
                
                weather_field += f"Last updated: {data['updateTime'][11:16]}\n"
                weather_field += "-" * 50 + "\n"
                
                if "warningMessage" in data and data["warningMessage"]:
                    weather_field += "⚠️ Warnings\n"
                    for warning in data["warningMessage"]:
                        weather_field += f"  • {warning}\n"
                    weather_field += "-" * 50 + "\n"
                
                weather_field += "🌡️ Temperature\n"
                for temp_data in data["temperature"]["data"]:
                    if temp_data["place"] == "Sai Kung":
                        weather_field += f"  Sai Kung (HKUST): {temp_data['value']}°C\n"
                        break
                
                weather_field += "\n🌧️ Rainfall\n"
                for rain_data in data["rainfall"]["data"]:
                    if rain_data["place"] == "Sai Kung":
                        weather_field += f"  Sai Kung: {rain_data['max']}mm\n"
                        break
                
                weather_field += "-" * 50
                weather_field += "```"
                return weather_field
                
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

