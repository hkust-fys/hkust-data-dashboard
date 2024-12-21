import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
import re
import json

class BusRoute:
    def __init__(self, route_name, timings=None, fare=0.0, from_ust=False):
        self.route_name = route_name
        self.timings = timings if timings else []
        self.fare = fare
        self.from_ust = from_ust 

    def add_timing(self, timing):
        if timing not in self.timings:
            self.timings.append(timing)
            self.timings.sort()

    def __str__(self):
        direction = "From HKUST" if self.from_ust else "To HKUST"
        return f"Route: {self.route_name}\nDirection: {direction}\nTimings: {', '.join(self.timings)}\nFare: ${self.fare}"
    
    def to_dict(self):
        return {
            "route_name": self.route_name,
            "timings": self.timings,
            "fare": self.fare,
            "from_ust": self.from_ust
        }
    
def save_to_json(students_and_staff, students_only, filename="bus_routes.json"):
    json_data = {
        "students_and_staff": {
            key: route.to_dict() 
            for key, route in students_and_staff.items()
        },
        "students_only": {
            key: route.to_dict() 
            for key, route in students_only.items()
        },
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def is_date_in_range(current_date, date_range_str):
    if "23 December 2024 to 28 January 2025" in date_range_str:
        return True if current_date >= datetime(2024, 12, 23) and current_date <= datetime(2025, 1, 28) else False
    elif "2 September to 20 December 2024 & 3 February to 29 May 2025" in date_range_str:
        return (
            (current_date >= datetime(2024, 9, 2) and current_date <= datetime(2024, 12, 20)) or
            (current_date >= datetime(2025, 2, 3) and current_date <= datetime(2025, 5, 29))
        )
    return False

def extract_times(text):
    return re.findall(r'\d{2}:\d{2}', text)

def extract_fare(text):
    fare_match = re.search(r'\$(\d+\.?\d*)', text)
    if fare_match:
        return round(float(fare_match.group(1)), 1)
    else:
        return 0.0


def parse_shuttle_bus_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    students_and_staff = {}
    students_only = {}
    
    current_date = datetime.now()
    
    staff_section = soup.find('h3', text='Students and Staff')
    if staff_section:
 
        red_morning = staff_section.find_next('p', style=lambda x: x and 'color:red' in x)
        green_morning = staff_section.find_next('p', style=lambda x: x and 'color:green' in x)
        
 
        afternoon_header = soup.find('strong', text=lambda x: x and 'Afternoon Schedule' in str(x))
        if afternoon_header:
            red_afternoon = afternoon_header.find_parent('p').find_next('p', style=lambda x: x and 'color:red' in x)
            green_afternoon = afternoon_header.find_parent('p').find_next('p', style=lambda x: x and 'color:green' in x)

            morning_routes_ul = staff_section.find_next('ul')
            for li in morning_routes_ul.find_all('li'):
                text = li.get_text(strip=True)
                route_name = re.search(r'From:\s*([^-]+)', text).group(1).strip()
                fare = extract_fare(text)
                
     
                if route_name not in students_and_staff:
                    students_and_staff[route_name] = BusRoute(route_name, [], fare, from_ust=False)
                

                if current_date >= datetime(2024, 12, 23) and current_date <= datetime(2025, 1, 28):
                    morning_times = extract_times(red_morning.text)
                else:
                    morning_times = extract_times(green_morning.text)
                
                for time in morning_times:
                    students_and_staff[route_name].add_timing(time)


            afternoon_routes_ul = afternoon_header.find_parent('p').find_next('ul')
            for li in afternoon_routes_ul.find_all('li'):
                text = li.get_text(strip=True)
                route_name = re.search(r'To:\s*([^-]+)', text).group(1).strip()
                fare = extract_fare(text)
                

                route_key = f"{route_name}_from_ust"  
                students_and_staff[route_key] = BusRoute(route_name, [], fare, from_ust=True)
                

                if current_date >= datetime(2024, 12, 23) and current_date <= datetime(2025, 1, 28):
                    afternoon_times = extract_times(red_afternoon.text)
                else:
                    afternoon_times = extract_times(green_afternoon.text)
                
                for time in afternoon_times:
                    students_and_staff[route_key].add_timing(time)


    student_section = soup.find('h3', text=lambda t: t and 'Student Only' in t)
    if student_section and not (current_date >= datetime(2024, 12, 23) and current_date <= datetime(2025, 1, 28)):

        morning_lists = student_section.find_next('ul')
        if morning_lists:
            for li in morning_lists.find_all('li'):
                text = li.get_text(strip=True)
                if 'From:' in text:
                    route_name = re.search(r'From:\s*([^-]+)', text).group(1).strip()
                    fare = extract_fare(text)
                    timings = extract_times(text)
                    
                    students_only[route_name] = BusRoute(route_name, [], fare, from_ust=False)
                    for time in timings:
                        students_only[route_name].add_timing(time)

        afternoon_header = soup.find('strong', text='Afternoon Schedule')
        if afternoon_header:
            current = afternoon_header
            while current:
                current = current.find_next()
                if current and current.name == 'ul':
                    for li in current.find_all('li'):
                        text = li.get_text(strip=True)
                        if 'To:' in text:
                            if 'North Point' in text and 'Causeway Bay' in text:
                                route_name = 'North Point'
                            else:
                                route_name = re.search(r'To:\s*([^-(]+)', text).group(1).strip()
                            
                            fare = extract_fare(text)
                            timings = extract_times(text)
                            
                            route_key = f"{route_name}_from_ust"
                            students_only[route_key] = BusRoute(route_name, [], fare, from_ust=True)
                            for time in timings:
                                students_only[route_key].add_timing(time)

    return students_and_staff, students_only

async def fetch():
    print("Fetching the latest bus routes...")
    url = "https://cso.ust.hk/tran/stud_sh_b"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html_content = await response.text()
                    students_and_staff, students_only = parse_shuttle_bus_data(html_content)       
                  
                    save_to_json(students_and_staff, students_only)
                    
                    print("Bus routes have been saved to bus_routes.json")
                    
                    return students_and_staff, students_only
                else:
                    print(f"Failed to fetch data. Status code: {response.status}")
                    return {}, {}
                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}, {}


def format_route_name(route_name):
    main_name = re.sub(r'\s*\((.*?)\)', '', route_name).strip()
    extra_info = re.search(r'\((.*?)\)', route_name)
    if extra_info:
        return f"{main_name} - \n  {extra_info.group(1)}"
    return main_name

def format_times_with_wrap(times):
    filtered_times = []
    has_later_times = False
    for t in times:
        if int(t) <= 60:
            filtered_times.append(f"\u001b[0;41;37m{t}\u001b[0m" if int(t) <= 5 else t)
        else:
            has_later_times = True
        
    if has_later_times:
        filtered_times.append("...")
        
    return ', '.join(filtered_times)



async def load_or_fetch_shuttle_data():
    current_time = datetime.now()
    students_and_staff = {}
    students_only = {}

    try:
        print("Loading bus routes from JSON file...")
        with open('bus_routes.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Bus routes loaded successfully!")
        last_updated = datetime.strptime(data['last_updated'], "%Y-%m-%d %H:%M:%S").date()
        today = current_time.date()
        
        if last_updated == today:
            
            for k, v in data['students_and_staff'].items():
                route = BusRoute(v['route_name'], [], v['fare'], v['from_ust'])
                for time_str in v['timings']:
                    time_obj = datetime.strptime(time_str, "%H:%M").replace(
                        year=current_time.year,
                        month=current_time.month,
                        day=current_time.day
                    )
                    if time_obj > current_time:
                        minutes_until = int((time_obj - current_time).total_seconds() / 60)
                        route.timings.append(str(minutes_until))
                if route.timings: 
                    students_and_staff[k] = route

            for k, v in data['students_only'].items():
                route = BusRoute(v['route_name'], [], v['fare'], v['from_ust'])
                for time_str in v['timings']:
                    time_obj = datetime.strptime(time_str, "%H:%M").replace(
                        year=current_time.year,
                        month=current_time.month,
                        day=current_time.day
                    )
                    if time_obj > current_time:
                        minutes_until = int((time_obj - current_time).total_seconds() / 60)
                        route.timings.append(str(minutes_until))
                if route.timings: 
                    students_only[k] = route
        else:
            students_and_staff, students_only = await fetch()
            
    except (FileNotFoundError, json.JSONDecodeError):
        students_and_staff, students_only = await fetch()
    
    shuttle_bus_field = "```ansi\n"
    shuttle_bus_field += "🚌 HKUST Shuttle Bus Schedule\n"
    shuttle_bus_field += "🟦 Students & Staff | 🟨 Students Only\n"
    shuttle_bus_field += f"Current time: {current_time.strftime('%H:%M')}\n"
    shuttle_bus_field += "* Scheduled departure, not real-time.\n\n"
    shuttle_bus_field += "Times shown in minutes from now\n"
    if current_time >= datetime(2024, 12, 23) and current_time <= datetime(2025, 1, 28):
        shuttle_bus_field += "* Holiday schedule active\n"
    shuttle_bus_field += "\n"
    

    shuttle_bus_field += "TO HKUST\n"
    shuttle_bus_field += f"{'🚍Route':<20}| {'Fare':<6}| ETA\n"
    shuttle_bus_field += "-" * 50 + "\n"

    if len(students_and_staff) == 0:
        shuttle_bus_field += f"🟦 No more buses\n"
    for route in students_and_staff.values():
        if not route.from_ust and route.timings:
            times = route.timings
            if len(times) > 0:
                formatted_name = format_route_name(route.route_name)
                formatted_eta = format_times_with_wrap(times)
                if '\n' in formatted_name:
                    name_parts = formatted_name.split('\n')
                    shuttle_bus_field += f"🟦{name_parts[0]:<19}\n"
                    shuttle_bus_field += f"{name_parts[1]:<19}| ${route.fare:<5}| {formatted_eta}\n"
                else:
                    shuttle_bus_field += f"🟦{formatted_name:<19}| ${route.fare:<5}| {formatted_eta}\n"
    
    shuttle_bus_field += "\nFROM HKUST\n"
    shuttle_bus_field += f"{'🚍Route':<20}| {'Fare':<6}| ETA\n"
    shuttle_bus_field += "-" * 50 + "\n"
    
    if len(students_and_staff) == 0:
        shuttle_bus_field += f"🟦 No more buses\n"
    for route in students_and_staff.values():
        if route.from_ust and route.timings:
            times = route.timings
            if len(times) > 0:
                formatted_name = format_route_name(route.route_name)
                formatted_eta = format_times_with_wrap(times)
                if '\n' in formatted_name:
                    name_parts = formatted_name.split('\n')
                    shuttle_bus_field += f"🟦{name_parts[0]:<19}\n"
                    shuttle_bus_field += f"{name_parts[1]:<19}| ${route.fare:<5}| {formatted_eta}\n"
                else:
                    shuttle_bus_field += f"🟦{formatted_name:<19}| ${route.fare:<5}| {formatted_eta}\n"
            else:
                shuttle_bus_field += f"🟦{formatted_name:<19}| ${route.fare:<5}| No more buses\n"
    
    if len(students_only) == 0:
        shuttle_bus_field += f"🟨 No more buses\n"
    for route in students_only.values():
        if route.from_ust and route.timings:
            times = route.timings
            if len(times) > 0:
                formatted_name = format_route_name(route.route_name)
                formatted_eta = format_times_with_wrap(times)
                shuttle_bus_field += f"🟨{formatted_name:<19}| ${route.fare:<5}| {formatted_eta}\n"
            else:
                shuttle_bus_field += f"🟨{formatted_name:<19}| ${route.fare:<5}| No more buses\n"

    shuttle_bus_field += "\n... means there are still later buses"
    shuttle_bus_field += "```"

    return shuttle_bus_field

## This literally exists just to test if it works thru the command line 

def main():
    
    students_and_staff, students_only = fetch()


    print("=== Students and Staff Services ===")
    for route in students_and_staff.values():
        print(f"\n{route}")
    
    print("\n=== Student Only Services ===")
    for route in students_only.values():
        print(f"\n{route}")

if __name__ == "__main__":
    main()