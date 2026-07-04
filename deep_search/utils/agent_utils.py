import requests

def get_have_internet():
    try:
        response = requests.get("http://www.bing.com", timeout=5)
        # If the request was successful, status code will be 200
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False

def current_datetime():
    from datetime import datetime
    import tzlocal

    # Get the local time zone
    local_timezone = tzlocal.get_localzone()

    # Get the current time in the local time zone
    now = datetime.now(local_timezone)

    # Format the date, time, and time zone
    formatted_date_time = now.strftime("%A, %B %d, %Y - %I:%M %p %Z")

    # Print the formatted date, time, and time zone
    return "对于当前用户查询：当前日期、时间和本地时区：%s。请注意，一些 API 可能包含来自不同时区的数据，因此可能反映不同的日期。" % formatted_date_time
