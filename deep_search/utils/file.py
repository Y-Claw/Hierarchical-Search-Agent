import base64
import cv2
import mimetypes
import uuid
from io import BytesIO
import re
import os
import requests
from urllib.parse import urlparse
from pathlib import Path

def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def base64_to_img(img_str, output_path):
    """
    Convert a base64 string to an image or video file.

    :param img_str: The base64 encoded string with the image or video data.
    :param output_path: The path (without extension) where the output file will be saved.
    :return: The path to the saved file.
    """
    if img_str.startswith("b'"):
        # check if was a string of bytes joined like when str_bytes=True in above function
        img_str = img_str[2:-1]  # This removes the first b' and the last '

    # Split the string on "," to separate the metadata from the base64 data
    meta, base64_data = img_str.split(",", 1)
    # Extract the format from the metadata
    img_format = meta.split(';')[0].split('/')[-1]
    # Decode the base64 string to bytes
    img_bytes = base64.b64decode(base64_data)
    # Create output file path with the correct format extension
    output_file = f"{output_path}.{img_format}"
    # Write the bytes to a file
    with open(output_file, "wb") as f:
        f.write(img_bytes)
    print(f"Image saved to {output_file} with format {img_format}")
    return output_file

def download_image(image_url, save_dir):
    """
    Download an image from a URL and save it to a specified directory.

    Parameters:
    image_url (str): The URL of the image to download.
    save_dir (str): The directory path where the image will be saved.

    Returns:
    str or None: The file path where the image was saved, or None if an error occurred.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful

        # Extract the file name from the URL
        parsed_url = urlparse(image_url)
        file_name = os.path.basename(parsed_url.path)

        # Create the full save path
        save_path = os.path.join(save_dir, file_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save the image
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return None

def make_image_url(base64_image):
    return f"data:image/png;base64,{base64_image}"

def get_youtube_urls():
    # https://www.netify.ai/resources/applications/youtube
    base = ['googlevideo.com',
            'video.google.com',
            'video.l.google.com',
            'wide-youtube.l.google.com',
            'youtu.be',
            'youtube.ae',
            'youtube.al',
            'youtube.am',
            'youtube.at',
            'youtube.az',
            'youtube.ba',
            'youtube.be',
            'youtube.bg',
            'youtube.bh',
            'youtube.bo',
            'youtube.by',
            'youtube.ca',
            'youtube.cat',
            'youtube.ch',
            'youtube.cl',
            'youtube.co',
            'youtube.co.ae',
            'youtube.co.at',
            'youtube.co.cr',
            'youtube.co.hu',
            'youtube.co.id',
            'youtube.co.il',
            'youtube.co.in',
            'youtube.co.jp',
            'youtube.co.ke',
            'youtube.co.kr',
            'youtube.com',
            'youtube.co.ma',
            'youtube.com.ar',
            'youtube.com.au',
            'youtube.com.az',
            'youtube.com.bd',
            'youtube.com.bh',
            'youtube.com.bo',
            'youtube.com.br',
            'youtube.com.by',
            'youtube.com.co',
            'youtube.com.do',
            'youtube.com.ec',
            'youtube.com.ee',
            'youtube.com.eg',
            'youtube.com.es',
            'youtube.com.gh',
            'youtube.com.gr',
            'youtube.com.gt',
            'youtube.com.hk',
            'youtube.com.hn',
            'youtube.com.hr',
            'youtube.com.jm',
            'youtube.com.jo',
            'youtube.com.kw',
            'youtube.com.lb',
            'youtube.com.lv',
            'youtube.com.ly',
            'youtube.com.mk',
            'youtube.com.mt',
            'youtube.com.mx',
            'youtube.com.my',
            'youtube.com.ng',
            'youtube.com.ni',
            'youtube.com.om',
            'youtube.com.pa',
            'youtube.com.pe',
            'youtube.com.ph',
            'youtube.com.pk',
            'youtube.com.pt',
            'youtube.com.py',
            'youtube.com.qa',
            'youtube.com.ro',
            'youtube.com.sa',
            'youtube.com.sg',
            'youtube.com.sv',
            'youtube.com.tn',
            'youtube.com.tr',
            'youtube.com.tw',
            'youtube.com.ua',
            'youtube.com.uy',
            'youtube.com.ve',
            'youtube.co.nz',
            'youtube.co.th',
            'youtube.co.tz',
            'youtube.co.ug',
            'youtube.co.uk',
            'youtube.co.ve',
            'youtube.co.za',
            'youtube.co.zw',
            'youtube.cr',
            'youtube.cz',
            'youtube.de',
            'youtube.dk',
            'youtubeeducation.com',
            'youtube.ee',
            'youtubeembeddedplayer.googleapis.com',
            'youtube.es',
            'youtube.fi',
            'youtube.fr',
            'youtube.ge',
            'youtube.googleapis.com',
            'youtube.gr',
            'youtube.gt',
            'youtube.hk',
            'youtube.hr',
            'youtube.hu',
            'youtube.ie',
            'youtubei.googleapis.com',
            'youtube.in',
            'youtube.iq',
            'youtube.is',
            'youtube.it',
            'youtube.jo',
            'youtube.jp',
            'youtubekids.com',
            'youtube.kr',
            'youtube.kz',
            'youtube.la',
            'youtube.lk',
            'youtube.lt',
            'youtube.lu',
            'youtube.lv',
            'youtube.ly',
            'youtube.ma',
            'youtube.md',
            'youtube.me',
            'youtube.mk',
            'youtube.mn',
            'youtube.mx',
            'youtube.my',
            'youtube.ng',
            'youtube.ni',
            'youtube.nl',
            'youtube.no',
            'youtube-nocookie.com',
            'youtube.pa',
            'youtube.pe',
            'youtube.ph',
            'youtube.pk',
            'youtube.pl',
            'youtube.pr',
            'youtube.pt',
            'youtube.qa',
            'youtube.ro',
            'youtube.rs',
            'youtube.ru',
            'youtube.sa',
            'youtube.se',
            'youtube.sg',
            'youtube.si',
            'youtube.sk',
            'youtube.sn',
            'youtube.soy',
            'youtube.sv',
            'youtube.tn',
            'youtube.tv',
            'youtube.ua',
            'youtube.ug',
            'youtube-ui.l.google.com',
            'youtube.uy',
            'youtube.vn',
            'yt3.ggpht.com',
            'yt.be',
            'ytimg.com',
            'ytimg.l.google.com',
            'ytkids.app.goo.gl',
            'yt-video-upload.l.google.com']

    url_prefixes_youtube1 = []
    for x in base:
        url_prefixes_youtube1.extend([
            # '%s/watch?v=' % x,
            '%s' % x,
            # '%s/shorts/' % x,
        ])
    return set(url_prefixes_youtube1)

url_prefixes_youtube = get_youtube_urls()

url_pattern = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
    r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def check_input_type(input_string):
    """
    Check if the input string is a file path, URL, or a base64 encoded image.

    Parameters:
    input_string (str): The input string to check.

    Returns:
    str: 'file', 'url', 'base64', or 'unknown' based on the input type.
    """
    if not isinstance(input_string, str):
        return 'unknown'

    # Check if the input string looks like a base64 encoded image
    if input_string.startswith("data:image/") or input_string.startswith("b'data:image/"):
        return 'base64'

    if re.match(url_pattern, input_string):
        return 'url'

    is_youtube = any(
        input_string.replace('http://', '').replace('https://', '').replace('www.', '').startswith(prefix) for prefix in
        url_prefixes_youtube)
    if is_youtube:
        return 'youtube'

    # Check if the input is a file path
    if os.path.isfile(input_string):
        return 'file'

    return 'unknown'

def video_to_base64(video_path):
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames