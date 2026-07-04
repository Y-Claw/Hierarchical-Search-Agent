IMAGE_EXTENSIONS = {'.png': 'PNG', '.apng': 'PNG', '.blp': 'BLP', '.bmp': 'BMP', '.dib': 'DIB', '.bufr': 'BUFR',
                    '.cur': 'CUR', '.pcx': 'PCX', '.dcx': 'DCX', '.dds': 'DDS',
                    # '.ps': 'EPS', '.eps': 'EPS',
                    '.fit': 'FITS', '.fits': 'FITS', '.fli': 'FLI', '.flc': 'FLI', '.fpx': 'FPX', '.ftc': 'FTEX',
                    '.ftu': 'FTEX', '.gbr': 'GBR', '.gif': 'GIF', '.grib': 'GRIB',
                    # '.h5': 'HDF5', '.hdf': 'HDF5',
                    '.jp2': 'JPEG2000', '.j2k': 'JPEG2000', '.jpc': 'JPEG2000', '.jpf': 'JPEG2000', '.jpx': 'JPEG2000',
                    '.j2c': 'JPEG2000', '.icns': 'ICNS', '.ico': 'ICO', '.im': 'IM', '.iim': 'IPTC', '.jfif': 'JPEG',
                    '.jpe': 'JPEG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.tif': 'TIFF', '.tiff': 'TIFF', '.mic': 'MIC',
                    #'.mpg': 'MPEG', '.mpeg': 'MPEG',
                    '.mpo': 'MPO', '.msp': 'MSP', '.palm': 'PALM', '.pcd': 'PCD',
                    #'.pdf': 'PDF',
                     '.pxr': 'PIXAR', '.pbm': 'PPM', '.pgm': 'PPM', '.ppm': 'PPM', '.pnm': 'PPM',
                    '.psd': 'PSD', '.qoi': 'QOI', '.bw': 'SGI', '.rgb': 'SGI', '.rgba': 'SGI', '.sgi': 'SGI',
                    '.ras': 'SUN', '.tga': 'TGA', '.icb': 'TGA', '.vda': 'TGA', '.vst': 'TGA', '.webp': 'WEBP',
                    '.wmf': 'WMF', '.emf': 'WMF', '.xbm': 'XBM', '.xpm': 'XPM'}

FILE_TYPE = {
            ".pdb": "text",
            ".pdf": "text",
            ".docx": "text",
            ".doc": "text",
            ".txt": "text",
            ".md": "text",
            ".pptx": "text",
            ".ppt": "text",
            ".xls": "text",
            ".xlsx": "text",
            ".csv": "text",
            ".json": "text",
            ".html": "text",
            ".htm": "text",
            ".xml": "text",
            ".css": "text",
            ".js": "text",
            ".py": "text",
            ".jsonld": "text",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".mp4": "video",
            ".avi": "video",
            ".mov": "video",
            ".mp3": "audio",
            ".m4a": "audio",
            ".wav": "audio",
            ".flac": "audio",
        }

OPENAI_MODEL = ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "o1-2024-12-17", "o3-mini-2025-01-31", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "openrouter:gpt-4o-mini-2024-07-18"]
CLAUDE_MODEL = ["claude-3-5-sonnet-20240620", "claude-3-5-haiku-20240307", "claude-3-3-4-20241022", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219-think"]
QWQ_MODEL = ["QwQ-32B", "Qwen3-32B"]
QWEN_MODEL = ["Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct-128k", "Qwen2.5-14B-Instruct-128k", "Qwen2.5-14B-Instruct","Qwen2.5-7B-Instruct"]

REACT_MODEL = ["QwQ-32B", "Qwen3-32B", "claude-3-7-sonnet-20250219-thinking"]
FUNC_ONLY_MODEL = ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "o1-2024-12-17", "o3-mini-2025-01-31", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "anthropic-claude-3-7-sonnet-20250219", "Qwen2.5-32B-Instruct-128k"]
