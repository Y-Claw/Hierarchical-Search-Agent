import re
import os
from collections import defaultdict

def extract_xml_tags(full_text, tags=['name', 'page']):
    results_dict = {k: None for k in tags}
    for tag in tags:
        pattern = fr'<{tag}>(.*?)</{tag}>'
        values = re.findall(pattern, full_text, re.DOTALL)
        if values:
            results_dict[tag] = values[0]
    return results_dict

def generate_unique_filename(name_page_dict):
    name = name_page_dict.get('name', 'unknown') or 'unknown'
    page = name_page_dict.get('page', '0') or '0'

    # Remove file extension if present
    name = os.path.splitext(name)[0]

    # Clean the name: remove any characters that aren't alphanumeric, underscore, or hyphen
    clean_name = re.sub(r"[^\w\-]", "_", name)

    # Create the unique filename
    unique_filename = f"{clean_name}_page_{page}.txt"

    return unique_filename, clean_name, page


def deduplicate_filenames(filenames):
    seen = defaultdict(int)
    result = []
    needs_renumbering = set()

    # First pass: identify duplicates and mark for renumbering
    for filename in filenames:
        if seen[filename] > 0:
            needs_renumbering.add(filename)
        seen[filename] += 1

    # Reset the seen counter for the second pass
    seen = defaultdict(int)

    # Second pass: rename files
    for filename in filenames:
        base, ext = filename.rsplit(".", 1)
        if filename in needs_renumbering:
            new_filename = f"{base}_chunk_{seen[filename]}.{ext}"
        else:
            new_filename = filename

        seen[filename] += 1
        result.append(new_filename)

    return result