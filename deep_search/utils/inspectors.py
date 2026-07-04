import base64
import json
import mimetypes
import os
import uuid
import shutil
import textwrap
from io import BytesIO
from typing import Optional

import requests

from search_toolkit.converters.mdconvert import MarkdownConverter
from deep_search.utils.api import call_openai
from deep_search.utils.file import encode_image

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('ONE_API_KEY')}"}

def get_image_description(file_name: str, question: str, visual_inspection_tool) -> str:
    prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the image."""
    return visual_inspection_tool(image_path=file_name, question=prompt)


def get_document_description(file_path: str, question: str, document_inspection_tool) -> str:
    prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the document."""
    return document_inspection_tool.forward_initial_exam_mode(file_path=file_path, question=prompt)


def get_single_file_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    file_extension = file_path.split(".")[-1]
    if file_extension in ["png", "jpg", "jpeg"]:
        file_description = f" - Attached image: {file_path}"
        file_description += (
            f"\n     -> Image description: {get_image_description(file_path, question, visual_inspection_tool)}"
        )
        return file_description
    elif file_extension in ["pdf", "xls", "xlsx", "docx", "doc", "xml"]:
        file_description = f" - Attached document: {file_path}"
        image_path = file_path.split(".")[0] + ".png"
        if os.path.exists(image_path):
            description = get_image_description(image_path, question, visual_inspection_tool)
        else:
            description = get_document_description(file_path, question, document_inspection_tool)
        file_description += f"\n     -> File description: {description}"
        return file_description
    elif file_extension in ["mp3", "m4a", "wav"]:
        return f" - Attached audio: {file_path}"
    else:
        return f" - Attached file: {file_path}"

def get_zip_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    folder_path = file_path.replace(".zip", "")
    os.makedirs(folder_path, exist_ok=True)
    shutil.unpack_archive(file_path, folder_path)

    prompt_use_files = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            prompt_use_files += "\n" + textwrap.indent(
                get_single_file_description(file_path, question, visual_inspection_tool, document_inspection_tool),
                prefix="    ",
            )
    return prompt_use_files



def visualizer(image_path: str, question: Optional[str] = None, model: str="gpt-4o-2024-11-20") -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """

    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."
    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")

    mime_type, _ = mimetypes.guess_type(image_path)
    base64_image = encode_image(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
    }

    api_base = os.getenv("ONE_API_BASE")
    response = requests.post(api_base + "/chat/completions", headers=headers, json=payload)
    try:
        output = response.json()["choices"][0]["message"]["content"]
    except Exception:
        raise Exception(f"Response format unexpected: {response.json()}")

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output

class TextInspectorTool:
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()

    def __init__(self, model: str, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward_initial_exam_mode(self, file_path, question):
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return call_openai(self.model, messages)

    def forward(self, file_path, question: Optional[str] = None) -> str:
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return call_openai(self.model, messages)

    def __call__(self, file_path, question: Optional[str] = None) -> str:
        return self.forward(file_path, question)

from deep_search.utils.enums import FILE_TYPE
from deep_search.utils.file import video_to_base64

def vqa_with_llm(query: str, file: str, model: str="gpt-4o-2024-11-20") -> str:
    if not os.path.isfile(file):
        return None, f"文件 {file} 不存在，请检查文件路径是否正确。如果你想访问网络资源，请使用 visit_web_page.py 工具。"

    file_type = os.path.splitext(file)[1]
    file_type = FILE_TYPE[file_type]
    if file_type == "image":
        mime_type, _ = mimetypes.guess_type(file)
        base64_image = encode_image(file)
        message = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ]
            }
        ]
        response = call_openai(model, message)
        return response["content"], response.get("error", "")
    elif file_type == "video":
        base64Frames = video_to_base64(file)
        messages = [
        {
            "role": "user",
            "content": [
                query,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
                ],
            },
        ]
        response = call_openai(model, messages)
        return response["content"], response.get("error", "")
    elif file_type == "audio":
        converter = MarkdownConverter()
        content = converter.convert_local(file).text_content
        user_prompt = f"""请根据以下音频内容回答问题：
{content}
问题：{query}
        """
        messages = [
        {
            "role": "user",
            "content": user_prompt
            },
        ]
        response = call_openai(model, messages)
        return response["content"], response.get("error", "")
    else:
        raise ValueError(f"File type {file_type} is not supported.")