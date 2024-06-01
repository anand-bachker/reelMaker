import uuid
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import re

import os
import shutil
import subprocess
import json

from dotenv import load_dotenv

load_dotenv()


import google.generativeai as genai
from openai import OpenAI


GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

OPENAI_KEY = os.environ["OPENAI_KEY"]
client = OpenAI(api_key=OPENAI_KEY)

base_prompt = "Extract a list of all {topic_prompt} and the corresponding start and end timestamps where they are discussed in the following YouTube transcript. Ensure duration is within 20 to 80 seconds with recommended length of 30 sec,it is IMPORTANT video length should not be less than 20 sec and should be bound by limits. Format the output as a JSON array:"

output_format = """

[
    {
      "title": "string",
      "info": "string",
      "why_clip_was_chosen": "string",
      "url":string,
      "start_time": int,
      "end_time": int,
    },
    ...
]

"""

additional_base_prompt = """
if nothing can be extracted return an empty array i.e. []

{url_source_prompt}

dont return anything else as output it should always be an array

Transcript:

"""


def get_message(transcript, topic_prompt="Books", url_source=""):
  url_source_prompt = ""
  if url_source:
    url_source_prompt = f"\nFor the URL, use the {url_source} link relevant to the discussed topic."
  else:
    url_source_prompt = "keep the 'url' filed empty don't put any value in that key"

  base_prompt_with_topic = base_prompt.format(topic_prompt=topic_prompt)
  additional_base_prompt_with_url_source_prompt = additional_base_prompt.format(url_source_prompt=url_source_prompt)

  transcript_text = "\n".join([f"{item['start']} - {item['text']}" for item in transcript])
  result = base_prompt_with_topic + output_format + additional_base_prompt_with_url_source_prompt + transcript_text
  return result


def get_video_id(url):
    # Extract video id from YouTube URL
    video_id = re.search(r'(?<=v=)[^&#]+', url)
    if video_id is None:
        video_id = re.search(r'(?<=be/)[^&#]+', url)
    if video_id:
        return video_id.group(0)
    else:
        raise ValueError("Invalid YouTube URL")

def get_raw_transcript(url):
    video_id = get_video_id(url)
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript_list

def get_llm_output(messageText, service = "gemini", model = ""):
  if service == "gemini":
      model = genai.GenerativeModel(model)
      output = model.generate_content(messageText)
      return output.text
  elif service == "openai":
      chat_completion = client.chat.completions.create(
        messages=[
          {
            "role": "user",
            "content": messageText,
          }
        ],
        model=model,
      )
      return chat_completion.choices[0].message.content
  else:
    return "[]"


def download_and_trim(url, base_directory, content_list):
    try:
        # If the directory already exists, remove it first
        if os.path.exists(base_directory):
            shutil.rmtree(base_directory)

        yt = YouTube(url)
        video = yt.streams.first().download(output_path=base_directory)
        print(f"Downloaded video: {video}")

        # Ensure the clips directory exists
        clips_directory = os.path.join(base_directory, "clips")
        os.makedirs(clips_directory, exist_ok=True)

        for content in content_list:
            start_time = content["start_time"]
            end_time = content["end_time"]
            title = content["title"]
            output_file = os.path.join(clips_directory, f"{title}.mp4")

            # Construct the ffmpeg command
            ffmpeg_command = [
                "ffmpeg",
                "-i", video,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c", "copy",
                output_file,
                "-loglevel", "error"
            ]
            # Run the ffmpeg command
            subprocess.run(ffmpeg_command, check=True)

    except Exception as e:
        print(f"An error occurred: {e}")


def get_clips(config):
  name = config.get("name", uuid.uuid4().hex)
  url = config.get("url", "")
  topic_prompt = config.get("topic_prompt", "Books")
  service = config.get("service", "gemini")
  model = config.get("model", "gemini-pro")
  url_source = config.get("url_source", "")


  pwd = os.getcwd()
  base_directory = os.path.join(pwd, "output", name)
  os.makedirs(base_directory, exist_ok=True)
  
  path_for_output = os.path.join(base_directory, "llmOutput.json")
  if (os.path.exists(path_for_output)):
    print("Output already exists, skipping...")
    return
  

  transcript = get_raw_transcript(url)
  message = get_message(transcript, topic_prompt, url_source)
  llmOutput = get_llm_output(message ,service,model)
  llmOutput = llmOutput.replace("```", "")
  llmOutput = llmOutput.replace("json", "")
  result = json.loads(llmOutput)
  for item in result:
     item["title"] = item["title"].replace(" ", "_").replace("/", "_")
  output = {
    "message": message,
    "model": model,
    "service": service,
    "llmOutput": llmOutput,
    "clips": result
  }
  download_and_trim(
      url = url,
      base_directory = base_directory,
      content_list = result
  )
  try:
      with open(path_for_output, "w") as f:
          json.dump(output, f, indent=4)
      print(f"Output saved to: {path_for_output}")
  except Exception as e:
      print(f"An error occurred while saving the file: {e}")
