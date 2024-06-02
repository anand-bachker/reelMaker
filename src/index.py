from helper_download import get_clips
from helper_subtitle import add_subtitles_to_clips
import json

with open("./config.json") as f:
    configs = json.load(f)

config = configs["2"]

print(config)

get_clips(config)

add_subtitles_to_clips(config)



