import os
import json
import uuid
import ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from faster_whisper import WhisperModel
from moviepy.editor import (
    TextClip,
    CompositeVideoClip,
    VideoFileClip,
    ColorClip,
    ImageClip,
)

# Create an image with rounded corners using PIL

with open("fontConfig.json") as f:
    font_config_list = json.load(f)


def create_rounded_image(size, radius, color, opacity):
    width, height = size
    # Parse the color and apply the opacity
    r, g, b = color
    color_with_opacity = (r, g, b, int(255 * opacity))  # Opacity is scaled to 0-255

    # Create an RGBA image to accommodate opacity
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [0, 0, width, height], radius=radius, fill=color_with_opacity
    )
    return image


model_size = "medium"
model = WhisperModel(model_size)


def extract_audio_from_video(video_path, audio_path):
    try:
        input_stream = ffmpeg.input(video_path)
        audio = input_stream.audio
        output_stream = ffmpeg.output(audio, audio_path, loglevel="quiet")
        output_stream = ffmpeg.overwrite_output(output_stream)
        ffmpeg.run(output_stream)
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_text_from_audio(audio_path):
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    segments = list(segments)
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append(
                {"word": word.word, "start": float(word.start), "end": float(word.end)}
            )
    return wordlevel_info


def split_text_into_lines(data, subtitle_config):
    max_chars = subtitle_config.get("max_chars", 30)
    max_duration = subtitle_config.get("max_duration", 2.5)
    max_gap = subtitle_config.get("max_gap", 1.5)

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0

    for idx, word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)

        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > max_duration
        chars_exceeded = new_line_chars > max_chars
        if idx > 0:
            gap = word_data["start"] - data[idx - 1]["end"]
            #   print (word,start,end,gap)
            maxgap_exceeded = gap > max_gap
        else:
            maxgap_exceeded = False

        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line,
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0

    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line,
        }
        subtitles.append(subtitle_line)

    return subtitles


def create_caption(text_json, frame_size, config):
    font_config_id = config.get("font_config_id", "1")

    font_config = font_config_list[font_config_id]

    wordcount = len(text_json["textcontents"])
    full_duration = text_json["end"] - text_json["start"]

    word_clips = []
    xy_textclips_positions = []

    x_pos = 0
    y_pos = 0
    line_width = 0  # Total width of words in the current line
    frame_width = frame_size[0]
    frame_height = frame_size[1]

    x_buffer = frame_width * 1 / 10

    max_line_width = frame_width - 2 * (x_buffer)

    fontsize = int(frame_height * font_config["font_size_factor"])

    space_width = ""
    space_height = ""

    for index, word_json in enumerate(text_json["textcontents"]):
        duration = word_json["end"] - word_json["start"]
        normal_word_duration = full_duration
        normal_word_start = text_json["start"]
        if font_config["words_on_the_go"]:
            normal_word_duration = text_json["end"] - word_json["start"]
            normal_word_start = word_json["start"]

        word_clip = (
            TextClip(
                word_json["word"],
                font=font_config["normal"]["font"],
                fontsize=fontsize * font_config["normal"]["font_size_factor"],
                color=font_config["normal"]["color"],
                stroke_color=font_config["normal"]["stroke_color"],
                stroke_width=font_config["normal"]["stroke_width"],
            )
            .set_start(normal_word_start)
            .set_duration(normal_word_duration)
        )
        word_clip_space = (
            TextClip(
                font_config["spacing"]["text"],
                font=font_config["spacing"]["text"],
                fontsize=fontsize * font_config["spacing"]["font_size_factor"],
                color=font_config["spacing"]["color"],
            )
            .set_start(normal_word_start)
            .set_duration(normal_word_duration)
        )
        word_width, word_height = word_clip.size
        space_width, space_height = word_clip_space.size
        if line_width + word_width + space_width <= max_line_width:
            # Store info of each word_clip created
            xy_textclips_positions.append(
                {
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "width": word_width,
                    "height": word_height,
                    "word": word_json["word"],
                    "start": word_json["start"],
                    "end": word_json["end"],
                    "duration": duration,
                }
            )

            word_clip = word_clip.set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_position((x_pos + word_width, y_pos))

            x_pos = x_pos + word_width + space_width
            line_width = line_width + word_width + space_width
        else:
            # Move to the next line
            x_pos = 0
            y_pos = y_pos + word_height + 10
            line_width = word_width + space_width

            # Store info of each word_clip created
            xy_textclips_positions.append(
                {
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "width": word_width,
                    "height": word_height,
                    "word": word_json["word"],
                    "start": word_json["start"],
                    "end": word_json["end"],
                    "duration": duration,
                }
            )

            word_clip = word_clip.set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_position((x_pos + word_width, y_pos))
            x_pos = word_width + space_width

        word_clips.append(word_clip)
        word_clips.append(word_clip_space)

    for highlight_word in xy_textclips_positions:
        # Define the size, corner radius, color, and opacity from the configuration
        size = (int(highlight_word["width"] * 1.1 * font_config["highlighted"]["font_size_factor"]), int(highlight_word["height"] * 1.1 * font_config["highlighted"]["font_size_factor"]))
        radius = font_config["highlighted"]["back_ground_color_clip"]["radius"]
        color = font_config["highlighted"]["back_ground_color_clip"]["color"]  # Expecting an (R, G, B) tuple
        opacity = font_config["highlighted"]["back_ground_color_clip"]["opacity"]
        # Create the rounded background clip
        rounded_image = create_rounded_image(size, radius, color, opacity)
        background_clip = ImageClip(np.array(rounded_image))

        # Create a text clip
        text_clip = TextClip(
            highlight_word["word"],
            font=font_config["highlighted"]["font"],
            fontsize=fontsize * font_config["highlighted"]["font_size_factor"],
            color=font_config["highlighted"]["color"],
            stroke_color=font_config["highlighted"]["stroke_color"],
            stroke_width=font_config["highlighted"]["stroke_width"],
        )

        # Calculate the position to center the text in the background clip
        text_position = (
            (background_clip.size[0] - text_clip.size[0]) / 2,
            (background_clip.size[1] - text_clip.size[1]) / 2,
        )

        # Set the position of the text clip
        text_clip = text_clip.set_position(text_position)

        # Create a composite video clip
        composite_clip = CompositeVideoClip([background_clip, text_clip])

        # Set position of the entire composition based on x and y
        final_clip = composite_clip.set_position((highlight_word["x_pos"], highlight_word["y_pos"]))
        final_clip = final_clip.set_start(highlight_word["start"]).set_duration(highlight_word["duration"])
        
        # add random rotation
        final_clip = final_clip.rotate(np.random.uniform(-font_config["highlighted"]["rotate_random_degree"], font_config["highlighted"]["rotate_random_degree"]), resample="bicubic")

        # Append the composite clip to the list
        word_clips.append(final_clip)

    return word_clips, xy_textclips_positions


def add_subtitles(video_path, linelevel_subtitles, config):
    name = config.get("name", uuid.uuid4().hex)
    output_resolution = config.get("output_resolution", "1080p")
    video_aspect_ratio = config.get("video_aspect_ratio", [9, 16])
    font_configId = config.get("font_config_id", "1")
    font_config = font_config_list[font_configId]

    video_name = os.path.basename(video_path)
    base_path = os.path.join(os.getcwd(), "output", name, "final")
    os.makedirs(base_path, exist_ok=True)
    output_path = os.path.join(base_path, video_name)

    input_video = VideoFileClip(video_path)
    frame_size = input_video.size
    scale_factor = int(output_resolution.split("p")[0]) / frame_size[1]
    input_video = input_video.resize(scale_factor)
    frame_size = input_video.size

    new_width = int(frame_size[1] * video_aspect_ratio[0] / video_aspect_ratio[1])
    if new_width % 2 != 0:
        new_width -= 1

    x_offset = (frame_size[0] - new_width) // 2
    if x_offset % 2 != 0:
        x_offset -= 1

    input_video = input_video.crop(
        x1=x_offset,
        y1=0,
        width=new_width,
        height=frame_size[1],
    )

    frame_size = input_video.size

    all_linelevel_splits = []

    for line in linelevel_subtitles:
        out_clips, positions = create_caption(line, frame_size, config)

        max_width = 0
        max_height = 0

        for position in positions:
            # break
            x_pos, y_pos = position["x_pos"], position["y_pos"]
            width, height = position["width"], position["height"]

            max_width = max(max_width, x_pos + width)
            max_height = max(max_height, y_pos + height)

        
        color_clip = ColorClip(
            size=(int(max_width * 1.1 * font_config["background"]["size_factor"]), int(max_height * 1.1 * font_config["background"]["size_factor"])),
            color=font_config["background"]["color"],
        )
        color_clip = color_clip.set_opacity(font_config["background"]["opacity"])
        color_clip = color_clip.set_start(line["start"]).set_duration(line["end"] - line["start"])

        # centered_clips = [each.set_position('center') for each in out_clips]

        clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)
        clip_to_overlay = clip_to_overlay.set_position("bottom")
        clip_to_overlay = clip_to_overlay.set_position(
            (
                "center",
                frame_size[1]
                - max_height
                - font_config["bottom_offset_factor"] * frame_size[1],
            )
        )

        all_linelevel_splits.append(clip_to_overlay)

    final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

    # Set the audio of the final video to be the same as the input video
    final_video = final_video.set_audio(input_video.audio)

    # Save the final clip as a video file with the audio included
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")


def add_subtitles_to_video(video_name, config):
    name = config.get("name")
    subtitle_config = config.get("subtitle_config")

    base_path = os.path.join(os.getcwd(), "output", name)
    video_path = os.path.join(base_path, "clips", video_name)
    audio_path = os.path.join(base_path, "audio", video_name.replace(".mp4", ".mp3"))
    subtitles_path = os.path.join(
        base_path, "subtitles", video_name.replace(".mp4", ".json")
    )

    if os.path.exists(subtitles_path):
        print(f"Subtitles already exist for {video_name}")
        with open(subtitles_path, "r") as f:
            linelevel_subtitles = json.load(f)
        add_subtitles(video_path, linelevel_subtitles, config)
        return

    extract_audio_from_video(video_path, audio_path)
    print("Extracted audio from video")
    wordlevel_info = extract_text_from_audio(audio_path)
    print("Extracted text from audio")
    linelevel_subtitles = split_text_into_lines(wordlevel_info, subtitle_config)
    with open(subtitles_path, "w") as f:
        json.dump(linelevel_subtitles, f)
    print("Split text into lines")
    add_subtitles(video_path, linelevel_subtitles, config)
    print("Added subtitles to video")


def add_subtitles_to_clips(config):
    name = config.get("name")
    cwd = os.getcwd()
    base_path_clips = os.path.join(cwd, "output", name, "clips")
    base_path_final = os.path.join(cwd, "output", name, "final")
    base_path_audio = os.path.join(cwd, "output", name, "audio")
    base_path_subtitles = os.path.join(cwd, "output", name, "subtitles")

    os.makedirs(base_path_final, exist_ok=True)
    os.makedirs(base_path_audio, exist_ok=True)
    os.makedirs(base_path_clips, exist_ok=True)
    os.makedirs(base_path_subtitles, exist_ok=True)

    clips = os.listdir(base_path_clips)
    final_clips = os.listdir(base_path_final)

    clips_to_process = list(set(clips) - set(final_clips))
    print(f"Clips to process: {clips_to_process}")
    for clip in clips_to_process:
        add_subtitles_to_video(clip, config)
