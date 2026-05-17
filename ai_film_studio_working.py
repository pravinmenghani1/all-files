#!/usr/bin/env python
"""
AI Film Studio - Complete Short Film Generator
"""

import os
import json
import re
import requests
from io import BytesIO
from PIL import Image
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI
try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
except ImportError:
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ImageGeneratorTool(BaseTool):
    name: str = "Generate Scene Image"
    description: str = "Generate cinematic images for film scenes."

    def _run(self, prompt: str) -> str:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Cinematic film still: {prompt}. Professional cinematography, dramatic lighting, 4K.",
                size="1792x1024",
                quality="hd",
                n=1,
            )
            return f"✅ Scene generated!\nURL: {response.data[0].url}"
        except Exception as e:
            return f"Error: {str(e)}"

class VoiceGeneratorTool(BaseTool):
    name: str = "Generate Voiceover"
    description: str = "Generate narration audio using TTS."

    def _run(self, text: str, filename: str = "narration") -> str:
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="onyx",
                input=text,
                speed=0.95
            )
            audio_path = f"{filename}.mp3"
            response.stream_to_file(audio_path)
            return f"✅ Voiceover: {audio_path}"
        except Exception as e:
            return f"Error: {str(e)}"

class SearchTool(BaseTool):
    name: str = "Search the internet"
    description: str = "Research film techniques and storytelling."

    def _run(self, query: str) -> str:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.environ.get('SERPER_API_KEY'),
            'content-type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        if 'organic' not in response.json():
            return "No results."
        results = response.json()['organic'][:3]
        return '\n'.join([f"{r['title']}: {r['snippet']}" for r in results if 'title' in r])

class FilmCrewAgents:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.9)

    def film_director(self):
        return Agent(
            role="Film Director",
            backstory="Visionary director like Christopher Nolan. Master of visual storytelling.",
            goal="Create compelling short film with powerful narrative arc.",
            tools=[SearchTool()],
            verbose=True,
            llm=self.llm
        )

    def cinematographer(self):
        return Agent(
            role="Director of Photography",
            backstory="Award-winning cinematographer. Expert in composition and lighting.",
            goal="Design stunning cinematic scenes.",
            tools=[ImageGeneratorTool(), SearchTool()],
            verbose=True,
            llm=self.llm
        )

    def composer(self):
        return Agent(
            role="Film Composer",
            backstory="Hans Zimmer-level composer. Creates emotionally powerful scores.",
            goal="Compose music that enhances storytelling.",
            tools=[SearchTool()],
            verbose=True,
            llm=self.llm
        )

    def voice_director(self):
        return Agent(
            role="Voice Director",
            backstory="Professional voice actor. Master of dramatic delivery.",
            goal="Write and deliver captivating narration.",
            tools=[VoiceGeneratorTool(), SearchTool()],
            verbose=True,
            llm=self.llm
        )

class FilmTasks:
    def write_screenplay(self, agent, concept, duration, genre):
        return Task(
            description=f"""Write short film screenplay:
                - Opening hook
                - 5-7 key scenes with timestamps
                - Character and conflict
                - Climax and resolution

                Concept: {concept}
                Duration: {duration}s
                Genre: {genre}""",
            expected_output="Screenplay with 5-7 scenes, timestamps, and emotional arc.",
            agent=agent
        )

    def shoot_scenes(self, agent):
        return Task(
            description="""Generate cinematic images:
                - Write DALL-E prompts for each scene from the screenplay
                - Use image generator tool for each scene
                - Include camera angles and lighting""",
            expected_output="5-7 scene images with URLs and cinematography notes.",
            agent=agent
        )

    def compose_score(self, agent):
        return Task(
            description="""Compose film score:
                - Main theme
                - Scene-by-scene music cues
                - Instrumentation and tempo
                - Emotional progression""",
            expected_output="Complete score with theme, cues, and instrumentation.",
            agent=agent
        )

    def create_narration(self, agent):
        return Task(
            description="""Create voiceover:
                - Write narration script (150-300 words)
                - Add timestamp markers
                - Use voice generator tool
                - Provide audio filename""",
            expected_output="Narration script with timestamps and audio file.",
            agent=agent
        )

class AIFilmStudio:
    def __init__(self, concept, duration, genre):
        self.concept = concept
        self.duration = duration
        self.genre = genre

    def run(self):
        agents = FilmCrewAgents()
        tasks = FilmTasks()

        director = agents.film_director()
        cinematographer = agents.cinematographer()
        composer = agents.composer()
        voice_director = agents.voice_director()

        screenplay_task = tasks.write_screenplay(director, self.concept, self.duration, self.genre)
        scenes_task = tasks.shoot_scenes(cinematographer)
        music_task = tasks.compose_score(composer)
        narration_task = tasks.create_narration(voice_director)

        film_crew = Crew(
            agents=[director, cinematographer, composer, voice_director],
            tasks=[screenplay_task, scenes_task, music_task, narration_task],
            verbose=True
        )

        return film_crew.kickoff()

def extract_image_urls(text):
    pattern = r'https://oaidalleapiprodscus\.blob\.core\.windows\.net/[^\s\)\]]+'
    return re.findall(pattern, text)

def create_film(image_urls, audio_file="narration.mp3", output="short_film.mp4"):
    print("\n🎬 Compiling your film...")
    
    clips = []
    for i, url in enumerate(image_urls):
        print(f"📥 Downloading scene {i+1}...")
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img_path = f"scene_{i+1}.jpg"
        img.save(img_path)
        
        clip = ImageClip(img_path).with_duration(3)
        clips.append(clip)
    
    video = concatenate_videoclips(clips, method="compose")
    
    if os.path.exists(audio_file):
        print("🎙️ Adding narration...")
        audio = AudioFileClip(audio_file)
        video = video.with_audio(audio)
    
    print("💾 Rendering final film...")
    video.write_videofile(output, fps=24, codec='libx264', audio_codec='aac')
    
    print(f"✅ Film saved as: {output}")
    return output

if __name__ == "__main__":
    print("🎥 Starting AI Film Studio...\n")
    
    studio = AIFilmStudio(
        concept="A robot's last day on Earth before humans return from space",
        duration=60,
        genre="Sci-Fi Drama"
    )
    
    print("🎬 Starting film production...\n")
    result = studio.run()
    
    print("\n🎉 FILM PRODUCTION COMPLETE!\n")
    
    print("🔍 Extracting scene URLs...")
    image_urls = []
    for task_output in result.tasks_output:
        urls = extract_image_urls(str(task_output.raw))
        image_urls.extend(urls)
    
    print(f"Found {len(image_urls)} scenes!")
    
    audio_files = [f for f in os.listdir('.') if f.endswith('.mp3')]
    audio_file = audio_files[0] if audio_files else "narration.mp3"
    
    if image_urls:
        film_path = create_film(image_urls, audio_file=audio_file)
        print(f"\n🎬 YOUR FILM IS READY: {film_path}\n")
    else:
        print("⚠️ No images found in output.")
