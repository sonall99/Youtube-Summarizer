import os
import google.generativeai as genai
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

try:
    api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = genai.GenerativeModel("gemini-2.5-pro")

def get_video_id_from_url(video_url: str) -> str | None:
    try:
        query = urlparse(video_url).query
        params = parse_qs(query)
        if 'v' in params:
            return params['v'][0]
        if 'youtu.be/' in video_url:
            return video_url.split('youtu.be/')[-1].split('?')[0]
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return None
    return None

def get_transcript(video_id: str) -> str:
    try:
        yt = YouTubeTranscriptApi()
        transcripts = yt.list(video_id)
        
        try:
            transcript_obj = transcripts.find_transcript(['en-IN'])
        except Exception:
            print("No 'en-IN' transcript found. Falling back to 'hi'.")
            try:
                transcript_obj = transcripts.find_transcript(['hi'])
            except Exception:
                print("No 'hi' transcript found. Falling back to 'en'.")
                transcript_obj = transcripts.find_transcript(['en'])

        transcript_data = transcript_obj.fetch()
        full_text = " ".join([item.text for item in transcript_data])
        
        return full_text

    except Exception as e:
        print(f"Error getting transcript: {e}")
        raise ValueError("Could not fetch transcript. The video may not have one, or it's disabled.")

def summarize_with_gemini(transcript: str) -> str:
    prompt = f"""
    You are an expert video summarizer.
    Summarize the following video transcript into 3-5 concise bullet points, 
    highlighting the main ideas and conclusions.

    Transcript:
    "{transcript}"

    Summary:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        raise ValueError("Error from summarization model.")

class VideoRequest(BaseModel):
    video_url: str

class SummaryResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryResponse)
async def handle_summarize_request(request: VideoRequest):
    print(f"Received request for URL: {request.video_url}")
    try:
        video_id = get_video_id_from_url(request.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL. Could not find video ID.")

        print(f"Fetching transcript for video ID: {video_id}")
        transcript = get_transcript(video_id)

        print("Sending transcript to Gemini...")
        summary = summarize_with_gemini(transcript)
        
        print("Returning summary to frontend.")
        return SummaryResponse(summary=summary)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


print("AVAILABLE MODELS:")
for m in genai.list_models():
    print(m.name)
