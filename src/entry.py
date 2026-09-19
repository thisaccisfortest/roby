from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from groq import Groq
from workers import asgi


app = FastAPI()

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

chat_responses = []

chat_log = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "chat_responses": chat_responses
        }
    )


@app.post("/")
async def chat(
    request: Request,
    user_input: str = Form(...)
):
    chat_log.append({
        "role": "user",
        "content": user_input
    })

    chat_responses.append(user_input)

    client = Groq(
        api_key=request.scope["env"].GROQ_API_KEY
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=chat_log,
        temperature=0.6
    )

    bot_response = response.choices[0].message.content

    chat_log.append({
        "role": "assistant",
        "content": bot_response
    })

    chat_responses.append(bot_response)

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "chat_responses": chat_responses
        }
    )


@app.get("/image")
async def image_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="image.html",
        context={
            "image_url": None
        }
    )


@app.post("/image")
async def create_image(
    request: Request,
    user_input: str = Form(...)
):
    pollinations_api_key = request.scope["env"].POLLINATIONS_API_KEY

    url = "https://gen.pollinations.ai/v1/images/generations"

    data = {
        "prompt": user_input,
        "model": "black-forest-labs/flux.1-schnell",
        "size": "1024x1024",
        "response_format": "url"
    }

    headers = {
        "Authorization": f"Bearer {pollinations_api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json=data,
            timeout=120.0
        )

    response.raise_for_status()

    result = response.json()
    image_url = result["data"][0]["url"]

    return templates.TemplateResponse(
        request=request,
        name="image.html",
        context={
            "image_url": image_url
        }
    )


Default = asgi.entrypoint(app)