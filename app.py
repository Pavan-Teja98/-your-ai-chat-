from flask import Flask, request, render_template, jsonify
import os
import google.generativeai as genai
from werkzeug.utils import secure_filename
import PIL.Image
import requests
import json
import base64
from io import BytesIO
import time
import re

app = Flask(__name__)

# Hardcoded API Keys (not recommended for production)
GOOGLE_API_KEY = "AIzaSyB-lqw-1VS27qMxxWPEIwRVvIsanCY2rWA"  # Replace with your actual Google API key
HUGGINGFACE_API_KEY = "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"  # Replace with your actual Hugging Face API key

# Configure Google API Key
try:
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key not set.")

    genai.configure(api_key=GOOGLE_API_KEY)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config,)
    chat_session = model.start_chat(history=[])

except ValueError as e:
    print(f"Configuration error: {e}")
    exit()

except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Configure Hugging Face API for image generation
IMAGE_GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # Default model, can be changed
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{IMAGE_GEN_MODEL}"

# Set up image directories
UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "generated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GENERATED_FOLDER"] = GENERATED_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_image(prompt):
    """Generate an image using Hugging Face API."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            # Save the image
            image_data = response.content
            timestamp = int(time.time())  # Correct way to get a unique timestamp
            filename = f"generated_{timestamp}.png"
            filepath = os.path.join(app.config["GENERATED_FOLDER"], filename)
            
            with open(filepath, "wb") as f:
                f.write(image_data)
            
            # Also convert to base64 for immediate display
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return {
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "image_data": f"data:image/png;base64,{image_base64}"
            }
        else:
            # Handle error cases
            error_info = response.json() if response.headers.get('content-type') == 'application/json' else {"error": "Unknown error"}
            return {
                "success": False,
                "error": f"API Error: {error_info.get('error', 'Unknown error')}"
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {
            "success": False, 
            "error": f"Exception: {str(e)}"
        }

@app.route("/", methods=["GET"])
def index():
    """Render the main HTML page."""
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Handles text-based chat with Gemini AI."""
    user_input = request.json.get("message")
    try:
        response = chat_session.send_message(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handles image uploads and generates a description using Gemini AI."""
    if "image" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Load image with PIL and send to Gemini
            image = PIL.Image.open(filepath)
            
            # Create a new prompt with the image
            response = model.generate_content(["Describe this image in detail:", image])
            description = response.text

            return jsonify({"filename": filename, "description": description})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Invalid file type"})

@app.route("/api/generate-image", methods=["POST"])
def api_generate_image():
    """Handles image generation requests using Hugging Face API."""
    prompt = request.json.get("prompt")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"})
    
    # Optional: Use Gemini to enhance the prompt
    try:
        enhancement_prompt = f"Enhance this image generation prompt to create a more detailed and visually appealing result: '{prompt}'"
        enhanced_response = chat_session.send_message(enhancement_prompt)
        enhanced_prompt = enhanced_response.text
        
        # Use regex to strip only unwanted formatting but preserve meaning
        # Remove only formatting patterns like "Enhanced prompt:" but keep the content
        enhanced_prompt = re.sub(r'^(Enhanced prompt:|Here\'s an enhanced version:|Enhanced:)\s*', '', enhanced_prompt, flags=re.IGNORECASE)
        
        # If the enhanced prompt is significantly different and not just formatting changes
        if len(enhanced_prompt) > len(prompt) * 1.2:  # 20% longer at minimum
            prompt = enhanced_prompt
    except:
        # If enhancement fails, just use the original prompt
        pass
    
    result = generate_image(prompt)
    
    if result["success"]:
        return jsonify({
            "success": True,
            "filename": result["filename"],
            "image_data": result["image_data"]
        })
    else:
        return jsonify({
            "success": False,
            "error": result["error"]
        })

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Changed port to 5001