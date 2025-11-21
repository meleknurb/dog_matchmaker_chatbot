import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip # type: ignore
from urllib.parse import quote
import re

def recommend_dog_breeds(raw_user_input,scaled_dogs,numeric_traits,scaler,ohe_cols,top_n=3):
    # Prepare numeric input
    raw_numeric = pd.DataFrame(
        [[raw_user_input[t] for t in numeric_traits]],
        columns=numeric_traits
    )
    
    scaled_user_numeric = scaler.transform(raw_numeric)[0]

    user_vec = pd.Series(0, index=scaled_dogs.columns, dtype=float)
    user_vec[numeric_traits] = scaled_user_numeric

    # Coat Length (encoded)
    length_map = {'Short': 1, 'Medium': 2, 'Long': 3}
    user_vec['Coat_Length_Encoded'] = length_map.get(
        raw_user_input['Coat Length'],
        2  # default Medium
    )

    # Coat Type (one-hot)
    for col in ohe_cols:
        expected_col = f"Coat_Type_{raw_user_input['Coat Type']}"
        user_vec[col] = 1 if col == expected_col else 0

    # Cosine similarity
    similarities = cosine_similarity(
        user_vec.values.reshape(1, -1),
        scaled_dogs.values
    ).flatten()

    results = pd.DataFrame({
        "Breed": scaled_dogs.index,
        "Similarity": similarities
    }).sort_values("Similarity", ascending=False)

    return results.head(top_n)

def generate_breed_explanation(breed, top_traits, trait_df):

    explanation_parts = []
    for trait in top_traits:
        row = trait_df[trait_df["Trait"] == trait].iloc[0]
        description = row["Description"]
        explanation_parts.append(f"- **{trait}**: {description}")

    explanation_text = (
        f"\n🐶 **{breed}**:\n"
        + "\n".join(explanation_parts)
    )

    return explanation_text

def explain_top_breeds(ranked_breeds, dog_breeds, trait_df):
    results = []
    for breed, similarity in ranked_breeds[:3]:
        clean_breed_name = str(breed).replace('\xa0', ' ').strip()
        breed_traits = dog_breeds.loc[clean_breed_name]
        top_traits = breed_traits.sort_values(ascending=False).head(3).index.tolist()

        explanation = generate_breed_explanation(breed, top_traits, trait_df)

        results.append({
            "Breed": breed,
            "Explanation": explanation
        })

    return results

def fetch_breed_image(breed, mapping=None, image_name="Image_5.jpg"):
    base_url = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master"

    if breed in mapping.keys():
      folder = mapping[breed]
      folder_encoded = quote(folder)
    else:
      print(f"Breed '{breed}' not found in mapping!")
      return None

    image_url = f"{base_url}/{folder_encoded}/{image_name}"

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Image not found for: {breed}")
            return None

        img = Image.open(BytesIO(response.content))
        return img  

    except Exception as e:
        print(f"Error fetching image for {breed}: {e}")
        return None

def generate_breed_video(breed, mapping, max_images=10, size=(300, 300), sec_per_image=1):
    
    if breed not in mapping:
        print(f"⚠️ Breed '{breed}' not found in mapping!")
        return None, None

    folder = mapping[breed]

    repo_url = "https://api.github.com/repos/maartenvandenbroeck/Dog-Breeds-Dataset/contents"
    breed_url = f"{repo_url}/{folder}"

    resp = requests.get(breed_url)
    if resp.status_code != 200:
        print("⚠️ GitHub folder fetch failed!")
        return None, None

    files = resp.json()

    image_urls = [
        f["download_url"]
        for f in files
        if f["type"] == "file" and f["name"].lower().endswith((".jpg", ".png"))
    ]

    image_urls = image_urls[:max_images]

    if len(image_urls) == 0:
        print("⚠️ No images found for breed!")
        return None, None

    pil_images = []
    for url in image_urls:
        r = requests.get(url)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize(size)
        pil_images.append(img)

    numpy_images = [np.array(img) for img in pil_images]

    fps = 1 / sec_per_image

    clip = ImageSequenceClip(numpy_images, fps=fps)

    mp4_path = f"{folder}.mp4"
    clip.write_videofile(mp4_path, fps=fps,verbose=False,logger=None)

    return mp4_path

def detect_content_intent(user_text):
    text = user_text.lower()

    video_keywords = ['video', 'gif', 'animated', 'animation', 'loop', 'mp4']
    post_keywords = ['post', 'caption', 'instagram', 'content', 'story', 'reel']

    if any(k in text for k in video_keywords):
        return "video"

    if any(k in text for k in post_keywords):
        return "post"

    return None

def extract_breed_from_text(user_text, cleaned_breed_list):
    text = user_text.lower()
    sorted_breeds = sorted(cleaned_breed_list, key=len, reverse=True)

    for breed in sorted_breeds:
        if breed.lower() in text:
            return breed

    return None