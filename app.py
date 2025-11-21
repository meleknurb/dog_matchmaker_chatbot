import streamlit as st
import os
import re
import json
import google.generativeai as genai # type: ignore
from PIL import Image

from logics import (
    detect_content_intent,
    extract_breed_from_text,
    explain_top_breeds,
    recommend_dog_breeds,
    fetch_breed_image,
    generate_breed_video
)
from data_loader import load_breed_data, load_trait_descriptions
from utils import (
    process_breed_data,
    list_github_folders,
    get_cleaned_breed_list,
    create_breed_github_mapping,
    system_prompt
)

st.set_page_config(
    page_title="PAWS Chatbot",
    page_icon="🐾",
    layout="centered"
)

try:
    api_key = st.secrets["GENAI_API_KEY"]
except KeyError:
    st.error("GENAI_API_KEY not found in Streamlit Secrets!")
    st.stop()

@st.cache_resource
def load_data_once():
    d_breeds = load_breed_data()    
    t_descriptions = load_trait_descriptions()
    sclr, s_dogs, ohe, num_traits = process_breed_data(d_breeds)
    fldrs = list_github_folders()
    cleaned = get_cleaned_breed_list(d_breeds)
    mpng = create_breed_github_mapping(cleaned, fldrs)
    
    return d_breeds, t_descriptions, sclr, s_dogs, ohe, num_traits, cleaned, mpng

dog_breeds, trait_descriptions, scaler, scaled_dogs, ohe_cols, numeric_traits, cleaned_breed_list, mapping = load_data_once()


if "chat_session" not in st.session_state:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    st.session_state.chat_session = model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])

if "messages" not in st.session_state:
    st.session_state.messages = []

if "top3_shown" not in st.session_state:
    st.session_state.top3_shown = False

st.title("🐾 PAWS Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        
        if message.get("content"):
            st.markdown(message["content"])
        
        if message.get("recommendations"):
            for rec in message["recommendations"]:
                st.markdown(f"### 🐶 {rec['breed_name']}")
                st.markdown(rec['description'])
                if rec['image']:
                    st.image(rec['image'], caption=rec['breed_name'], use_column_width=True)
                else:
                    pass 
        
        if message.get("video"):
            st.video(message["video"])

if prompt := st.chat_input("Type your message here..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt, "recommendations": None, "video": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            intent = detect_content_intent(prompt)
            
            final_text_content = ""
            final_recommendations = []
            final_video = None
            
            if st.session_state.top3_shown and intent in ["post", "video"]:
                breed = extract_breed_from_text(prompt, cleaned_breed_list)
                
                if not breed:
                    resp = st.session_state.chat_session.send_message(prompt)
                    final_text_content = resp.text
                else:
                    if intent == "post":
                        post_prompt = f"Generate a short, playful social media caption for {breed}. Theme: {prompt}. Max 2 sentences."
                        post_response = st.session_state.chat_session.send_message(post_prompt)
                        final_text_content = f"**PAWS (Social Media Post):**\n\n{post_response.text.strip()}"
                        
                        img = fetch_breed_image(breed, mapping=mapping)
                        if img:
                            final_recommendations.append({
                                "breed_name": breed,
                                "description": "",
                                "image": img
                            })

                    elif intent == "video":
                        video_prompt = f"Caption for looping video of {breed}. Theme: {prompt}."
                        video_caption = st.session_state.chat_session.send_message(video_prompt)
                        final_text_content = f"**PAWS (Video Caption):**\n\n{video_caption.text.strip()}"
                        
                        mp4_path = generate_breed_video(breed, mapping)
                        if mp4_path:
                            final_video = mp4_path

            else:
                try:
                    response = st.session_state.chat_session.send_message(prompt)
                    full_response_text = response.text

                    
                    json_match = re.search(r'```json\n({.*?})\n```', full_response_text, re.DOTALL)
                    json_pattern = r'```json\n{.*?}\n```'
                    
                    cleaned_text = re.sub(json_pattern, '', full_response_text, flags=re.DOTALL).strip()
                    final_text_content = cleaned_text

                    if json_match:
                        parsed = json.loads(json_match.group(1))
                        
                        if 'Coat Length' in parsed and 'Coat Type' in parsed:
                            ranked_df = recommend_dog_breeds(parsed, scaled_dogs, numeric_traits, scaler, ohe_cols)
                            
                            ranked_list_for_explanation = []
                            for idx, row in ranked_df.iterrows():
                                raw_name = row['Breed']
                                clean_name = str(raw_name).replace('\xa0', ' ').strip()
                                ranked_list_for_explanation.append((clean_name, row['Similarity']))

                            final_results_data = explain_top_breeds(ranked_list_for_explanation, dog_breeds, trait_descriptions)
                            if not final_text_content:
                                final_text_content = "Great news! Here are our top 3 dog breed recommendations, handpicked just for you: 🐾\n\n"

                            for r in final_results_data:
                                raw_name = r['Breed']
                                b_name = str(raw_name).replace('\xa0', ' ').strip()

                                img = fetch_breed_image(b_name, mapping=mapping)
                                
                                final_recommendations.append({
                                    "breed_name": b_name,
                                    "description": r['Explanation'],
                                    "image": img
                                })
                            
                            st.session_state.top3_shown = True
                            
                except Exception as e:
                    print(f"DEBUG ERROR: {e}")
                    if not final_text_content:
                        final_text_content = "I'm thinking..? 🐾"
            
            if final_text_content:
                st.markdown(final_text_content)
            
            for rec in final_recommendations:
                st.markdown(f"### 🐶 {rec['breed_name']}")
                st.markdown(rec['description'])
                st.image(rec['image'], caption=rec['breed_name'], use_column_width=True)

            if final_video:
                st.video(final_video)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_text_content,
        "recommendations": final_recommendations, 
        "video": final_video
    })