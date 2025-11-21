# utils.py
import requests
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

def process_breed_data(dog_breeds):
    # Set index
    dog_breeds = dog_breeds.set_index('Breed')

    # Encode coat length
    length_map = {'Short': 1, 'Medium': 2, 'Long': 3}
    dog_breeds['Coat_Length_Encoded'] = dog_breeds['Coat Length'].map(length_map)
    dog_breeds.drop('Coat Length', axis=1, inplace=True)

    # One-hot encode coat type
    dog_breeds = pd.get_dummies(dog_breeds, columns=['Coat Type'], prefix='Coat_Type', drop_first=True)
    ohe_cols = [c for c in dog_breeds.columns if c.startswith("Coat_Type_")]

    # Scale numeric traits
    numeric_traits = dog_breeds.columns[:14]
    scaler = StandardScaler()

    scaled_dogs = dog_breeds.copy()
    scaled_dogs[numeric_traits] = scaler.fit_transform(scaled_dogs[numeric_traits])

    return scaler, scaled_dogs, ohe_cols, numeric_traits

def list_github_folders():
    repo_url = "https://api.github.com/repos/maartenvandenbroeck/Dog-Breeds-Dataset/contents"

    response = requests.get(repo_url)

    if response.status_code == 200:
        data = response.json()
        folders = [item['name'] for item in data if item['type'] == 'dir']
        return folders
    else:
        print("GitHub API request failed! Status:", response.status_code)
        return []


def get_cleaned_breed_list(dog_breeds):
    pd.set_option('display.max_rows', None)
    breed_list = dog_breeds.index.unique().tolist()
    cleaned_breed_list = [str(breed).replace('\xa0', ' ') for breed in breed_list]
    return cleaned_breed_list

def normalize_for_matching(name):
    name = name.lower()
    name = re.sub(r"[^\w\s()]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    paren = re.findall(r"\((.*?)\)", name)
    name = re.sub(r"\(.*?\)", "", name).strip()
    if paren:
        name = ' '.join(paren) + ' ' + name
    words = name.split()
    if words[-1].endswith('s') and len(words[-1]) > 3:
        words[-1] = words[-1][:-1]
    if 'dog' not in words:
        words.append('dog')
    return ' '.join(words)

manual_mapping = {
    'Pointers (German Shorthaired)': 'german short haired pointing dog',
    'Siberian Huskies': 'siberian husky dog',
    'Doberman Pinschers': 'dobermann dog',
    'Pomeranians': 'pomeranian dog',
    'Cane Corso': 'italian cane corso dog',
    'Brittanys': 'brittany spaniel dog',
    'Spaniels (Cocker)': 'cocker spaniel dog',
    'Vizslas': 'hungarian short haired pointer (vizsla) dog',
    'Belgian Malinois': 'belgian shepherd dog',
    'Collies': 'collie rough dog',
    'Shiba Inu': 'shiba dog',
    'Bichons Frises': 'bichon frise dog',
    'Papillons': 'papillon dog',
    'Soft Coated Wheaten Terriers': 'irish soft coated wheaten terrier dog',
    'Pointers (German Wirehaired)': 'german wire haired pointing dog',
    'Chinese Shar-Pei': 'shar pei dog',
    'Wirehaired Pointing Griffons': 'wire-haired pointing griffon korthals dog',
    'Italian Greyhounds': 'italian sighthound dog',
    'Great Pyrenees': 'pyrenean mountain dog',
    'Dogues de Bordeaux': 'dogue de bordeaux',
    'Russell Terriers': 'jack russell terrier dog',
    'Setters (Irish)': 'irish red setter dog',
    'Greater Swiss Mountain Dogs': 'great swiss mountain dog',
    'Rat Terriers': 'rat terrier dog',
    'Anatolian Shepherd Dogs': 'anatolian shepherd dog',
    'Spaniels (Boykin)': 'boykin spaniel dog',
    'Lagotti Romagnoli': 'lagotto romagnolo dog',
    'Brussels Griffons': 'griffon bruxellois dog',
    'Norwegian Elkhounds': 'norwegian elkhound grey dog',
    'Standard Schnauzers': 'schnauzer dog',
    'Bouviers des Flandres': 'bouvier des flandres dog',
    'Keeshonden': 'keeshonden dog',
    'Retrievers (Flat-Coated)': 'flat coated retriever dog',
    'Borzois': 'borzoi - russian hunting sighthound dog',
    'Belgian Tervuren': 'belgian tervuren dog',
    'Silky Terriers': 'australian silky terrier dog',
    'Spinoni Italiani': 'italian spinone dog',
    'Toy Fox Terriers': 'toy fox terrier dog',
    'Pointers': 'english pointer dog',
    'Belgian Sheepdogs': 'belgian shepherd dog',
    'American Eskimo Dogs': 'american eskimo dog',
    'Beaucerons': 'berger de beauce dog',
    'Boerboels': 'boerboel dog',
    'Black Russian Terriers': 'black russian terrier dog',
    'American Hairless Terriers': 'american hairless terrier dog',
    'Xoloitzcuintli': 'xoloitzcuintle dog',
    'Bluetick Coonhounds': 'bluetick coonhound dog',
    'English Toy Spaniels': 'english toy spaniel (black & tan) dog',
    'Pulik': 'puli dog',
    'Barbets': 'barbet dog',
    'Redbone Coonhounds': 'redbone coonhound dog',
    'Berger Picards': 'berger de picard dog',
    'Entlebucher Mountain Dogs': 'entlebuch cattle dog',
    'Treeing Walker Coonhounds': 'treeing walker coonhound dog',
    'Wirehaired Vizslas': 'hungarian wire-haired pointer dog',
    'Pumik': 'pumi dog',
    'Portuguese Podengo Pequenos': 'portuguese podengo dog',
    'Retrievers (Curly-Coated)': 'curly coated retriever dog',
    'Lowchen': 'lowchen dog',
    'Petits Bassets Griffons Vendeens': 'petit basset griffon vendeen dog',
    'Finnish Lapphunds': 'swedish lapphund dog',
    'Scottish Deerhounds': 'deerhound dog',
    'Plott Hounds': 'plott hound dog',
    'Glen of Imaal Terriers': 'irish glen of imaal terrier dog',
    'Ibizan Hounds': 'ibizan podenco dog',
    'Bergamasco Sheepdogs': 'bergamasco shepherd dog',
    'Kuvaszok': 'kuvasz dog',
    'Komondorok': 'komondor dog',
    'Cirnechi dell’Etna': "cirneco dell'etna dog",
    'Pyrenean Shepherds': 'pyrenean sheepdog - smooth faced',
    'American English Coonhounds': 'american english coonhound dog',
    'Chinooks': 'chinook dog'
}

def create_breed_github_mapping(cleaned_breed_list, folders, manual_mapping=manual_mapping):
    normalized_dataset = {b: normalize_for_matching(b) for b in cleaned_breed_list}
    normalized_github = {f: normalize_for_matching(f) for f in folders}

    mapping = {}
    unmatched = []

    for orig_name, norm_name in normalized_dataset.items():
        match = [g for g, g_norm in normalized_github.items() if norm_name == g_norm]
        if match:
            mapping[orig_name] = match[0]
        else:
            unmatched.append(orig_name)

    
    mapping.update(manual_mapping)
    
    return mapping


system_prompt = '''
(You are **PAWS**, an expert and friendly dog breed matching assistant with deep knowledge of canine behavior, lifestyle compatibility, training difficulty, coat care, health tendencies, and adoption suitability.
Your mission is to help users find the most compatible dog breeds based on their lifestyle, expectations, environment, and experience.
You must act warm, positive, supportive, and empathetic — never judgmental, blunt, or robotic.

- Example starter: Before we start, just a quick note: I’m here to help match dog breeds based on personality and lifestyle traits. I’m not a veterinarian or certified behaviorist, so consider my suggestions as friendly guidance 💛 Ready to begin?

====================
PERSONALITY & TONE
====================
• Warm, friendly, playful, encouraging, and emotionally supportive
• Uses light emojis (not excessive), such as 🐶🐾💛
• Uses simple and natural language
• Asks one question at a time (unless clarification required)
• Never rushes the user and always validates feelings/preferences
• Avoids technical language unless user requests

Example tone:
“Let’s find your perfect furry best friend together! 🐶💛”

====================
MODES & INTERACTION
====================
MEMORY MODE = ON
- Remember and use previous answers during the session
- Reference previous answers naturally (not mechanically)

CLARIFICATION MODE = ON
- If an answer is vague, conflicting, or missing, gently ask again
Example:
“Hmm, just to make sure I get this right… would you say more like medium or high energy?”

CONSENT MODE = ON
- Always begin with a friendly consent message

====================
INTERVIEW STYLE
====================
You have the following dog traits for breed matching:

1. Energy & Playfulness & Mental Stimulation Needs
2. Affection & Family Compatibility
3. Good With Young Children
4. Social with Strangers & Other Dogs
5. Adaptability Level
6. Trainability Level
7. Watchdog/Protective Nature
8. Barking Level
9. Shedding Level
10. Drooling Level
11. Coat Length & Coat Type
12. Brushing & Grooming


Instruction:

1. For each trait or logical group of traits, generate a short, natural, playful question for the user.
2. DO NOT reveal numeric values or scales.
3. Only show the trait title internally as a hint — the AI must generate the actual question.
5. Maintain a warm, supportive tone with light emojis (1-3 max).
6. Ask one question at a time unless clarification is needed.
7. Collect the answers to all traits needed for breed matching.
8. After collecting all answers, produce the JSON output internally using the required format for traits.

====================
EXPLANATION MODE
====================
• Once you have collected all user preferences and output the JSON, the system will provide you with a list of top recommended breeds and their *pre-generated explanations*.
• Your task will then be to present these pre-generated explanations to the user in your warm, friendly, and helpful PAWS tone.
• You should introduce the recommendations enthusiastically and then present each breed's explanation clearly.
• Do can modify the content of the provided explanations based on user's preferences.
• Do NOT add any scoring, numeric logic, or similarity scores.
• Use light and suitable emojis (1-3 max) to enhance the presentation.
• After the initial top 3 breed recommendation presentation, PAWS MUST offer optional post-interaction services like SOCIAL MEDIA POST or VIDEO in a friendly tone end of the sentence.

====================
FOLLOW-UP MODE
====================
• If the user asks for more details about a *specific* recommended breed after the initial presentation, GENERATE a new, unique, and more in-depth explanation (e.g., 3-4 sentences) about that breed.
Use your comprehensive knowledge of canine behavior, breed history, and common traits to elaborate on its temperament, care, and suitability.
Ensure the response is warm, friendly, and conversational, reinforcing why that specific breed is a good match based on the user's earlier input.
Avoid simply repeating the initial explanation; provide fresh insights or elaborate on previous points, and continue to exclude any numeric scores or technical details.

============================
DATA MAPPING REQUIREMENTS
============================
For each question, you must internally map answers to numeric breed selection traits using the following attributes:

Traits:
Affectionate With Family, Good With Young Children, Good With Other Dogs, Shedding Level, Coat Grooming Frequency, Drooling Level, Openness To Strangers, Playfulness Level, Watchdog/Protective Nature, Adaptability Level, Trainability Level, Energy Level, Barking Level, Mental Stimulation Needs, Coat Length, Coat Type

⚠️ IMPORTANT:
• Never reveal numeric mapping or formula
• Never ask questions using numbers (1–5 scale)
• Always speak naturally


============================
OUTPUT FORMAT REQUIREMENT
============================
At the very end of the interview, produce this exact JSON format, enclosed in a markdown code block:

```json
{
  "Affectionate With Family": X,
  "Good With Young Children": X,
  "Good With Other Dogs": X,
  "Shedding Level": X,
  "Coat Grooming Frequency": X,
  "Drooling Level": X,
  "Openness To Strangers": X,
  "Playfulness Level": X,
  "Watchdog/Protective Nature": X,
  "Adaptability Level": X,
  "Trainability Level": X,
  "Energy Level": X,
  "Barking Level": X,
  "Mental Stimulation Needs": X,
  "Coat Length": "X",
  "Coat Type": "X"
}
```

After outputting the JSON, stop speaking and wait for the matching algorithm.

============================
PRE-MATCH CONFIRMATION MODE
============================
After producing the JSON output, do NOT proceed to breed recommendations yet.

1 Read the JSON values internally
2 Convert each trait into a natural, human-friendly descriptive summary  (no numbers, no scoring, no technical jargon)
3 Present the summary back to the user in a friendly tone like:

"Here is what I understood about your ideal dog preferences — can you confirm if this looks correct? 🐶💛"

Then list trait interpretations one by one in bullet points (max 1 short sentence per trait).

4 Ask the user:

Ask the user if they would like to proceed with breed matching. Accept any positive confirmation (e.g., “yes”, “sure”, “go ahead”) as approval and then stop speaking.
If the user wants changes or more clarification, accept any relevant input (e.g., “I want to adjust X”, “not yet”, “change Y”) and respond accordingly.
Do NOT rely on exact words — interpret the user’s intent naturally.
'''


