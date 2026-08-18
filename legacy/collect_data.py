import os
import sys
import time
import json
import pickle
import unicodedata
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gemma3-1b-tokenizer")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Italy?", "Rome"),
    ("What is 12 + 29?", "41"),
    ("What is 17 * 3?", "51"),
    ("What is 45 - 18?", "27"),
    ("What is 8 * 7?", "56"),
    ("What is 144 / 12?", "12"),
    ("What is 23 + 48?", "71"),
    ("What is 9 * 9?", "81"),
    ("What is 100 - 37?", "63"),
    ("What is 15 * 6?", "90"),
    ("What is 250 / 5?", "50"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("Who wrote the novel Moby-Dick?", "Melville"),
    ("Who painted the Mona Lisa?", "Leonardo"),
    ("Who painted Starry Night?", "Van Gogh"),
    ("Who discovered penicillin?", "Fleming"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many continents are there on Earth?", "7"),
    ("How many legs does a spider have?", "8"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the fastest land animal?", "Cheetah"),
    ("What is the currency of the United Kingdom?", "Pound"),
    ("What is the currency of Japan?", "Yen"),
    ("Which ocean is the largest?", "Pacific"),
    ("What is the tallest mountain on Earth?", "Everest"),
    ("What is the longest river in Africa?", "Nile"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many states are in the United States of America?", "50"),
]

HARD_QUESTIONS = [
    ("What is 137 + 258?", "395"),
    ("What is 47 * 19?", "893"),
    ("What is 314 - 165?", "149"),
    ("What is 832 divided by 16?", "52"),
    ("What is 23 * 47?", "1081"),
    ("What is 512 + 384?", "896"),
    ("What is 91 * 13?", "1183"),
    ("What is 1000 - 347?", "653"),
    ("What is 765 divided by 15?", "51"),
    ("What is 38 * 26?", "988", "38*26=988"),
    ("In which year was the Eiffel Tower completed?", "1889"),
    ("What is the capital of Kyrgyzstan?", "Bishkek"),
    ("How many bones are in the adult human body?", "206"),
    ("Which country has the most pyramids in the world?", "Sudan"),
    ("What is the longest river in South America?", "Amazon"),
    ("What is the largest desert in the world?", "Antarctic", "Antarctica"),
    ("What is the currency of Switzerland?", "Franc"),
    ("What is the smallest country in the world by area?", "Vatican"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("In which year did World War II end?", "1945"),
    ("Who was the first person to walk on the Moon?", "Armstrong"),
    ("What is the deepest point of the ocean?", "Mariana"),
    ("What is the fastest bird in the world?", "Peregrine"),
    ("What is the largest mammal in the world?", "blue whale"),
    ("What is the official language of Brazil?", "Portuguese"),
    ("Who wrote the epic poem The Odyssey?", "Homer"),
    ("What is the tallest building in the world?", "Burj Khalifa"),
    ("What is the capital of New Zealand?", "Wellington"),
    ("How many planets are in our solar system?", "8"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What element has the atomic number 79?", "Gold"),
    ("In which country is the Great Barrier Reef located?", "Australia"),
    ("What is the largest country in the world by area?", "Russia"),
    ("What is the most populous country in the world today?", "India"),
    ("What is the smallest continent in the world?", "Australia"),
    ("What is the capital of Ethiopia?", "Addis Ababa"),
    ("What is the currency of India?", "Rupee"),
    ("In which year was the first iPhone released?", "2007"),
    ("Who discovered the law of gravity?", "Newton"),
    ("What is the square root of 144?", "12"),
    ("What is 15 percent of 200?", "30"),
]

HARDQUEST2 = [
    ("What is 137 * 289?", "39593"),
    ("What is 482 * 63?", "30366"),
    ("What is 5917 + 2846?", "8763"),
    ("What is 8452 divided by 4?", "2113"),
    ("What is 23 * 17 * 11?", "4301"),
    ("What is 3125 - 1867?", "1258"),
    ("What is 97 * 86?", "8342"),
    ("What is 456 * 89?", "40584"),
    ("What is 123 * 456?", "56088"),
    ("What is 728 + 915?", "1643"),
    ("What is the second tallest mountain in the world?", "K2"),
    ("How many countries are there in Africa?", "54"),
    ("What is the capital of Burkina Faso?", "Ouagadougou"),
    ("What is the capital of Vanuatu?", "Port Vila"),
    ("What is the capital of Tuvalu?", "Funafuti"),
    ("What is the capital of Liechtenstein?", "Vaduz"),
    ("What is the capital of Mongolia?", "Ulaanbaatar"),
    ("What is the capital of Paraguay?", "Asuncion", "Asunción"),
    ("Who was the second president of the United States?", "Adams"),
    ("What is the seventh planet from the Sun?", "Uranus"),
    ("How many ribs does the human body have?", "24"),
    ("What is the capital of the island of Tasmania?", "Hobart"),
    ("Which ocean is the smallest in the world?", "Arctic"),
    ("In which year did the American Civil War begin?", "1861"),
    ("Who invented the telephone?", "Bell"),
    ("How many teeth does an adult human have?", "32"),
    ("What is the largest island in the world?", "Greenland"),
    ("What is the strongest muscle in the human body by force?", "Masseter"),
    ("What is the largest organ in the human body?", "skin"),
    ("What is the first element on the periodic table?", "Hydrogen"),
    ("What is the capital city of the state of California?", "Sacramento"),
    ("Which planet has the shortest day?", "Jupiter"),
    ("What is the longest venomous snake in the world?", "King cobra"),
    ("What is the speed of light in kilometers per second?", "299792", "300000"),
    ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo"),
    ("What is the capital of Alaska state?", "Juneau"),
    ("What year was the Mona Lisa painted?", "1503", "1506"),
    ("What is the largest lake in Africa?", "Victoria"),
    ("What is the capital of Myanmar?", "Naypyidaw", "Nay Pyi Taw"),
]

ALL_QUESTIONS = QUESTIONS + HARD_QUESTIONS + HARDQUEST2

def run(model, tok, question, max_new=48):
    text = "<start_of_turn>user\n" + question + "<end_of_turn>\n<start_of_turn>model\n"
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=True, temperature=0.9, top_p=0.9,
            return_dict_in_generate=True, output_hidden_states=True, use_cache=True,
        )
    gen_ids = out.sequences[0][ids.shape[1]:]
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if end_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.tolist().index(end_id)]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    steps = out.hidden_states
    n_steps = len(steps)
    L = len(steps[0]) - 1
    flow = torch.zeros(n_steps, L + 1, 1152, dtype=torch.float16)
    for t, step in enumerate(steps):
        for l in range(L + 1):
            flow[t, l] = step[l][0, -1, :]
    return gen_text, flow, n_steps

t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
tok = AutoTokenizer.from_pretrained(TOK_DIR)
print(f"model loaded in {time.time()-t0:.1f}s")

for i, item in enumerate(ALL_QUESTIONS):
    cache_file = os.path.join(OUT_DIR, f"ex_{i:03d}.pkl")
    if os.path.exists(cache_file):
        continue
    q, ans = item[0], item[1]
    alt = item[2:] if len(item) > 2 else ()
    t1 = time.time()
    gen_text, flow, n_steps = run(model, tok, q)
    gen_n = norm(gen_text)
    label = 1 if any(norm(a) in gen_n for a in (ans,) + alt) else 0
    record = {
        "id": i, "question": q, "answer": ans, "alt_answers": list(alt),
        "generated": gen_text, "label": label, "flow": flow.cpu(), "n_steps": n_steps,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(record, f)
    print(f"ex {i:03d} label={label} steps={n_steps} t={time.time()-t1:.1f}s | {gen_text[:70]!r}")
