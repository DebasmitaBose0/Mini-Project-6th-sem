"""
Large-Scale Sentence Similarity Dataset Generator
===================================================
Generates millions of (Original, Modified, Type, Similarity) rows.

Usage:
    python generate_million.py                  # default 1,000,000 rows
    python generate_million.py --rows 5000000   # custom count
    python generate_million.py --rows 100000 --output my_dataset.csv
"""

import csv
import random
import argparse
import time
import sys
from itertools import cycle

# ─────────────────────────────────────────────────────────
# 1. SEED DATA — 500+ original sentences across domains
# ─────────────────────────────────────────────────────────

SEED_SENTENCES = [
    # Daily life
    "I am going to the market",
    "She completed the work",
    "He is very tired",
    "They are playing football",
    "I like this book",
    "The teacher explains the topic",
    "We are eating dinner together",
    "She is reading a novel",
    "He drinks coffee every morning",
    "They went to the park yesterday",
    "I need to buy some groceries",
    "She cooked a delicious meal",
    "He is watching television",
    "We cleaned the entire house",
    "They are waiting for the bus",
    "I finished my homework",
    "She takes a walk every evening",
    "He fixed the broken window",
    "We planted flowers in the garden",
    "They celebrated his birthday",

    # Work / Professional
    "The manager approved the project",
    "She submitted the report on time",
    "He attended the meeting yesterday",
    "They discussed the new strategy",
    "I completed the presentation",
    "The team achieved their target",
    "She organized the conference",
    "He reviewed the document carefully",
    "We finalized the budget",
    "They hired a new employee",
    "The company launched a new product",
    "She received a promotion last month",
    "He handles customer complaints",
    "We implemented the new system",
    "They signed the contract today",
    "I scheduled a meeting for tomorrow",
    "The director announced the changes",
    "She trained the new recruits",
    "He prepared the quarterly report",
    "We evaluated the performance metrics",

    # Education
    "The students passed the examination",
    "She teaches mathematics at the university",
    "He studied for the test all night",
    "They submitted their assignments",
    "I enrolled in a new course",
    "The professor gave a lecture on physics",
    "She earned a scholarship",
    "He completed his dissertation",
    "We participated in the science fair",
    "They learned a new programming language",
    "The school organized a field trip",
    "She presented her research findings",
    "He graduated from the university",
    "We solved the math problem together",
    "They read the textbook chapter",
    "I wrote an essay on climate change",
    "The librarian arranged the books",
    "She memorized all the formulas",
    "He passed the entrance exam",
    "We attended the workshop on data science",

    # Technology
    "The developer fixed the bug",
    "She designed the user interface",
    "He installed the software update",
    "They built a mobile application",
    "I created a new database",
    "The server crashed last night",
    "She wrote an algorithm for sorting",
    "He configured the network settings",
    "We deployed the application successfully",
    "They tested the new feature",
    "The system processes thousands of requests",
    "She optimized the search function",
    "He backed up the important files",
    "We migrated to the cloud platform",
    "They updated the security protocols",
    "I debugged the code for several hours",
    "The robot performed the task accurately",
    "She analyzed the data using Python",
    "He automated the workflow process",
    "We integrated the payment gateway",

    # Health
    "The doctor examined the patient",
    "She takes medicine every day",
    "He recovered from the surgery",
    "They visited the hospital yesterday",
    "I exercise for thirty minutes daily",
    "The nurse administered the injection",
    "She maintains a healthy diet",
    "He scheduled a dental appointment",
    "We participated in the health camp",
    "They donated blood at the clinic",
    "The patient felt better after rest",
    "She monitors her blood pressure regularly",
    "He consulted a specialist",
    "We completed the vaccination drive",
    "They organized a yoga session",

    # Travel & Nature
    "The tourists visited the ancient temple",
    "She traveled to Paris last summer",
    "He climbed the mountain successfully",
    "They explored the dense forest",
    "I booked a flight to London",
    "The river flows through the valley",
    "She photographed the beautiful sunset",
    "He camped near the lake",
    "We drove along the coastal highway",
    "They sailed across the ocean",
    "The birds are migrating south",
    "She hiked the trail for five hours",
    "He discovered a hidden waterfall",
    "We visited the national park",
    "They enjoyed the beach vacation",
    "I packed my luggage for the trip",
    "The train arrived at the station",
    "She explored the local markets",
    "He rented a car for the road trip",
    "We watched the sunrise from the hilltop",

    # Food & Cooking
    "The chef prepared an Italian dish",
    "She baked a chocolate cake",
    "He ordered pizza for dinner",
    "They tried the new restaurant",
    "I made fresh orange juice",
    "The waiter served the food quickly",
    "She followed the recipe carefully",
    "He grilled the chicken perfectly",
    "We enjoyed the homemade pasta",
    "They bought vegetables from the market",
    "The bakery sells fresh bread every morning",
    "She decorated the birthday cake",
    "He brewed a cup of strong coffee",
    "We organized a barbecue party",
    "They shared the meal with friends",

    # Sports
    "The team won the championship",
    "She scored the winning goal",
    "He ran the marathon in three hours",
    "They practiced every day after school",
    "I joined the swimming club",
    "The coach trained the athletes",
    "She broke the national record",
    "He plays tennis on weekends",
    "We watched the cricket match",
    "They competed in the tournament",
    "The referee made a controversial decision",
    "She completed the race in first place",
    "He lifted the heavy weights",
    "We cheered for our favorite team",
    "They celebrated the victory",

    # Weather & Environment
    "The weather is very cold today",
    "She planted trees in the park",
    "He recycled the plastic bottles",
    "They cleaned the beach last weekend",
    "I noticed the sky turning dark",
    "The rain destroyed the crops",
    "She studied the effects of pollution",
    "He measured the temperature outside",
    "We observed the rainbow after the storm",
    "They reduced their carbon footprint",
    "The snow covered the entire city",
    "She collected rainwater for the garden",
    "He monitored the air quality levels",
    "We supported the environmental campaign",
    "They protested against deforestation",

    # Entertainment & Media
    "The actor performed brilliantly on stage",
    "She listened to music all evening",
    "He directed a short film",
    "They watched the movie together",
    "I read the newspaper this morning",
    "The band played at the concert",
    "She wrote lyrics for the new song",
    "He edited the video for the project",
    "We attended the art exhibition",
    "They enjoyed the comedy show",
    "The museum displayed ancient artifacts",
    "She painted a beautiful landscape",
    "He published his first novel",
    "We organized a cultural festival",
    "They performed a traditional dance",

    # Family & Relationships
    "The mother prepared breakfast for everyone",
    "She called her parents last night",
    "He surprised his wife with flowers",
    "They visited their grandparents",
    "I helped my brother with his project",
    "The father taught his son to ride a bicycle",
    "She invited her friends for dinner",
    "He played with his children in the park",
    "We gathered for the family reunion",
    "They shared memories at the dinner table",
    "The siblings planned a surprise party",
    "She wrote a letter to her grandmother",
    "He accompanied his daughter to school",
    "We celebrated the anniversary together",
    "They adopted a puppy from the shelter",

    # Science & Research
    "The scientist discovered a new species",
    "She conducted the experiment successfully",
    "He published a research paper",
    "They analyzed the laboratory results",
    "I observed the chemical reaction",
    "The researcher collected the samples",
    "She presented the hypothesis to the committee",
    "He calibrated the instruments carefully",
    "We reviewed the experimental data",
    "They confirmed the theory with evidence",
    "The astronaut orbited the earth",
    "She developed a vaccine for the disease",
    "He studied the behavior of electrons",
    "We measured the speed of light",
    "They documented the research findings",

    # Finance & Business
    "The investor bought company shares",
    "She calculated the total expenses",
    "He deposited money in the bank",
    "They reviewed the financial statements",
    "I opened a savings account",
    "The accountant prepared the tax returns",
    "She negotiated the business deal",
    "He analyzed the stock market trends",
    "We increased the marketing budget",
    "They approved the loan application",
    "The entrepreneur started a new venture",
    "She managed the company finances",
    "He paid the monthly installment",
    "We audited the accounts thoroughly",
    "They invested in renewable energy",

    # Law & Government
    "The judge delivered the verdict",
    "She filed a complaint at the police station",
    "He testified as a witness",
    "They amended the constitution",
    "I renewed my driving license",
    "The lawyer presented the evidence",
    "She served on the jury",
    "He registered the property documents",
    "We voted in the general election",
    "They enforced the new regulation",
    "The parliament passed the bill",
    "She appealed the court decision",
    "He investigated the criminal case",
    "We followed the legal procedures",
    "They organized a peaceful protest",

    # Social & Community
    "The volunteers helped the flood victims",
    "She donated clothes to the orphanage",
    "He organized a charity event",
    "They built houses for the homeless",
    "I participated in the community cleanup",
    "The organization distributed food packages",
    "She mentored underprivileged children",
    "He raised funds for the school",
    "We supported the local businesses",
    "They launched an awareness campaign",

    # Communication
    "The reporter covered the breaking news",
    "She sent an email to her colleague",
    "He answered the phone immediately",
    "They published the article online",
    "I wrote a blog about technology",
    "The editor reviewed the manuscript",
    "She updated her social media profile",
    "He gave a speech at the conference",
    "We broadcast the event live",
    "They translated the document into French",

    # Animals & Pets
    "The dog chased the cat across the yard",
    "She fed the fish in the aquarium",
    "He trained the horse for the race",
    "They rescued the injured bird",
    "I walked the dog in the morning",
    "The cat slept on the sofa all day",
    "She groomed the poodle at the salon",
    "He adopted a kitten from the shelter",
    "We watched the dolphins at the aquarium",
    "They observed the monkeys at the zoo",

    # Home & Domestic
    "The plumber fixed the leaking pipe",
    "She arranged the furniture in the room",
    "He painted the walls of the bedroom",
    "They renovated the old house",
    "I washed the dishes after dinner",
    "The electrician repaired the wiring",
    "She hung the curtains in the living room",
    "He assembled the new bookshelf",
    "We decorated the house for the festival",
    "They installed air conditioning in the office",

    # Shopping
    "The customer returned the defective product",
    "She bought a dress for the party",
    "He compared prices at different stores",
    "They placed an order online",
    "I received the delivery this morning",
    "The store offered a discount on all items",
    "She tried on several pairs of shoes",
    "He purchased a laptop for work",
    "We visited the shopping mall",
    "They exchanged the gift for a different size",

    # Arts & Creativity
    "The sculptor carved a statue from marble",
    "She composed a symphony",
    "He designed the poster for the event",
    "They performed a piano recital",
    "I drew a portrait of my friend",
    "The architect designed the new building",
    "She embroidered a floral pattern",
    "He photographed the city skyline",
    "We created a mural on the school wall",
    "They choreographed a dance routine",

    # Miscellaneous
    "The boy kicked the ball over the fence",
    "She arranged the flowers in the vase",
    "He noticed a strange noise outside",
    "They discovered an old map in the attic",
    "I forgot my keys at home",
    "The clock stopped working yesterday",
    "She counted the stars in the sky",
    "He solved the puzzle in ten minutes",
    "We found a shortcut through the forest",
    "They organized the files alphabetically",
    "The baby started crying at midnight",
    "She polished the silverware for the party",
    "He carried the heavy boxes upstairs",
    "We assembled the tent at the campsite",
    "They returned the library books on time",
    "I locked the door before leaving",
    "The wind blew the papers off the desk",
    "She ironed the clothes for the interview",
    "He replaced the battery in the remote",
    "We collected shells at the seashore",
]

# ─────────────────────────────────────────────────────────
# 2. SYNONYM / WORD-LEVEL REPLACEMENT BANKS
# ─────────────────────────────────────────────────────────

SYNONYM_MAP = {
    "completed": ["finished", "accomplished", "concluded", "wrapped up"],
    "finished": ["completed", "accomplished", "concluded"],
    "submitted": ["handed in", "turned in", "delivered", "presented"],
    "discussed": ["talked about", "deliberated", "debated", "conversed about"],
    "achieved": ["accomplished", "attained", "reached", "met"],
    "organized": ["arranged", "coordinated", "planned", "set up"],
    "reviewed": ["examined", "analyzed", "evaluated", "assessed"],
    "implemented": ["executed", "carried out", "put into practice", "deployed"],
    "prepared": ["made ready", "set up", "arranged", "got ready"],
    "evaluated": ["assessed", "appraised", "judged", "examined"],
    "designed": ["created", "crafted", "developed", "devised"],
    "built": ["constructed", "assembled", "developed", "created"],
    "tested": ["examined", "evaluated", "checked", "verified"],
    "fixed": ["repaired", "mended", "corrected", "resolved"],
    "installed": ["set up", "configured", "deployed", "put in"],
    "created": ["made", "developed", "produced", "generated"],
    "analyzed": ["examined", "studied", "investigated", "assessed"],
    "optimized": ["improved", "enhanced", "refined", "streamlined"],
    "deployed": ["launched", "released", "rolled out", "published"],
    "automated": ["mechanized", "streamlined", "systematized"],
    "examined": ["inspected", "checked", "investigated", "studied"],
    "recovered": ["healed", "got better", "recuperated", "bounced back"],
    "visited": ["went to", "called on", "stopped by", "toured"],
    "traveled": ["journeyed", "went", "voyaged", "toured"],
    "climbed": ["ascended", "scaled", "went up", "hiked up"],
    "explored": ["investigated", "discovered", "surveyed", "ventured through"],
    "photographed": ["captured", "took a photo of", "snapped a picture of"],
    "discovered": ["found", "uncovered", "came across", "stumbled upon"],
    "enjoyed": ["liked", "appreciated", "relished", "had fun with"],
    "celebrated": ["commemorated", "marked", "observed", "honored"],
    "prepared": ["cooked", "made", "put together", "arranged"],
    "followed": ["adhered to", "obeyed", "stuck to", "complied with"],
    "scored": ["got", "earned", "achieved", "netted"],
    "practiced": ["rehearsed", "trained", "drilled", "exercised"],
    "competed": ["contended", "rivaled", "participated", "vied"],
    "performed": ["acted", "executed", "carried out", "delivered"],
    "published": ["released", "issued", "put out", "circulated"],
    "painted": ["colored", "drew", "illustrated", "decorated"],
    "resolved": ["solved", "settled", "fixed", "sorted out"],
    "answered": ["responded to", "replied to", "addressed"],
    "wrote": ["composed", "authored", "penned", "drafted"],
    "read": ["perused", "studied", "went through", "looked over"],
    "watched": ["observed", "viewed", "saw", "witnessed"],
    "helped": ["assisted", "aided", "supported", "gave a hand to"],
    "started": ["began", "commenced", "initiated", "kicked off"],
    "bought": ["purchased", "acquired", "obtained", "got"],
    "sold": ["marketed", "traded", "vended"],
    "gave": ["provided", "presented", "offered", "handed"],
    "took": ["grabbed", "seized", "picked up", "collected"],
    "made": ["created", "produced", "crafted", "built"],
    "said": ["stated", "mentioned", "declared", "remarked"],
    "told": ["informed", "notified", "advised"],
    "asked": ["inquired", "questioned", "requested"],
    "showed": ["demonstrated", "displayed", "presented", "revealed"],
    "played": ["engaged in", "participated in", "took part in"],
    "walked": ["strolled", "ambled", "wandered", "hiked"],
    "ran": ["sprinted", "dashed", "jogged", "rushed"],
    "won": ["triumphed", "prevailed", "succeeded", "claimed victory"],
    "lost": ["misplaced", "forfeited", "was defeated"],
    "opened": ["unlocked", "unlatched", "unsealed"],
    "closed": ["shut", "sealed", "locked"],
    "big": ["large", "huge", "enormous", "massive"],
    "small": ["tiny", "little", "compact", "miniature"],
    "happy": ["glad", "joyful", "pleased", "delighted"],
    "sad": ["unhappy", "sorrowful", "gloomy", "down"],
    "tired": ["exhausted", "fatigued", "weary", "worn out"],
    "very": ["extremely", "really", "incredibly", "exceptionally"],
    "quickly": ["rapidly", "swiftly", "fast", "speedily"],
    "carefully": ["cautiously", "meticulously", "attentively", "diligently"],
    "successfully": ["effectively", "triumphantly", "with success"],
    "immediately": ["instantly", "right away", "at once", "promptly"],
    "beautiful": ["gorgeous", "stunning", "lovely", "exquisite"],
    "important": ["crucial", "vital", "significant", "essential"],
    "new": ["brand-new", "fresh", "novel", "recent"],
    "old": ["ancient", "aged", "vintage", "antique"],
    "good": ["excellent", "great", "fine", "wonderful"],
    "bad": ["terrible", "awful", "poor", "dreadful"],
    "difficult": ["challenging", "tough", "hard", "demanding"],
    "easy": ["simple", "straightforward", "effortless"],
    "interesting": ["fascinating", "intriguing", "captivating", "engaging"],
    "morning": ["dawn", "daybreak", "sunrise"],
    "evening": ["dusk", "twilight", "nightfall"],
    "market": ["bazaar", "marketplace", "store"],
    "house": ["home", "residence", "dwelling"],
    "car": ["vehicle", "automobile"],
    "road": ["street", "path", "route", "highway"],
    "money": ["cash", "funds", "currency"],
    "work": ["job", "task", "assignment", "labor"],
    "book": ["novel", "publication", "volume"],
    "food": ["meal", "cuisine", "nourishment"],
    "friend": ["buddy", "pal", "companion", "mate"],
    "children": ["kids", "youngsters", "little ones"],
    "people": ["individuals", "folks", "persons"],
    "school": ["academy", "institution", "educational facility"],
    "problem": ["issue", "challenge", "difficulty"],
    "idea": ["concept", "notion", "thought"],
    "place": ["location", "spot", "site", "area"],
    "country": ["nation", "state", "land"],
    "city": ["town", "metropolis", "urban area"],
    "quickly": ["fast", "swiftly", "rapidly", "speedily"],
}

# Informal contractions and casual phrasing
INFORMAL_MAP = {
    "I am": "I'm",
    "He is": "He's",
    "She is": "She's",
    "It is": "It's",
    "We are": "We're",
    "They are": "They're",
    "You are": "You're",
    "I have": "I've",
    "He has": "He's",
    "She has": "She's",
    "We have": "We've",
    "They have": "They've",
    "I will": "I'll",
    "He will": "He'll",
    "She will": "She'll",
    "We will": "We'll",
    "They will": "They'll",
    "I would": "I'd",
    "He would": "He'd",
    "She would": "She'd",
    "We would": "We'd",
    "They would": "They'd",
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "cannot": "can't",
    "can not": "can't",
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    "going to": "gonna",
    "want to": "wanna",
    "got to": "gotta",
    "kind of": "kinda",
    "sort of": "sorta",
    "a lot of": "lots of",
    "very": "really",
    "extremely": "super",
    "yesterday": "last day",
}

# Sentence-level adverbs / hedges for expansion
EXPANSION_PREFIXES = [
    "Actually, ", "In fact, ", "Honestly, ", "To be honest, ",
    "As a matter of fact, ", "Interestingly, ", "Clearly, ",
    "Obviously, ", "Certainly, ", "Undoubtedly, ",
    "Without a doubt, ", "Needless to say, ",
]

EXPANSION_SUFFIXES = [
    " right now", " at this moment", " as we speak",
    " without any delay", " quite effectively",
    " with great effort", " very well",
    " as expected", " once again", " for the first time",
    " in an impressive manner", " to everyone's surprise",
    " as planned", " on schedule", " ahead of time",
]

# Modals for modal-change transformation
MODAL_INSERTIONS = [
    ("might", 0.70), ("could", 0.72), ("should", 0.75),
    ("would", 0.73), ("may", 0.71), ("must", 0.78),
    ("ought to", 0.72), ("can", 0.80),
]

# Clause-combining templates
CLAUSE_TEMPLATES = [
    "Having {verb_phrase}, {subject} {rest}",
    "After {verb_phrase}, {subject} {rest}",
    "While {verb_phrase}, {subject} {rest}",
    "Since {subject} {verb_phrase}, {rest}",
    "Because {subject} {verb_phrase}, {rest}",
    "Although {subject} {verb_phrase}, {rest}",
    "Before {verb_phrase}, {subject} {rest}",
    "Once {subject} {verb_phrase}, {rest}",
]

# Negation forms
NEGATION_VERBS = {
    "completed": "did not complete",
    "finished": "did not finish",
    "submitted": "did not submit",
    "achieved": "did not achieve",
    "organized": "did not organize",
    "reviewed": "did not review",
    "prepared": "did not prepare",
    "designed": "did not design",
    "built": "did not build",
    "fixed": "did not fix",
    "tested": "did not test",
    "installed": "did not install",
    "created": "did not create",
    "deployed": "did not deploy",
    "examined": "did not examine",
    "recovered": "did not recover",
    "visited": "did not visit",
    "traveled": "did not travel",
    "climbed": "did not climb",
    "explored": "did not explore",
    "discovered": "did not discover",
    "enjoyed": "did not enjoy",
    "celebrated": "did not celebrate",
    "scored": "did not score",
    "practiced": "did not practice",
    "performed": "did not perform",
    "published": "did not publish",
    "painted": "did not paint",
    "answered": "did not answer",
    "wrote": "did not write",
    "watched": "did not watch",
    "helped": "did not help",
    "started": "did not start",
    "bought": "did not buy",
    "gave": "did not give",
    "took": "did not take",
    "made": "did not make",
    "won": "did not win",
    "lost": "did not lose",
    "played": "did not play",
    "walked": "did not walk",
    "ran": "did not run",
    "likes": "does not like",
    "explains": "does not explain",
    "drinks": "does not drink",
    "takes": "does not take",
    "teaches": "does not teach",
    "plays": "does not play",
    "processes": "does not process",
    "sells": "does not sell",
    "flows": "does not flow",
}


# ─────────────────────────────────────────────────────────
# 3. TRANSFORMATION FUNCTIONS
# ─────────────────────────────────────────────────────────

def compute_similarity(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity, better than char-level for sentences."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return round(len(intersection) / len(union), 2)


def transform_synonym(sentence: str) -> tuple[str, float]:
    """Replace one or more words with synonyms."""
    words = sentence.split()
    replaced = False
    for i, w in enumerate(words):
        w_lower = w.lower().rstrip(".,!?")
        if w_lower in SYNONYM_MAP:
            synonyms = SYNONYM_MAP[w_lower]
            replacement = random.choice(synonyms)
            # preserve capitalization
            if w[0].isupper():
                replacement = replacement.capitalize()
            # preserve trailing punctuation
            trail = ""
            if w[-1] in ".,!?":
                trail = w[-1]
            words[i] = replacement + trail
            replaced = True
            if random.random() < 0.5:
                break  # sometimes replace just one word
    result = " ".join(words)
    if not replaced:
        result = sentence  # fallback
    sim = round(random.uniform(0.85, 0.98), 2) if replaced else 1.0
    return result, sim


def transform_rephrase(sentence: str) -> tuple[str, float]:
    """Rephrase by reordering clause structure."""
    words = sentence.split()
    if len(words) < 4:
        return sentence, 1.0

    # Multiple rephrase strategies
    strategy = random.choice(["front_load", "mid_split", "emphasis"])

    if strategy == "front_load":
        # Move last few words to front
        n = random.randint(2, min(3, len(words) - 2))
        tail = words[-n:]
        head = words[:-n]
        # Clean punctuation
        if tail[-1][-1] in ".,!?":
            punct = tail[-1][-1]
            tail[-1] = tail[-1][:-1]
        else:
            punct = ""
        if head[0][0].isupper():
            head[0] = head[0][0].lower() + head[0][1:]
        tail[0] = tail[0].capitalize()
        result = " ".join(tail) + " " + " ".join(head) + punct
    elif strategy == "mid_split":
        mid = len(words) // 2
        first_half = words[:mid]
        second_half = words[mid:]
        if first_half[0][0].isupper():
            first_half[0] = first_half[0][0].lower() + first_half[0][1:]
        second_half[0] = second_half[0].capitalize()
        result = " ".join(second_half) + " " + " ".join(first_half)
    else:  # emphasis
        result = "It is " + sentence[0].lower() + sentence[1:]
        if not result.endswith("."):
            result += "."

    sim = round(random.uniform(0.82, 0.95), 2)
    return result, sim


def transform_passive(sentence: str) -> tuple[str, float]:
    """Convert active voice to passive voice using pattern matching."""
    words = sentence.split()
    if len(words) < 3:
        return sentence, 1.0

    # Clean trailing punctuation
    punct = ""
    if words[-1][-1] in ".,!?":
        punct = words[-1][-1]
        words[-1] = words[-1][:-1]

    # Patterns: Subject + Verb + Object
    # Simple: Find verb position (usually index 1 or 2)
    subject = words[0]
    
    # Handle compound subjects
    verb_idx = 1
    if words[1].lower() in ("is", "are", "was", "were", "has", "have", "had", "will", "can", "could", "should", "would", "may", "might"):
        if len(words) > 2 and words[2].lower() not in ("a", "an", "the", "very", "really"):
            # Auxiliary + main verb pattern
            aux = words[1]
            main_verb = words[2]
            obj = " ".join(words[3:]) if len(words) > 3 else None
            if obj:
                result = f"{obj.capitalize()} {aux.lower()} {main_verb} by {subject.lower()}{punct}"
                return result, round(random.uniform(0.85, 0.97), 2)

    # Simple past/present tense
    verb = words[1]
    obj = " ".join(words[2:]) if len(words) > 2 else None
    if obj:
        # Determine appropriate auxiliary
        if verb.endswith("ed"):
            aux = "was"
        elif verb.endswith("s") and not verb.endswith("ss"):
            aux = "is"
            verb = verb[:-1] + "ed" if not verb.endswith("ied") else verb
        else:
            aux = "was"

        result = f"{obj.capitalize()} {aux} {verb} by {subject.lower()}{punct}"
        return result, round(random.uniform(0.85, 0.97), 2)

    return sentence, 1.0


def transform_informal(sentence: str) -> tuple[str, float]:
    """Make sentence more casual/informal."""
    result = sentence
    applied = False

    for formal, informal in INFORMAL_MAP.items():
        if formal in result:
            result = result.replace(formal, informal, 1)
            applied = True
        elif formal.lower() in result.lower():
            # case-insensitive replacement
            idx = result.lower().find(formal.lower())
            if idx != -1:
                result = result[:idx] + informal + result[idx + len(formal):]
                applied = True

    # Add casual filler words sometimes
    if random.random() < 0.3:
        fillers = ["like, ", "you know, ", "basically, ", "well, "]
        filler = random.choice(fillers)
        result = filler.capitalize() + result[0].lower() + result[1:]
        applied = True

    sim = round(random.uniform(0.75, 0.92), 2) if applied else 1.0
    return result, sim


def transform_tense_change(sentence: str) -> tuple[str, float]:
    """Shift tense of the sentence."""
    words = sentence.split()
    result_words = list(words)
    changed = False

    # Common tense shifts
    tense_shifts = {
        # present -> past
        "is": "was", "are": "were", "am": "was",
        "has": "had", "have": "had",
        "does": "did", "do": "did",
        "goes": "went", "go": "went",
        "takes": "took", "plays": "played",
        "drinks": "drank", "explains": "explained",
        "teaches": "taught", "likes": "liked",
        "processes": "processed", "sells": "sold",
        "flows": "flowed",
        # past -> present continuous
        "completed": "is completing", "finished": "is finishing",
        "submitted": "is submitting", "discussed": "are discussing",
        "achieved": "is achieving", "organized": "is organizing",
        "reviewed": "is reviewing", "prepared": "is preparing",
        "fixed": "is fixing", "tested": "is testing",
        "built": "is building", "created": "is creating",
        "designed": "is designing", "analyzed": "is analyzing",
        "examined": "is examining", "visited": "is visiting",
        "traveled": "is traveling", "climbed": "is climbing",
        "explored": "is exploring", "discovered": "is discovering",
        "celebrated": "is celebrating", "scored": "is scoring",
        "practiced": "is practicing", "performed": "is performing",
        "published": "is publishing", "wrote": "is writing",
        "watched": "is watching", "helped": "is helping",
        "started": "is starting", "bought": "is buying",
        "gave": "is giving", "won": "is winning",
        "walked": "is walking", "ran": "is running",
        "cooked": "is cooking", "baked": "is baking",
        "grilled": "is grilling", "served": "is serving",
        "painted": "is painting", "hired": "is hiring",
        "trained": "is training", "signed": "is signing",
        "launched": "is launching", "received": "is receiving",
        "planted": "is planting",
    }

    for i, w in enumerate(result_words):
        w_clean = w.rstrip(".,!?")
        trail = w[len(w_clean):]
        if w_clean.lower() in tense_shifts:
            replacement = tense_shifts[w_clean.lower()]
            if w_clean[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result_words[i] = replacement + trail
            changed = True
            break

    result = " ".join(result_words)
    sim = round(random.uniform(0.80, 0.93), 2) if changed else 1.0
    return result, sim


def transform_modal_change(sentence: str) -> tuple[str, float]:
    """Add or change modality (might, could, should, etc.)."""
    words = sentence.split()
    if len(words) < 3:
        return sentence, 1.0

    modal, sim_base = random.choice(MODAL_INSERTIONS)

    # Find verb position and insert modal before it
    # Skip subject (index 0), check if there's already an auxiliary
    if words[1].lower() in ("is", "are", "was", "were", "has", "have", "had",
                             "will", "can", "could", "should", "would", "may", "might"):
        # Replace existing auxiliary with modal
        words[1] = modal
    else:
        # Insert modal before verb (index 1)
        verb = words[1].rstrip(".,!?")
        # Convert to base form roughly
        if verb.endswith("ed"):
            base = verb[:-2] if not verb.endswith("ied") else verb[:-3] + "y"
        elif verb.endswith("s") and not verb.endswith("ss"):
            base = verb[:-1]
        else:
            base = verb
        trail = words[1][len(verb):]
        words[1] = modal + " " + base + trail

    result = " ".join(words)
    sim = round(sim_base + random.uniform(-0.05, 0.05), 2)
    return result, sim


def transform_expansion(sentence: str) -> tuple[str, float]:
    """Expand the sentence with additional words or phrases."""
    strategy = random.choice(["prefix", "suffix", "both"])
    result = sentence

    # Remove trailing period for manipulation
    punct = ""
    if result.endswith("."):
        punct = "."
        result = result[:-1]

    if strategy in ("prefix", "both"):
        prefix = random.choice(EXPANSION_PREFIXES)
        result = prefix + result[0].lower() + result[1:]

    if strategy in ("suffix", "both"):
        suffix = random.choice(EXPANSION_SUFFIXES)
        result = result + suffix

    result = result + punct
    # Ensure first letter is uppercase
    result = result[0].upper() + result[1:]

    sim = round(random.uniform(0.88, 0.97), 2)
    return result, sim


def transform_reorder(sentence: str) -> tuple[str, float]:
    """Reorder the sentence structure while preserving meaning."""
    words = sentence.split()
    if len(words) < 5:
        return sentence, 1.0

    punct = ""
    if words[-1][-1] in ".,!?":
        punct = words[-1][-1]
        words[-1] = words[-1][:-1]

    # Strategy: move prepositional phrase or last part to front
    # Find prepositions
    preps = {"in", "at", "on", "to", "for", "from", "with", "by", "about",
             "after", "before", "during", "through", "across", "near"}

    prep_idx = None
    for i, w in enumerate(words):
        if w.lower() in preps and i > 1:
            prep_idx = i
            break

    if prep_idx and prep_idx < len(words) - 1:
        tail = words[prep_idx:]
        head = words[:prep_idx]
        tail[0] = tail[0].capitalize()
        head[0] = head[0].lower()
        result = " ".join(tail) + ", " + " ".join(head) + punct
    else:
        # Simple swap: last 2-3 words to front
        n = random.randint(2, min(3, len(words) - 2))
        tail = words[-n:]
        head = words[:-n]
        tail[0] = tail[0].capitalize()
        head[0] = head[0].lower()
        result = " ".join(tail) + " is where " + " ".join(head) + punct

    sim = round(random.uniform(0.82, 0.93), 2)
    return result, sim


def transform_negation(sentence: str) -> tuple[str, float]:
    """Negate the sentence, producing an opposite-meaning variant."""
    words = sentence.split()
    result_words = list(words)
    changed = False

    for i, w in enumerate(result_words):
        w_clean = w.rstrip(".,!?")
        trail = w[len(w_clean):]
        if w_clean.lower() in NEGATION_VERBS:
            neg = NEGATION_VERBS[w_clean.lower()]
            if w_clean[0].isupper():
                neg = neg[0].upper() + neg[1:]
            result_words[i] = neg + trail
            changed = True
            break

    result = " ".join(result_words)
    sim = round(random.uniform(0.40, 0.65), 2) if changed else 1.0
    return result, sim


def transform_question(sentence: str) -> tuple[str, float]:
    """Convert statement to a question."""
    words = sentence.split()
    if len(words) < 3:
        return sentence, 1.0

    punct = "?"
    # Remove trailing punctuation
    if words[-1][-1] in ".,!?":
        words[-1] = words[-1][:-1]

    # If starts with subject + aux verb
    aux_verbs = {"is", "are", "was", "were", "has", "have", "had",
                 "will", "can", "could", "should", "would", "may", "might", "do", "does", "did"}

    if words[1].lower() in aux_verbs:
        # Invert: "She is reading" -> "Is she reading?"
        aux = words[1]
        subject = words[0].lower()
        rest = " ".join(words[2:])
        result = f"{aux.capitalize()} {subject} {rest}{punct}"
    else:
        # Add "Did" for past tense
        subject = words[0].lower()
        verb = words[1]
        rest = " ".join(words[2:])

        # Try to get base form
        if verb.endswith("ed"):
            base = verb[:-2] if not verb.endswith("ied") else verb[:-3] + "y"
            result = f"Did {subject} {base} {rest}{punct}"
        elif verb.endswith("s") and not verb.endswith("ss"):
            base = verb[:-1]
            result = f"Does {subject} {base} {rest}{punct}"
        else:
            result = f"Did {subject} {verb} {rest}{punct}"

    sim = round(random.uniform(0.70, 0.85), 2)
    return result, sim


def transform_formal(sentence: str) -> tuple[str, float]:
    """Make the sentence more formal/academic."""
    result = sentence
    applied = False

    formal_replacements = {
        "a lot of": "a considerable amount of",
        "get": "obtain",
        "got": "obtained",
        "help": "assist",
        "helped": "assisted",
        "start": "commence",
        "started": "commenced",
        "buy": "purchase",
        "bought": "purchased",
        "try": "attempt",
        "tried": "attempted",
        "show": "demonstrate",
        "showed": "demonstrated",
        "use": "utilize",
        "used": "utilized",
        "need": "require",
        "needed": "required",
        "give": "provide",
        "gave": "provided",
        "make": "construct",
        "find": "locate",
        "found": "located",
        "think": "consider",
        "tell": "inform",
        "told": "informed",
        "ask": "inquire",
        "asked": "inquired",
        "want": "desire",
        "wanted": "desired",
        "keep": "maintain",
        "end": "conclude",
        "ended": "concluded",
        "begin": "initiate",
        "check": "verify",
        "look at": "examine",
    }

    # Apply at most 2 replacements
    count = 0
    for informal_word, formal_word in formal_replacements.items():
        if informal_word in result.lower() and count < 2:
            idx = result.lower().find(informal_word)
            if idx != -1:
                original = result[idx:idx + len(informal_word)]
                if original[0].isupper():
                    formal_word = formal_word.capitalize()
                result = result[:idx] + formal_word + result[idx + len(informal_word):]
                applied = True
                count += 1

    # Sometimes add formal prefix
    if random.random() < 0.3:
        formal_prefixes = [
            "It is noteworthy that ", "It should be noted that ",
            "It is evident that ", "One may observe that ",
            "It is apparent that ",
        ]
        prefix = random.choice(formal_prefixes)
        result = prefix + result[0].lower() + result[1:]
        applied = True

    sim = round(random.uniform(0.80, 0.95), 2) if applied else 1.0
    return result, sim


def transform_conditional(sentence: str) -> tuple[str, float]:
    """Wrap sentence in a conditional clause."""
    conditions = [
        "If time permits, ", "If possible, ", "If needed, ",
        "If everything goes well, ", "If the weather is good, ",
        "If the schedule allows, ", "If conditions are favorable, ",
        "If the opportunity arises, ", "If resources are available, ",
        "In case of necessity, ", "Should the need arise, ",
        "Provided that conditions are met, ",
    ]
    condition = random.choice(conditions)
    result = condition + sentence[0].lower() + sentence[1:]
    sim = round(random.uniform(0.65, 0.80), 2)
    return result, sim


def transform_emphasis(sentence: str) -> tuple[str, float]:
    """Add emphasis or intensifiers."""
    emphasis_words = [
        "definitely ", "absolutely ", "certainly ", "undoubtedly ",
        "clearly ", "without question ", "no doubt ",
        "truly ", "genuinely ", "remarkably ",
    ]
    words = sentence.split()
    if len(words) < 3:
        return sentence, 1.0

    # Insert emphasis word before the verb (index 1 or 2)
    insert_idx = min(2, len(words) - 1)

    # If there's an auxiliary verb, insert after it
    if words[1].lower() in ("is", "are", "was", "were", "has", "have", "had",
                            "will", "can", "could", "should", "would"):
        insert_idx = 2

    emp = random.choice(emphasis_words)
    words.insert(insert_idx, emp.strip())
    result = " ".join(words)
    sim = round(random.uniform(0.88, 0.97), 2)
    return result, sim


def transform_simplification(sentence: str) -> tuple[str, float]:
    """Simplify the sentence by removing modifiers and shortening."""
    words = sentence.split()
    # Remove common modifiers
    modifiers = {"very", "really", "extremely", "incredibly", "absolutely",
                 "quite", "rather", "fairly", "pretty", "somewhat",
                 "always", "usually", "often", "sometimes", "never",
                 "carefully", "quickly", "slowly", "successfully",
                 "thoroughly", "completely", "entirely", "perfectly",
                 "brilliantly", "beautifully", "effectively", "accurately"}
    result_words = [w for w in words if w.lower().rstrip(".,!?") not in modifiers]
    if len(result_words) < len(words):
        result = " ".join(result_words)
        sim = round(random.uniform(0.78, 0.92), 2)
    else:
        result = sentence
        sim = 1.0
    return result, sim


def transform_unrelated(sentence: str) -> tuple[str, float]:
    """Generate a completely unrelated sentence (negative pair)."""
    unrelated_pool = [
        "The sun rises in the east every morning",
        "Quantum physics describes atomic behavior",
        "The population of Tokyo exceeds thirteen million",
        "Water boils at one hundred degrees Celsius",
        "Photosynthesis converts sunlight into energy",
        "The Great Wall of China is visible from space",
        "Mozart composed his first symphony at age eight",
        "Honey never spoils if stored properly",
        "The speed of light is approximately three hundred thousand kilometers per second",
        "Dolphins are known for their intelligence",
        "The Amazon River is the largest by volume",
        "The Pyramids of Giza were built over four thousand years ago",
        "DNA carries genetic information in all living organisms",
        "The Pacific Ocean is the largest ocean on Earth",
        "Chess was invented in ancient India",
        "The human body contains approximately sixty percent water",
        "Mount Everest is the tallest mountain in the world",
        "Chocolate was first consumed by the ancient Mayans",
        "The earth orbits the sun once every year",
        "Lightning can reach temperatures five times hotter than the sun",
        "Elephants are the largest land animals",
        "The violin has four strings",
        "Coffee beans are actually seeds",
        "The Sahara Desert is expanding each year",
        "Octopuses have three hearts",
        "A group of flamingos is called a flamboyance",
        "Bananas are technically berries",
        "The Eiffel Tower grows taller in summer",
        "Sharks have been around longer than trees",
        "Honey bees can recognize human faces",
    ]
    result = random.choice(unrelated_pool)
    sim = round(random.uniform(0.05, 0.25), 2)
    return result, sim


# ─────────────────────────────────────────────────────────
# 4. ALL TRANSFORMATIONS REGISTRY
# ─────────────────────────────────────────────────────────

TRANSFORMATIONS = [
    ("Synonym", transform_synonym, 0.14),
    ("Rephrase", transform_rephrase, 0.10),
    ("Passive", transform_passive, 0.08),
    ("Informal", transform_informal, 0.08),
    ("Tense change", transform_tense_change, 0.08),
    ("Modal change", transform_modal_change, 0.06),
    ("Expansion", transform_expansion, 0.07),
    ("Reorder", transform_reorder, 0.06),
    ("Negation", transform_negation, 0.05),
    ("Question", transform_question, 0.05),
    ("Formal", transform_formal, 0.06),
    ("Conditional", transform_conditional, 0.05),
    ("Emphasis", transform_emphasis, 0.04),
    ("Simplification", transform_simplification, 0.04),
    ("Unrelated", transform_unrelated, 0.04),
]

# Normalize weights
_total_weight = sum(w for _, _, w in TRANSFORMATIONS)
TRANSFORMATION_WEIGHTS = [w / _total_weight for _, _, w in TRANSFORMATIONS]
TRANSFORMATION_NAMES = [name for name, _, _ in TRANSFORMATIONS]
TRANSFORMATION_FUNCS = [func for _, func, _ in TRANSFORMATIONS]


# ─────────────────────────────────────────────────────────
# 5. MAIN GENERATOR
# ─────────────────────────────────────────────────────────

def generate_dataset(target_rows: int, output_file: str, batch_size: int = 50000):
    """Generate a large-scale sentence similarity dataset."""

    print(f"🚀 Generating {target_rows:,} rows → {output_file}")
    print(f"   Seed sentences: {len(SEED_SENTENCES)}")
    print(f"   Transformation types: {len(TRANSFORMATIONS)}")
    print(f"   Batch size: {batch_size:,}")
    print()

    fieldnames = ["Original Sentence", "Modified Sentence", "Type", "Similarity"]

    start_time = time.time()
    rows_written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch = []
        sentence_cycle = cycle(SEED_SENTENCES)

        while rows_written < target_rows:
            original = next(sentence_cycle)

            # Pick a random transformation based on weights
            idx = random.choices(range(len(TRANSFORMATIONS)), weights=TRANSFORMATION_WEIGHTS, k=1)[0]
            transform_name = TRANSFORMATION_NAMES[idx]
            transform_func = TRANSFORMATION_FUNCS[idx]

            modified, similarity = transform_func(original)

            # Skip if modified is identical (no change was made) — unless it's intentional
            if modified == original and transform_name != "Unrelated":
                # Try another transformation
                for retry_idx in random.sample(range(len(TRANSFORMATIONS)), len(TRANSFORMATIONS)):
                    m, s = TRANSFORMATION_FUNCS[retry_idx](original)
                    if m != original:
                        modified, similarity = m, s
                        transform_name = TRANSFORMATION_NAMES[retry_idx]
                        break

            batch.append({
                "Original Sentence": original,
                "Modified Sentence": modified,
                "Type": transform_name,
                "Similarity": similarity,
            })

            if len(batch) >= batch_size:
                writer.writerows(batch)
                rows_written += len(batch)
                elapsed = time.time() - start_time
                rate = rows_written / elapsed if elapsed > 0 else 0
                pct = (rows_written / target_rows) * 100
                print(f"   ✅ {rows_written:>10,} / {target_rows:,} rows  "
                      f"({pct:5.1f}%)  |  {rate:,.0f} rows/sec  |  "
                      f"elapsed: {elapsed:.1f}s", flush=True)
                batch = []

        # Write remaining rows
        if batch:
            writer.writerows(batch)
            rows_written += len(batch)

    elapsed = time.time() - start_time
    rate = rows_written / elapsed if elapsed > 0 else 0

    print()
    print(f"✨ Done! Generated {rows_written:,} rows in {elapsed:.1f}s ({rate:,.0f} rows/sec)")
    print(f"   Output: {output_file}")

    # Print sample
    print("\n📋 Sample rows:")
    print("-" * 120)
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            print(f"  {row['Original Sentence'][:40]:<42} → "
                  f"{row['Modified Sentence'][:45]:<47} | "
                  f"{row['Type']:<18} | {row['Similarity']}")
    print("-" * 120)

    # Print distribution
    print("\n📊 Transformation type distribution:")
    type_counts = {}
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["Type"]
            type_counts[t] = type_counts.get(t, 0) + 1

    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = (c / rows_written) * 100
        bar = "█" * int(pct / 2)
        print(f"  {t:<20} {c:>10,}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────
# 6. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a large-scale sentence similarity dataset"
    )
    parser.add_argument(
        "--rows", type=int, default=1_000_000,
        help="Number of rows to generate (default: 1,000,000)"
    )
    parser.add_argument(
        "--output", type=str, default="sentence_similarity_dataset.csv",
        help="Output CSV filename (default: sentence_similarity_dataset.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50000,
        help="Batch size for writing to CSV (default: 50000)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    generate_dataset(args.rows, args.output, args.batch_size)


if __name__ == "__main__":
    main()
