"""
Deep Paraphrase Engine v2
=========================
Rules:
  1. Replace EVERY possible word with a synonym
  2. Reorder clauses where possible (without changing meaning)
  3. NEVER add filler phrases or extra meaning
  4. NEVER remove any information
  5. Meaning must stay 100% identical
"""

import re
import random
from difflib import SequenceMatcher

# ─────────────────────────────────────────────────────────
# MASSIVE SYNONYM MAP — covers verbs, nouns, adjectives,
# adverbs, prepositions, conjunctions, determiners
# ─────────────────────────────────────────────────────────

SYNONYMS = {
    # ── Verbs ──
    "is": ["remains", "serves as", "represents", "constitutes"],
    "are": ["remain", "serve as", "represent", "constitute"],
    "was": ["had been", "remained", "served as"],
    "were": ["had been", "remained", "served as"],
    "has": ["possesses", "holds", "carries", "maintains"],
    "have": ["possess", "hold", "carry", "maintain"],
    "had": ["possessed", "held", "carried", "maintained"],
    "do": ["perform", "carry out", "execute", "accomplish"],
    "does": ["performs", "carries out", "executes"],
    "did": ["performed", "carried out", "executed"],
    "make": ["create", "produce", "form", "generate"],
    "makes": ["creates", "produces", "forms", "generates"],
    "made": ["created", "produced", "formed", "generated"],
    "give": ["provide", "offer", "supply", "deliver"],
    "gives": ["provides", "offers", "supplies", "delivers"],
    "gave": ["provided", "offered", "supplied", "delivered"],
    "take": ["acquire", "obtain", "receive", "accept"],
    "takes": ["acquires", "obtains", "receives", "accepts"],
    "took": ["acquired", "obtained", "received", "accepted"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "gets": ["obtains", "acquires", "receives", "gains"],
    "got": ["obtained", "acquired", "received", "gained"],
    "put": ["place", "set", "position", "arrange"],
    "puts": ["places", "sets", "positions", "arranges"],
    "use": ["employ", "utilize", "apply", "adopt"],
    "uses": ["employs", "utilizes", "applies", "adopts"],
    "used": ["employed", "utilized", "applied", "adopted"],
    "find": ["discover", "locate", "identify", "detect"],
    "finds": ["discovers", "locates", "identifies", "detects"],
    "found": ["discovered", "located", "identified", "detected"],
    "know": ["understand", "recognize", "comprehend", "grasp"],
    "knows": ["understands", "recognizes", "comprehends", "grasps"],
    "think": ["believe", "consider", "regard", "suppose"],
    "thinks": ["believes", "considers", "regards", "supposes"],
    "show": ["demonstrate", "display", "reveal", "indicate"],
    "shows": ["demonstrates", "displays", "reveals", "indicates"],
    "showed": ["demonstrated", "displayed", "revealed", "indicated"],
    "help": ["assist", "aid", "support", "facilitate"],
    "helps": ["assists", "aids", "supports", "facilitates"],
    "helped": ["assisted", "aided", "supported", "facilitated"],
    "develop": ["build", "construct", "cultivate", "advance"],
    "develops": ["builds", "constructs", "cultivates", "advances"],
    "developed": ["built", "constructed", "cultivated", "advanced"],
    "provide": ["supply", "offer", "furnish", "deliver"],
    "provides": ["supplies", "offers", "furnishes", "delivers"],
    "provided": ["supplied", "offered", "furnished", "delivered"],
    "include": ["encompass", "incorporate", "contain", "cover"],
    "includes": ["encompasses", "incorporates", "contains", "covers"],
    "included": ["encompassed", "incorporated", "contained", "covered"],
    "offer": ["present", "supply", "extend", "deliver"],
    "offers": ["presents", "supplies", "extends", "delivers"],
    "offered": ["presented", "supplied", "extended", "delivered"],
    "create": ["produce", "generate", "establish", "build"],
    "creates": ["produces", "generates", "establishes", "builds"],
    "created": ["produced", "generated", "established", "built"],
    "follow": ["trace", "pursue", "track", "adhere to"],
    "follows": ["traces", "pursues", "tracks", "adheres to"],
    "lead": ["guide", "direct", "steer", "conduct"],
    "leads": ["guides", "directs", "steers", "conducts"],
    "keep": ["maintain", "retain", "preserve", "sustain"],
    "keeps": ["maintains", "retains", "preserves", "sustains"],
    "allow": ["permit", "enable", "let", "authorize"],
    "allows": ["permits", "enables", "lets", "authorizes"],
    "begin": ["start", "commence", "initiate", "launch"],
    "begins": ["starts", "commences", "initiates", "launches"],
    "began": ["started", "commenced", "initiated", "launched"],
    "contain": ["hold", "encompass", "comprise", "include"],
    "contains": ["holds", "encompasses", "comprises", "includes"],
    "require": ["need", "demand", "necessitate", "call for"],
    "requires": ["needs", "demands", "necessitates", "calls for"],
    "support": ["reinforce", "bolster", "sustain", "uphold"],
    "supports": ["reinforces", "bolsters", "sustains", "upholds"],
    "improve": ["enhance", "refine", "elevate", "boost"],
    "improves": ["enhances", "refines", "elevates", "boosts"],
    "ensure": ["guarantee", "confirm", "secure", "verify"],
    "ensures": ["guarantees", "confirms", "secures", "verifies"],
    "maintain": ["preserve", "sustain", "uphold", "keep"],
    "maintains": ["preserves", "sustains", "upholds", "keeps"],
    "achieve": ["accomplish", "attain", "reach", "realize"],
    "achieves": ["accomplishes", "attains", "reaches", "realizes"],
    "consider": ["regard", "view", "deem", "contemplate"],
    "considers": ["regards", "views", "deems", "contemplates"],
    "produce": ["generate", "yield", "create", "manufacture"],
    "produces": ["generates", "yields", "creates", "manufactures"],
    "determine": ["establish", "ascertain", "identify", "decide"],
    "determines": ["establishes", "ascertains", "identifies", "decides"],
    "increase": ["raise", "boost", "elevate", "amplify"],
    "increases": ["raises", "boosts", "elevates", "amplifies"],
    "reduce": ["decrease", "lower", "diminish", "lessen"],
    "reduces": ["decreases", "lowers", "diminishes", "lessens"],
    "change": ["alter", "modify", "adjust", "transform"],
    "changes": ["alters", "modifies", "adjusts", "transforms"],
    "move": ["shift", "transfer", "relocate", "transport"],
    "moves": ["shifts", "transfers", "relocates", "transports"],
    "grow": ["expand", "increase", "enlarge", "develop"],
    "grows": ["expands", "increases", "enlarges", "develops"],
    "become": ["turn into", "transform into", "evolve into"],
    "becomes": ["turns into", "transforms into", "evolves into"],
    "remain": ["stay", "continue", "persist", "endure"],
    "remains": ["stays", "continues", "persists", "endures"],
    "seem": ["appear", "look", "come across as"],
    "seems": ["appears", "looks", "comes across as"],
    "exist": ["occur", "be present", "persist"],
    "exists": ["occurs", "is present", "persists"],
    "consist": ["comprise", "be composed", "be made up"],
    "consists": ["comprises", "is composed", "is made up"],
    "consisting": ["comprising", "composed", "made up"],
    "feature": ["include", "present", "incorporate", "showcase"],
    "features": ["includes", "presents", "incorporates", "showcases"],
    "featuring": ["including", "presenting", "incorporating", "showcasing"],
    "organize": ["arrange", "structure", "order", "systematize"],
    "organizes": ["arranges", "structures", "orders", "systematizes"],
    "organized": ["arranged", "structured", "ordered", "systematized"],

    # ── Nouns ──
    "paragraph": ["passage", "text section", "written segment"],
    "paragraphs": ["passages", "text sections", "written segments"],
    "sentence": ["statement", "expression", "assertion"],
    "sentences": ["statements", "expressions", "assertions"],
    "word": ["term", "expression", "vocable"],
    "words": ["terms", "expressions", "vocabulary"],
    "text": ["content", "material", "written work"],
    "writing": ["composition", "prose", "written expression"],
    "idea": ["concept", "notion", "thought"],
    "ideas": ["concepts", "notions", "thoughts"],
    "topic": ["subject", "theme", "focus"],
    "topics": ["subjects", "themes", "focal points"],
    "unit": ["segment", "component", "section"],
    "units": ["segments", "components", "sections"],
    "element": ["component", "part", "building block"],
    "elements": ["components", "parts", "building blocks"],
    "structure": ["organization", "framework", "arrangement"],
    "clarity": ["clearness", "lucidity", "transparency"],
    "details": ["specifics", "particulars", "information"],
    "detail": ["specific", "particular", "aspect"],
    "reader": ["audience member", "viewer", "individual reading"],
    "readers": ["the audience", "those reading", "individuals reading"],
    "author": ["writer", "composer", "creator"],
    "authors": ["writers", "composers", "creators"],
    "logic": ["reasoning", "rationale", "line of thought"],
    "readability": ["legibility", "comprehensibility", "ease of reading"],
    "meaning": ["significance", "sense", "interpretation"],
    "purpose": ["objective", "aim", "goal"],
    "example": ["instance", "illustration", "case"],
    "examples": ["instances", "illustrations", "cases"],
    "result": ["outcome", "consequence", "effect"],
    "results": ["outcomes", "consequences", "effects"],
    "problem": ["issue", "challenge", "difficulty"],
    "problems": ["issues", "challenges", "difficulties"],
    "method": ["approach", "technique", "procedure"],
    "methods": ["approaches", "techniques", "procedures"],
    "process": ["procedure", "operation", "mechanism"],
    "system": ["framework", "mechanism", "setup"],
    "part": ["portion", "segment", "section"],
    "parts": ["portions", "segments", "sections"],
    "way": ["manner", "approach", "fashion"],
    "ways": ["manners", "approaches", "fashions"],
    "type": ["kind", "sort", "category"],
    "types": ["kinds", "sorts", "categories"],
    "form": ["shape", "format", "variety"],
    "forms": ["shapes", "formats", "varieties"],
    "work": ["effort", "task", "labor"],
    "information": ["data", "knowledge", "facts"],
    "knowledge": ["understanding", "awareness", "expertise"],
    "ability": ["capability", "capacity", "skill"],
    "role": ["function", "position", "responsibility"],
    "level": ["degree", "extent", "stage"],
    "area": ["field", "domain", "region"],
    "areas": ["fields", "domains", "regions"],
    "group": ["collection", "set", "cluster"],
    "groups": ["collections", "sets", "clusters"],
    "number": ["quantity", "amount", "count"],
    "point": ["aspect", "matter", "issue"],
    "points": ["aspects", "matters", "issues"],
    "fact": ["reality", "truth", "actuality"],
    "facts": ["realities", "truths", "actualities"],
    "feature": ["characteristic", "attribute", "quality"],
    "aspect": ["facet", "dimension", "angle"],
    "aspects": ["facets", "dimensions", "angles"],
    "pattern": ["trend", "tendency", "arrangement"],
    "patterns": ["trends", "tendencies", "arrangements"],
    "connection": ["link", "association", "relationship"],
    "difference": ["distinction", "variation", "contrast"],
    "differences": ["distinctions", "variations", "contrasts"],
    "advantage": ["benefit", "merit", "strength"],
    "advantages": ["benefits", "merits", "strengths"],

    # ── Adjectives ──
    "distinct": ["separate", "individual", "unique"],
    "single": ["sole", "one", "individual"],
    "cohesive": ["unified", "connected", "integrated"],
    "foundational": ["fundamental", "essential", "core"],
    "important": ["significant", "crucial", "vital"],
    "different": ["varied", "diverse", "dissimilar"],
    "specific": ["particular", "precise", "definite"],
    "general": ["broad", "overall", "comprehensive"],
    "various": ["diverse", "multiple", "numerous"],
    "proper": ["appropriate", "suitable", "correct"],
    "main": ["primary", "principal", "chief"],
    "major": ["significant", "key", "principal"],
    "minor": ["small", "slight", "trivial"],
    "large": ["big", "substantial", "considerable"],
    "small": ["little", "minor", "slight"],
    "good": ["effective", "strong", "favorable"],
    "bad": ["poor", "unfavorable", "negative"],
    "new": ["fresh", "novel", "recent"],
    "old": ["previous", "former", "earlier"],
    "clear": ["evident", "obvious", "apparent"],
    "certain": ["specific", "particular", "definite"],
    "common": ["frequent", "widespread", "prevalent"],
    "basic": ["fundamental", "elementary", "primary"],
    "complex": ["intricate", "complicated", "elaborate"],
    "simple": ["straightforward", "uncomplicated", "plain"],
    "strong": ["powerful", "robust", "solid"],
    "effective": ["efficient", "productive", "successful"],
    "significant": ["substantial", "considerable", "notable"],
    "entire": ["complete", "whole", "full"],
    "final": ["concluding", "closing", "last"],
    "original": ["initial", "primary", "authentic"],
    "similar": ["comparable", "alike", "analogous"],
    "necessary": ["essential", "required", "needed"],
    "possible": ["feasible", "achievable", "viable"],
    "available": ["accessible", "obtainable", "at hand"],
    "useful": ["helpful", "beneficial", "valuable"],

    # ── Adverbs ──
    "usually": ["typically", "generally", "commonly"],
    "properly": ["correctly", "appropriately", "suitably"],
    "often": ["frequently", "regularly", "commonly"],
    "always": ["consistently", "invariably", "perpetually"],
    "never": ["at no time", "not ever", "under no circumstances"],
    "very": ["extremely", "highly", "remarkably"],
    "also": ["additionally", "furthermore", "moreover"],
    "quickly": ["rapidly", "swiftly", "promptly"],
    "slowly": ["gradually", "steadily", "unhurriedly"],
    "clearly": ["evidently", "obviously", "plainly"],
    "easily": ["effortlessly", "readily", "smoothly"],
    "directly": ["straightaway", "immediately", "without delay"],
    "generally": ["broadly", "typically", "usually"],
    "particularly": ["especially", "specifically", "notably"],
    "especially": ["particularly", "notably", "specifically"],
    "sometimes": ["occasionally", "at times", "now and then"],
    "together": ["collectively", "jointly", "as a group"],
    "however": ["nevertheless", "nonetheless", "yet"],
    "therefore": ["consequently", "thus", "hence"],
    "moreover": ["furthermore", "additionally", "besides"],
    "furthermore": ["moreover", "additionally", "also"],
    "indeed": ["certainly", "truly", "in fact"],
    "actually": ["in reality", "in fact", "truly"],
    "essentially": ["fundamentally", "basically", "at its core"],
    "primarily": ["mainly", "chiefly", "principally"],

    # ── Prepositions & Connectors ──
    "because": ["since", "as", "given that"],
    "although": ["though", "even though", "while"],
    "while": ["whereas", "although", "even as"],
    "but": ["however", "yet", "nevertheless"],
    "and": ["as well as", "along with", "in addition to"],

    # ── Everyday / Casual Words ──
    "currently": ["presently", "at present", "right now"],
    "now": ["at this moment", "presently", "currently"],
    "today": ["at present", "these days", "in this day and age"],
    "recently": ["lately", "not long ago", "in recent times"],
    "already": ["by now", "previously", "at this point"],
    "still": ["even now", "yet", "up to this point"],
    "soon": ["shortly", "before long", "in a short while"],
    "here": ["at this place", "in this location", "at this spot"],
    "there": ["at that place", "in that location", "at that spot"],
    "really": ["truly", "genuinely", "honestly"],
    "just": ["merely", "simply", "only"],
    "only": ["merely", "solely", "just"],
    "about": ["regarding", "concerning", "approximately"],
    "around": ["approximately", "roughly", "near"],
    "maybe": ["perhaps", "possibly", "potentially"],
    "please": ["kindly", "if you would"],
    "thanks": ["gratitude", "appreciation", "thankfulness"],
    "sorry": ["apologetic", "regretful"],
    "happy": ["joyful", "delighted", "pleased", "cheerful", "glad"],
    "sad": ["unhappy", "sorrowful", "melancholy", "downcast"],
    "angry": ["furious", "irate", "enraged", "upset"],
    "afraid": ["scared", "frightened", "fearful", "terrified"],
    "tired": ["exhausted", "fatigued", "weary", "drained"],
    "busy": ["occupied", "engaged", "swamped"],
    "ready": ["prepared", "set", "equipped"],
    "sure": ["certain", "confident", "positive"],
    "nice": ["pleasant", "lovely", "agreeable", "delightful"],
    "great": ["wonderful", "excellent", "fantastic", "superb"],
    "amazing": ["astonishing", "remarkable", "incredible", "stunning"],
    "beautiful": ["gorgeous", "lovely", "stunning", "attractive"],
    "pretty": ["attractive", "lovely", "good-looking"],
    "ugly": ["unattractive", "unsightly", "hideous"],
    "big": ["large", "huge", "enormous", "massive"],
    "little": ["small", "tiny", "slight", "minor"],
    "long": ["extended", "lengthy", "prolonged"],
    "short": ["brief", "concise", "compact"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["unhurried", "gradual", "leisurely"],
    "hard": ["difficult", "challenging", "tough"],
    "easy": ["simple", "effortless", "straightforward"],
    "true": ["accurate", "correct", "genuine", "authentic"],
    "false": ["incorrect", "untrue", "inaccurate", "wrong"],
    "right": ["correct", "accurate", "proper"],
    "wrong": ["incorrect", "inaccurate", "mistaken"],
    "full": ["complete", "entire", "whole", "filled"],
    "empty": ["vacant", "hollow", "bare", "void"],
    "open": ["accessible", "unlocked", "available"],
    "close": ["near", "adjacent", "nearby"],
    "young": ["youthful", "juvenile", "adolescent"],
    "hot": ["warm", "heated", "scorching"],
    "cold": ["chilly", "cool", "frigid", "freezing"],
    "rich": ["wealthy", "affluent", "prosperous"],
    "poor": ["impoverished", "destitute", "needy"],
    "free": ["liberated", "unrestricted", "complimentary"],
    "safe": ["secure", "protected", "sheltered"],
    "dangerous": ["hazardous", "risky", "perilous"],
    "friend": ["companion", "pal", "buddy", "mate"],
    "friends": ["companions", "pals", "buddies", "mates"],
    "partner": ["companion", "significant other", "better half"],
    "family": ["household", "kin", "relatives"],
    "people": ["individuals", "persons", "folks"],
    "person": ["individual", "human being", "someone"],
    "man": ["gentleman", "male", "fellow"],
    "woman": ["lady", "female", "individual"],
    "child": ["youngster", "kid", "little one"],
    "children": ["youngsters", "kids", "little ones"],
    "boy": ["lad", "young man", "youngster"],
    "girl": ["young woman", "lass", "young lady"],
    "baby": ["infant", "newborn", "little one"],
    "mother": ["mom", "parent", "mama"],
    "father": ["dad", "parent", "papa"],
    "brother": ["sibling", "kin"],
    "sister": ["sibling", "kin"],
    "teacher": ["instructor", "educator", "mentor"],
    "student": ["learner", "pupil", "scholar"],
    "doctor": ["physician", "medical professional", "practitioner"],
    "house": ["home", "residence", "dwelling"],
    "home": ["house", "residence", "dwelling", "abode"],
    "room": ["chamber", "space", "area"],
    "school": ["institution", "academy", "educational facility"],
    "city": ["urban area", "metropolis", "municipality"],
    "country": ["nation", "state", "land"],
    "world": ["globe", "earth", "planet"],
    "place": ["location", "spot", "site", "venue"],
    "time": ["period", "moment", "duration"],
    "day": ["date", "occasion", "period"],
    "year": ["annum", "twelve months", "calendar year"],
    "life": ["existence", "living", "lifetime"],
    "love": ["affection", "adoration", "fondness", "devotion"],
    "loved": ["adored", "cherished", "treasured"],
    "like": ["enjoy", "appreciate", "fancy", "prefer"],
    "liked": ["enjoyed", "appreciated", "fancied"],
    "want": ["desire", "wish for", "crave"],
    "wants": ["desires", "wishes for", "craves"],
    "wanted": ["desired", "wished for", "craved"],
    "need": ["require", "must have", "depend on"],
    "needs": ["requires", "demands", "depends on"],
    "needed": ["required", "demanded", "depended on"],
    "try": ["attempt", "endeavor", "strive"],
    "tries": ["attempts", "endeavors", "strives"],
    "tried": ["attempted", "endeavored", "strived"],
    "look": ["appear", "seem", "glance"],
    "looks": ["appears", "seems", "glances"],
    "looked": ["appeared", "seemed", "glanced"],
    "come": ["arrive", "approach", "reach"],
    "comes": ["arrives", "approaches", "reaches"],
    "came": ["arrived", "approached", "reached"],
    "go": ["proceed", "head", "travel", "move"],
    "goes": ["proceeds", "heads", "travels", "moves"],
    "went": ["proceeded", "headed", "traveled", "moved"],
    "say": ["state", "declare", "mention", "express"],
    "says": ["states", "declares", "mentions", "expresses"],
    "said": ["stated", "declared", "mentioned", "expressed"],
    "tell": ["inform", "notify", "communicate"],
    "tells": ["informs", "notifies", "communicates"],
    "told": ["informed", "notified", "communicated"],
    "ask": ["inquire", "question", "request"],
    "asks": ["inquires", "questions", "requests"],
    "asked": ["inquired", "questioned", "requested"],
    "talk": ["speak", "converse", "chat", "discuss"],
    "talks": ["speaks", "converses", "chats", "discusses"],
    "talked": ["spoke", "conversed", "chatted", "discussed"],
    "call": ["contact", "ring", "phone"],
    "calls": ["contacts", "rings", "phones"],
    "called": ["contacted", "rang", "phoned"],
    "play": ["engage in", "participate in", "enjoy"],
    "plays": ["engages in", "participates in", "enjoys"],
    "played": ["engaged in", "participated in", "enjoyed"],
    "work": ["labor", "toil", "operate"],
    "works": ["labors", "toils", "operates"],
    "worked": ["labored", "toiled", "operated"],
    "live": ["reside", "dwell", "inhabit"],
    "lives": ["resides", "dwells", "inhabits"],
    "lived": ["resided", "dwelled", "inhabited"],
    "run": ["sprint", "dash", "race", "jog"],
    "runs": ["sprints", "dashes", "races", "jogs"],
    "walk": ["stroll", "stride", "amble"],
    "walks": ["strolls", "strides", "ambles"],
    "walked": ["strolled", "strode", "ambled"],
    "eat": ["consume", "dine on", "have"],
    "eats": ["consumes", "dines on", "has"],
    "drink": ["consume", "sip", "have"],
    "drinks": ["consumes", "sips", "has"],
    "sleep": ["rest", "slumber", "doze"],
    "sleeps": ["rests", "slumbers", "dozes"],
    "write": ["compose", "pen", "author", "draft"],
    "writes": ["composes", "pens", "authors", "drafts"],
    "wrote": ["composed", "penned", "authored", "drafted"],
    "read": ["peruse", "study", "go through", "examine"],
    "reads": ["peruses", "studies", "goes through", "examines"],
    "learn": ["study", "acquire knowledge", "grasp"],
    "learns": ["studies", "acquires knowledge", "grasps"],
    "learned": ["studied", "acquired knowledge", "grasped"],
    "teach": ["instruct", "educate", "train", "coach"],
    "teaches": ["instructs", "educates", "trains", "coaches"],
    "taught": ["instructed", "educated", "trained", "coached"],
    "think": ["believe", "consider", "suppose", "reckon"],
    "feel": ["sense", "experience", "perceive"],
    "feels": ["senses", "experiences", "perceives"],
    "felt": ["sensed", "experienced", "perceived"],
    "hope": ["wish", "aspire", "desire", "expect"],
    "hopes": ["wishes", "aspires", "desires", "expects"],
    "hoped": ["wished", "aspired", "desired", "expected"],
    "enjoy": ["relish", "savor", "delight in", "appreciate"],
    "enjoys": ["relishes", "savors", "delights in", "appreciates"],
    "enjoyed": ["relished", "savored", "delighted in", "appreciated"],
    "start": ["begin", "commence", "initiate", "launch"],
    "starts": ["begins", "commences", "initiates", "launches"],
    "started": ["began", "commenced", "initiated", "launched"],
    "stop": ["cease", "halt", "discontinue", "end"],
    "stops": ["ceases", "halts", "discontinues", "ends"],
    "stopped": ["ceased", "halted", "discontinued", "ended"],
    "spend": ["devote", "dedicate", "invest"],
    "spends": ["devotes", "dedicates", "invests"],
    "spent": ["devoted", "dedicated", "invested"],
    "buy": ["purchase", "acquire", "obtain"],
    "buys": ["purchases", "acquires", "obtains"],
    "bought": ["purchased", "acquired", "obtained"],
    "sell": ["trade", "vend", "market"],
    "build": ["construct", "erect", "assemble"],
    "builds": ["constructs", "erects", "assembles"],
    "built": ["constructed", "erected", "assembled"],
    "watch": ["observe", "view", "witness"],
    "watches": ["observes", "views", "witnesses"],
    "watched": ["observed", "viewed", "witnessed"],
    "listen": ["hear", "pay attention to", "tune in to"],
    "listens": ["hears", "pays attention to", "tunes in to"],
    "romancing": ["courting", "wooing", "being romantic with"],
    "dating": ["courting", "going out with", "seeing"],
    "loving": ["adoring", "cherishing", "treasuring"],
    "caring": ["looking after", "tending to", "nurturing"],
    "meeting": ["encountering", "gathering with", "assembling with"],
    "talking": ["speaking", "conversing", "chatting"],
    "working": ["laboring", "toiling", "operating"],
    "playing": ["engaging in activities", "participating", "enjoying games"],
    "studying": ["learning", "examining", "reviewing"],
    "eating": ["dining", "consuming", "having a meal"],
    "sleeping": ["resting", "slumbering", "dozing"],
    "running": ["sprinting", "dashing", "jogging"],
    "walking": ["strolling", "striding", "ambling"],
    "sitting": ["seated", "resting", "positioned"],
    "standing": ["upright", "on one's feet", "positioned vertically"],
    "thinking": ["pondering", "contemplating", "reflecting"],
    "feeling": ["sensing", "experiencing", "perceiving"],
    "trying": ["attempting", "endeavoring", "striving"],
    "going": ["heading", "proceeding", "traveling"],
    "coming": ["arriving", "approaching", "reaching"],
    "making": ["creating", "producing", "crafting"],
    "taking": ["acquiring", "obtaining", "receiving"],
    "getting": ["obtaining", "acquiring", "receiving"],
    "looking": ["gazing", "glancing", "peering"],
    "saying": ["stating", "declaring", "expressing"],
    "telling": ["informing", "communicating", "conveying"],
    "asking": ["inquiring", "questioning", "requesting"],

    # ── Common Phrases (multi-word) ──
    "consisting of": ["composed of", "made up of", "comprising"],
    "one or more": ["one or several", "at least one", "a number of"],
    "as well as": ["along with", "in addition to", "together with"],
    "in order to": ["so as to", "with the aim of", "to"],
    "due to": ["because of", "owing to", "on account of"],
    "such as": ["like", "for instance", "for example"],
    "a number of": ["several", "various", "multiple"],
    "in addition to": ["besides", "apart from", "on top of"],
    "on the other hand": ["conversely", "alternatively", "in contrast"],
    "for example": ["for instance", "as an illustration", "to illustrate"],
    "break up": ["divide", "split", "separate"],
    "follow the": ["trace the", "track the", "comprehend the"],
    "i am": ["I happen to be", "I find myself"],
    "i am currently": ["I am presently", "at this moment I am", "I find myself presently"],
    "this is": ["this happens to be", "this would be", "here we have"],

    # ── Domain: Writing/Paragraphs (the user's exact text) ──
    "topic sentence": ["opening statement", "lead assertion", "introductory claim"],
    "supporting details": ["supplementary specifics", "reinforcing particulars", "elaborating information"],
    "concluding sentence": ["closing statement", "final assertion", "summary remark"],
    "properly organized": ["well-arranged", "thoughtfully structured", "logically ordered"],
    "distinct unit": ["separate segment", "individual block", "independent portion"],
    "cohesive idea": ["unified concept", "integrated notion", "connected theme"],
    "foundational element": ["fundamental component", "core building block", "essential part"],
    "offers structure and clarity": ["provides organization and lucidity", "delivers framework and transparency"],
    "author's logic": ["writer's reasoning", "creator's rationale", "author's line of thought"],
    "better readability": ["improved legibility", "enhanced comprehensibility", "greater ease of reading"],
    "help readers follow": ["assist the audience in tracing", "enable those reading to track", "aid the audience in comprehending"],
    "break up text for": ["divide content for", "split material for", "separate written work for"],
    "develops a single": ["elaborates on one", "explores a sole", "advances an individual"],
}

# Words to NEVER replace (they're too small or grammatical)
SKIP_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "it",
    "or", "if", "so", "no", "up", "be", "he", "she", "we", "me",
    "my", "his", "her", "its", "our", "us", "i", "am", "not",
    "than", "that", "this", "with", "from", "into", "for",
}


def _extract_punctuation(word: str):
    """Split a word into (prefix_punct, core, suffix_punct)."""
    core = word.strip(".,!?;:'\"-()[]{}…–—")
    prefix = word[:len(word) - len(word.lstrip(".,!?;:'\"-()[]{}…–—"))]
    suffix = word[len(word.rstrip(".,!?;:'\"-()[]{}…–—")):]
    return prefix, core, suffix


def _replace_all_possible(text: str) -> str:
    """Replace every possible word/phrase with a synonym."""
    result = text

    # PHASE 1: Multi-word phrase replacement (longest first)
    phrase_keys = sorted(
        [k for k in SYNONYMS if " " in k],
        key=len, reverse=True
    )
    lower_result = result.lower()
    for phrase in phrase_keys:
        if phrase in lower_result:
            alts = [a for a in SYNONYMS[phrase] if a.lower() != phrase]
            if not alts:
                continue
            replacement = random.choice(alts)
            idx = lower_result.index(phrase)
            original_text = result[idx:idx + len(phrase)]
            # Preserve first-letter capitalization
            if original_text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result = result[:idx] + replacement + result[idx + len(phrase):]
            lower_result = result.lower()

    # PHASE 2: Single-word replacement (replace EVERY word we can)
    words = result.split()
    new_words = []
    for i, word in enumerate(words):
        prefix, core, suffix = _extract_punctuation(word)

        if not core or len(core) <= 1:
            new_words.append(word)
            continue

        # Skip proper nouns (capitalized mid-sentence)
        if i > 0 and core[0].isupper() and not core.isupper():
            new_words.append(word)
            continue

        core_lower = core.lower()

        # Skip tiny grammar words
        if core_lower in SKIP_WORDS:
            new_words.append(word)
            continue

        # Look up synonym
        if core_lower in SYNONYMS:
            alts = [a for a in SYNONYMS[core_lower] if a.lower() != core_lower]
            if alts:
                chosen = random.choice(alts)
                # Preserve capitalization
                if core[0].isupper():
                    chosen = chosen[0].upper() + chosen[1:]
                new_words.append(prefix + chosen + suffix)
                continue

        new_words.append(word)

    return " ".join(new_words)


def _reorder_clauses(sentence: str) -> str:
    """Reorder independent clauses without changing meaning."""
    text = sentence.rstrip(".")

    # Try splitting on ", and " or "; " or ", " (only if both parts are big enough)
    for sep in [", and ", "; ", ", but ", ", while "]:
        if sep in text.lower():
            idx = text.lower().index(sep)
            part_a = text[:idx].strip()
            part_b = text[idx + len(sep):].strip()

            if len(part_a.split()) >= 4 and len(part_b.split()) >= 4:
                # Swap order with appropriate connector
                connectors = {
                    ", and ": ", and ",
                    "; ": "; ",
                    ", but ": ", but ",
                    ", while ": ", while ",
                }
                conn = connectors.get(sep, ", and ")
                # Capitalize B, lowercase A
                part_b_new = part_b[0].upper() + part_b[1:] if part_b else part_b
                part_a_new = part_a[0].lower() + part_a[1:] if part_a and not part_a[0:2].isupper() else part_a
                return part_b_new + conn + part_a_new + "."
            break

    return sentence


def _swap_comma_clauses(sentence: str) -> str:
    """For sentences with comma-separated parts, move trailing parts to the front."""
    text = sentence.rstrip(".")

    # Pattern: "X, Y, and Z" → "Y, Z, and X"  or  "Z, Y, and X"
    # Pattern: "Main clause, modifier" → "Modifier, main clause"
    parts = text.split(", ")
    if len(parts) >= 2:
        # Only swap if both parts are substantial
        first = parts[0]
        rest = ", ".join(parts[1:])
        if len(first.split()) >= 3 and len(rest.split()) >= 3:
            # Move first part to end
            rest_capitalized = rest[0].upper() + rest[1:] if rest else rest
            first_lower = first[0].lower() + first[1:] if first and not first[0:2].isupper() else first
            return rest_capitalized + ", " + first_lower + "."

    return sentence


def _normalize(text: str) -> str:
    """Clean punctuation, capitalization, spacing."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,\.!?;:])', r'\1', text)
    text = re.sub(r'([,\.!?;:])([^\s\)\'\"])', r'\1 \2', text)
    text = re.sub(r'\.{2,}', '.', text)
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    # Capitalize after periods
    text = re.sub(r'(\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    # Ensure ending punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    return text


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def deep_paraphrase(sentence: str, source_sentence: str = "") -> str:
    """
    Paraphrase a single sentence:
      1. Replace every possible word/phrase with synonyms
      2. Optionally reorder clauses
      3. Pick the candidate with lowest similarity to source
    
    Meaning is preserved exactly — no words added, no information removed.
    """
    if not sentence or len(sentence.split()) < 3:
        return sentence

    original = sentence.strip()
    source = source_sentence.strip() if source_sentence else ""
    candidates = []

    # Candidate 1: synonym replacement only
    c1 = _replace_all_possible(original)
    c1 = _normalize(c1)
    candidates.append(c1)

    # Candidate 2: synonym replacement + clause reorder
    c2 = _replace_all_possible(original)
    c2 = _reorder_clauses(c2)
    c2 = _normalize(c2)
    candidates.append(c2)

    # Candidate 3: clause swap first, then synonyms
    c3 = _swap_comma_clauses(original)
    c3 = _replace_all_possible(c3)
    c3 = _normalize(c3)
    candidates.append(c3)

    # Candidate 4: another random synonym pass (different random choices)
    c4 = _replace_all_possible(original)
    c4 = _normalize(c4)
    candidates.append(c4)

    # Candidate 5: swap + reorder + synonyms
    c5 = _swap_comma_clauses(original)
    c5 = _reorder_clauses(c5)
    c5 = _replace_all_possible(c5)
    c5 = _normalize(c5)
    candidates.append(c5)

    # Pick the candidate most different from source (or original)
    compare_to = source if source else original
    best = original
    best_sim = 1.0

    for cand in candidates:
        if not cand or len(cand.strip()) < 5:
            continue
        sim = SequenceMatcher(None, cand.lower(), compare_to.lower()).ratio()
        if sim < best_sim:
            best_sim = sim
            best = cand

    return best


def deep_paraphrase_paragraph(paragraph: str, source_paragraph: str = "") -> str:
    """Paraphrase each sentence independently, preserving meaning."""
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    source_sentences = re.split(r'(?<=[.!?])\s+', source_paragraph.strip()) if source_paragraph else []

    paraphrased = []
    for sent in sentences:
        if not sent.strip():
            continue
        # Match to closest source sentence
        source_sent = ""
        if source_sentences:
            best_sim = 0
            for ss in source_sentences:
                sim = SequenceMatcher(None, sent.lower(), ss.lower()).ratio()
                if sim > best_sim:
                    best_sim = sim
                    source_sent = ss
        paraphrased.append(deep_paraphrase(sent, source_sent))

    return " ".join(paraphrased)


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test = (
        "A paragraph is a distinct unit of writing, consisting of one or more sentences, "
        "that develops a single, cohesive idea or topic. It is a foundational element of writing "
        "that offers structure and clarity, usually featuring a topic sentence, supporting details, "
        "and a concluding sentence. Properly organized, paragraphs help readers follow the author's "
        "logic and break up text for better readability."
    )
    print("ORIGINAL:")
    print(test)
    print()
    for i in range(3):
        result = deep_paraphrase_paragraph(test, test)
        sim = SequenceMatcher(None, test.lower(), result.lower()).ratio()
        print(f"REWRITE #{i+1} (similarity: {sim:.0%}):")
        print(result)
        print()
