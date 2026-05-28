"""Canonical Smart ICT topic taxonomy + a lightweight classifier.

Folder names in the source Drive are messy, so a document's topic is derived
from its filename + original folder + a sample of its text. Shared by the
ingestion pipeline (src/ingest.py) and the re-classification script
(scripts/classify_topics.py).
"""

OTHER = "Autres"

TOPICS = [
    "Réseaux mobiles & radio",
    "Communications numériques & signal",
    "Réseaux & télécoms",
    "Systèmes, Cloud & IoT",
    "Informatique & données",
    "Maths & optimisation",
    "Management & soft skills",
    OTHER,
]

# High-confidence overrides checked first (substring in "filename folder",
# lowercased). For files the keyword scorer gets wrong on noisy content.
OVERRIDES = [
    ("matlab", "Autres"),
    ("intromatlab", "Autres"),
    ("unity", "Autres"),
    ("jeu_laby", "Autres"),
    ("mixamo", "Autres"),
    ("vr and ar", "Autres"),
    ("scada", "Autres"),
    ("smart-grid", "Autres"),
    ("smart grid", "Autres"),
    ("plate-frome", "Autres"),
    ("plateforme- energie", "Autres"),
    ("puissances_en_alternatif", "Autres"),
    ("semestres de la filiere", "Autres"),
    ("semestres de la filière", "Autres"),
    ("rapport tp", "Autres"),
    ("rapport de tp", "Autres"),
    ("exam_sig", "Autres"),
    ("smartict-dl", "Informatique & données"),
    ("ine2smartict-dl", "Informatique & données"),
    ("digital_modulations", "Communications numériques & signal"),
    ("notation_and_matrix_algebra", "Informatique & données"),
    ("cours lcs", "Informatique & données"),
    ("media de transmission", "Réseaux mobiles & radio"),
    ("média de transmission", "Réseaux mobiles & radio"),
]

# Topics in priority order (earlier wins ties). Keywords matched
# case-insensitively against "filename folder" (weight 3) and content (weight 1).
DOMAINS = [
    ("Maths & optimisation", [
        "proba", "probabilit", "statistiqu", "random process", "aleatoire",
        "recherche operationnelle", "recherche opérationnelle", "simplexe",
        "optimisation", "programmation lineaire", "programmation linéaire",
        "branch&bound", "branch and bound", "graphes", "dualit", "duaux",
        "moussaid", "souissi",
    ]),
    ("Management & soft skills", [
        "compta", "comptab", "bilan", "financ", " caf", "daf ", " cpc",
        "marotech", "management", "leadership", "pmbok", "projet", "gestion de projet",
        "conduite de changement", "cdc it", "sirh", "business", "english",
        "anglais", "cover letter", "placement test", "git", "github",
        "outils collab", "theories du management",
    ]),
    ("Communications numériques & signal", [
        "modulation", "constellation", "comnum", "communication numerique",
        "communication numérique", "codage", "theorie de l", "théorie de l",
        "quantification", "compression", "synthese des filtres", "synthèse des filtres",
        "filtre", "fourier", "transformee", "transformée", "traitement de signal",
        "traitement du signal", "decision", "estimation", "tiv", "traitement de la video",
        "traitement d'image", "ofdm", "large bande", "larges bandes", "tamtaoui",
        "benjilali", "benjillali", "gray", "sesnum", "emission", "reception",
        "constellations", "sd ",
    ]),
    ("Réseaux mobiles & radio", [
        "5g", "4g", "3g", "lte", "hsdpa", "hsupa", "mimo", "beamforming",
        "mmwave", "millimeter", "duplex", "relaying", "d2d", "cooperation",
        "antenne", "antenna", "satellite", "faisceaux", "telecom_fh", "propagation",
        "spectrum", "cognitive", "smart radio", "smart_radio", "wireless", "cellular",
        "cellulaire", "handover", "rayleigh", "fading", "goldsmith", "dahmouni",
        "erlang", "dimensionnement", "najid", "communications mobiles",
        "communication mobile", "raiss", "benmalek", "technologies 5g", "concept",
        "reseaux mobiles", "réseaux mobiles", "snr", "diversity",
    ]),
    ("Réseaux & télécoms", [
        "osi", "tcp", "ip et multimedia", "tcp-ip", "ccna", "packet tracer",
        "packet-tracer", "vlan", "ospf", "bgp", " nat", "is-is", "routing",
        "routage", "commutation", "multiplexage", "reseau", "réseau", "reseaux",
        "réseaux", "sdn", "nfv", "coeur", "core 5g", "oai", "openair", "flexran",
        "modelisation", "modélisation", "gestion de reseau", "interconnexion",
        "switch", "forouzan", "data communications", "vlsm", "mininet", "ss7",
    ]),
    ("Systèmes, Cloud & IoT", [
        "cloud", "virtu", "vmware", "docker", "linux", "systeme d", "système d",
        "exploitation", "kernel", "bash", "architecture des ordinateur",
        "assembleur", "pic18", "microcontroleur", "microcontrôleur", "iot",
        "mqtt", "arduino", "nodered", "node-red", "node red", "mosquitto",
        "live objects", "khallaayoune", "bajit", "command line",
    ]),
    ("Informatique & données", [
        "programmation c", "langage c", "language c", "polyexoc", "thecbook",
        "python", "poo", "prog web", "dev_web", "html", "css", " php", "http",
        "base de donnee", "bases de donnee", "base de données", "bases de données",
        "bdr", "algebre relationnelle", "algèbre relationnelle", "normalisation",
        "trigger", "sql", "gardarin", "elmasri", "aseds", "anwar",
        "machine learning", "deep learning", "statistical learning",
        "regression", "régression", "classification", "svm", "clustering",
        " ml", " dl ", "spark", "big data", "bigdata", "cyber", "securit",
        "sécurit", "forensic", "kali", "biometrie", "biométrie", "datahiding",
        "steganograph", "csim", "-sim-", "active directory", " cve", "trafic",
        "wireshark", "reinforcement", "supervised", "unsupervised",
    ]),
]

CONTENT_HINTS = {
    "Maths & optimisation": "probabilités statistiques variables aléatoires recherche opérationnelle simplexe optimisation programmation linéaire graphes",
    "Management & soft skills": "comptabilité bilan finance management leadership gestion de projet business plan anglais",
    "Communications numériques & signal": "modulation numérique constellation transformée de fourier filtre codage traitement du signal décision estimation ofdm",
    "Réseaux mobiles & radio": "5g 4g lte mimo antenne propagation radio mobile cellulaire fading beamforming spectre",
    "Réseaux & télécoms": "réseau osi tcp ip routage commutation multiplexage sdn nfv coeur réseau ccna",
    "Systèmes, Cloud & IoT": "cloud virtualisation linux système d'exploitation architecture ordinateur microcontrôleur iot mqtt arduino",
    "Informatique & données": "programmation c python base de données sql machine learning deep learning big data cybersécurité",
    OTHER: "scada smart grid énergie réalité virtuelle augmentée sig matlab simulation",
}


def classify_keywords(filename: str, folder: str, content: str):
    """Return (topic, score) from override + keyword matching, or (None, 0)."""
    fn = (str(filename) + " " + str(folder)).lower()
    for needle, topic in OVERRIDES:
        if needle in fn:
            return topic, 99
    ct = (content or "").lower()
    scores = {}
    for topic, keywords in DOMAINS:
        score = 0
        for kw in keywords:
            if kw in fn:
                score += 3
            elif kw in ct:
                score += 1
        if score:
            scores[topic] = score
    if not scores:
        return None, 0
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0], best[1]


_DOMAIN_NAMES = None
_DOMAIN_EMB = None


def classify(filename: str, folder: str, content: str, model=None) -> str:
    """Classify into a canonical topic. Falls back to embedding similarity
    against CONTENT_HINTS when keywords are inconclusive and `model` is given;
    otherwise returns OTHER."""
    topic, _ = classify_keywords(filename, folder, content)
    if topic:
        return topic
    if model is None:
        return OTHER

    global _DOMAIN_NAMES, _DOMAIN_EMB
    if _DOMAIN_EMB is None:
        _DOMAIN_NAMES = list(CONTENT_HINTS.keys())
        _DOMAIN_EMB = model.encode(
            [CONTENT_HINTS[d] for d in _DOMAIN_NAMES], normalize_embeddings=True
        )
    text = (content or filename or "")[:1500]
    emb = model.encode(text, normalize_embeddings=True)
    sims = _DOMAIN_EMB @ emb
    return _DOMAIN_NAMES[int(sims.argmax())]
