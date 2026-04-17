"""
skill_extractor.py (Production v5.0 - BTech All Roles)
-------------------------------------------------------
Advanced Resume Skill Extractor:
- Covers ALL common BTech career roles
- Regex-based extraction (word-boundary safe)
- Role-wise organized skill database
- Smart normalization + inferred skills
- Clean & deduplicated output
"""

import re


# ─────────────────────────────────────────────
# 🔹 SKILLS DATABASE — BTech ALL ROLES
# ─────────────────────────────────────────────

SKILLS_LIST = [

    # ── 1. SOFTWARE DEVELOPMENT ───────────────
    "python", "java", "javascript", "typescript", "c", "c++", "c#",
    "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "r", "matlab",
    "html", "css", "react", "react.js", "next.js", "angular", "vue",
    "redux", "tailwind css", "framer motion",
    "node.js", "nodejs", "express", "express.js",
    "django", "flask", "fastapi", "spring boot",
    "rest api", "rest apis", "websockets", "graphql",
    "jwt", "oauth", "authentication", "authorization",
    "mvc", "state management", "api design",
    "prisma", "mongoose",
    "vercel", "netlify",
    "frontend development", "backend development", "full stack", "web development",

    # ── 2. DATA SCIENCE / ML / AI ─────────────
    "machine learning", "deep learning", "data science", "data analysis",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "scipy",
    "nlp", "natural language processing", "computer vision",
    "opencv", "hugging face", "llm", "generative ai", "prompt engineering",
    "feature engineering", "model deployment", "mlops",
    "regression", "classification", "clustering", "neural network",
    "random forest", "xgboost", "time series",
    "artificial intelligence", "chatgpt", "copilot",

    # ── 3. DATA ENGINEERING / ANALYTICS ───────
    "sql", "mysql", "postgresql", "mongodb", "redis", "firebase",
    "oracle", "sqlite", "dynamodb", "cassandra",
    "power bi", "tableau", "excel", "google sheets",
    "etl", "data pipeline", "apache spark", "hadoop", "kafka",
    "airflow", "dbt", "snowflake", "bigquery", "data warehouse",
    "data visualization", "business intelligence",

    # ── 4. DEVOPS / CLOUD ─────────────────────
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
    "terraform", "ansible", "ci/cd", "linux", "bash", "shell scripting",
    "git", "github", "gitlab", "bitbucket",
    "nginx", "apache", "load balancing", "microservices",
    "serverless", "lambda", "cloud computing",

    # ── 5. CYBERSECURITY ──────────────────────
    "network security", "ethical hacking", "penetration testing",
    "kali linux", "metasploit", "wireshark", "nmap", "burp suite",
    "cryptography", "ssl", "tls", "firewall", "ids", "ips",
    "siem", "soc", "vulnerability assessment", "owasp",
    "digital forensics", "malware analysis", "incident response",
    "information security", "cyber security",

    # ── 6. EMBEDDED SYSTEMS / IOT ─────────────
    "embedded c", "arduino", "raspberry pi", "stm32", "esp32",
    "rtos", "freertos", "uart", "spi", "i2c", "can bus",
    "iot", "mqtt", "zigbee", "bluetooth", "wifi module",
    "pcb design", "kicad", "altium", "proteus", "multisim",
    "microcontroller", "microprocessor", "fpga", "vhdl", "verilog",
    "signal processing", "sensor integration",

    # ── 7. VLSI / CHIP DESIGN ─────────────────
    "vlsi", "verilog", "vhdl", "system verilog",
    "cadence", "synopsys", "mentor graphics",
    "rtl design", "synthesis", "sta", "place and route",
    "digital design", "analog design", "asic", "soc design",
    "dft", "timing analysis", "power analysis",

    # ── 8. MECHANICAL ENGINEERING ─────────────
    "autocad", "solidworks", "catia", "ansys", "creo",
    "fusion 360", "3d printing", "additive manufacturing",
    "cad", "cam", "cfd", "fea", "finite element analysis",
    "manufacturing processes", "cnc machining",
    "thermodynamics", "fluid mechanics", "heat transfer",
    "robotics", "automation", "plc", "scada", "hmi",
    "lean manufacturing", "six sigma", "quality control",

    # ── 9. CIVIL ENGINEERING ──────────────────
    "autocad civil", "staad pro", "etabs", "revit", "primavera",
    "ms project", "civil 3d",
    "structural analysis", "structural design", "rcc design",
    "surveying", "gis", "remote sensing", "arcgis",
    "construction management", "project planning",
    "soil mechanics", "geotechnical", "foundation design",
    "highway design", "transportation engineering",
    "water supply", "sanitation", "irrigation",

    # ── 10. ELECTRICAL ENGINEERING ────────────
    "power systems", "power electronics", "circuit design",
    "matlab simulink", "pspice", "ltspice", "labview",
    "electric vehicles", "battery management system",
    "solar energy", "wind energy", "renewable energy",
    "transformer", "motor drives", "inverter", "rectifier",
    "control systems", "pid controller",
    "high voltage", "switchgear", "protection relay",
    "smart grid", "energy audit",

    # ── 11. ELECTRONICS & COMMUNICATION ───────
    "signal processing", "dsp", "image processing",
    "communication systems", "wireless communication",
    "5g", "lte", "antenna design", "rf design",
    "hfss", "cst", "ads",
    "optical fiber", "photonics",
    "digital electronics", "analog electronics",
    "oscilloscope", "logic analyzer",

    # ── 12. ROBOTICS / AUTOMATION ─────────────
    "ros", "ros2", "gazebo", "slam", "path planning",
    "sensor fusion", "robotic arm", "drone", "uav",
    "autonomous systems", "control theory", "motion planning",
    "industrial automation", "plc programming",

    # ── 13. PRODUCT MANAGEMENT (Tech) ─────────
    "product roadmap", "user stories", "agile", "scrum", "kanban",
    "jira", "confluence", "notion", "trello",
    "wireframing", "prototyping", "figma", "balsamiq",
    "market research", "competitive analysis",
    "a/b testing", "product analytics", "kpi tracking",
    "stakeholder management", "go to market",

    # ── 14. UI/UX DESIGN ──────────────────────
    "figma", "adobe xd", "sketch", "invision",
    "photoshop", "illustrator", "after effects",
    "user research", "usability testing",
    "design thinking", "information architecture",
    "interaction design", "visual design", "typography",
    "color theory", "responsive design", "accessibility",

    # ── 15. BUSINESS ANALYST / CONSULTING ─────
    "requirement gathering", "brd", "frd", "use case",
    "uml", "flowchart", "process mapping", "gap analysis",
    "business analysis", "functional testing", "uat",

    # ── 16. TESTING / QA ──────────────────────
    "manual testing", "automation testing", "selenium",
    "cypress", "playwright", "jest", "pytest",
    "test cases", "test plan", "bug tracking",
    "api testing", "postman", "jmeter", "load testing",
    "performance testing", "regression testing",
    "black box testing", "white box testing",

    # ── 17. GAME DEVELOPMENT ──────────────────
    "unity", "unreal engine", "godot",
    "game design", "level design", "3d modeling",
    "blender", "maya", "3ds max",
    "ar", "vr", "mixed reality", "xr",
    "physics simulation", "shader programming",

    # ── 18. BLOCKCHAIN ────────────────────────
    "blockchain", "solidity", "ethereum", "web3.js", "ethers.js",
    "smart contracts", "nft", "defi", "hyperledger",
    "cryptocurrency", "metamask", "truffle", "hardhat",
    "ipfs", "decentralized applications", "dapps",

    # ── 19. SOFT SKILLS ───────────────────────
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "critical thinking",
    "time management", "presentation", "collaboration",
    "analytical thinking", "attention to detail",

    # ── 20. MOBILE DEVELOPMENT ────────────────
    "react native", "flutter", "dart", "swiftui", "jetpack compose",
    "android development", "ios development", "mobile development",
    "xcode", "android studio", "expo",

    # ── 21. DATA / MODERN TOOLS ───────────────
    "streamlit", "gradio", "langchain", "llamaindex", "pinecone",
    "chromadb", "weaviate", "vector database", "rag",
    "data lake", "lakehouse", "delta lake", "databricks",
    "looker", "metabase", "superset",

    # ── 22. CLOUD CERTIFICATIONS & SERVICES ───
    "aws lambda", "s3", "ec2", "ecs", "eks", "cloudfront",
    "azure devops", "azure functions", "cosmos db",
    "google cloud functions", "cloud run", "vertex ai",
    "heroku", "railway", "render", "fly.io",

    # ── 23. API & ARCHITECTURE ────────────────
    "grpc", "soap", "swagger", "openapi",
    "event driven", "message queue", "rabbitmq",
    "design patterns", "solid principles", "clean architecture",
    "domain driven design", "system design",

    # ── 24. VERSION CONTROL & CI/CD ──────────
    "github actions", "circleci", "travis ci", "argo cd",
    "helm", "prometheus", "grafana", "elk stack",
    "datadog", "new relic", "splunk",
]


# ─────────────────────────────────────────────
# 🔥 SKILL NORMALIZATION MAP
# ─────────────────────────────────────────────

SKILL_MAP = {
    # Variations
    "react":      ["react.js"],
    "node.js":    ["nodejs"],
    "express":    ["express.js"],
    "rest api":   ["rest apis"],
    "html":       ["html5"],
    "css":        ["css3"],

    # Software Dev
    "full stack":              ["mern", "fullstack", "mean"],
    "authentication":          ["jwt", "oauth", "auth"],
    "frontend development":    ["react", "next.js", "angular", "vue", "html", "css"],
    "backend development":     ["node.js", "express", "django", "flask", "fastapi", "spring boot"],
    "version control":         ["git", "github", "gitlab"],
    "database management":     ["sql", "mysql", "postgresql", "mongodb", "redis", "firebase"],

    # AI / ML
    "deep learning":           ["tensorflow", "keras", "pytorch"],
    "data science":            ["pandas", "numpy", "data analysis", "machine learning"],
    "nlp":                     ["llm", "natural language processing", "text classification", "tokenization", "hugging face"],
    "machine learning":        ["ml", "scikit-learn", "model training", "regression", "classification", "xgboost"],
    "artificial intelligence": ["ai", "neural network", "deep learning", "machine learning", "generative ai"],
    "computer vision":         ["image recognition", "object detection", "opencv", "cnn"],
    "mlops":                   ["model deployment", "mlflow", "kubeflow"],
    "seaborn":                 ["matplotlib"],
    "classification":          ["classify", "classification", "predict", "model training"],

    # DevOps / Cloud
    "devops":                  ["ci/cd", "docker", "kubernetes", "jenkins", "terraform", "ansible"],
    "cloud computing":         ["aws", "azure", "gcp", "serverless", "lambda"],

    # Security
    "cyber security":          ["ethical hacking", "penetration testing", "network security", "owasp"],
    "ethical hacking":         ["kali linux", "metasploit", "burp suite", "nmap"],

    # Embedded / IoT
    "iot":                     ["mqtt", "arduino", "raspberry pi", "esp32", "zigbee"],
    "embedded systems":        ["embedded c", "rtos", "microcontroller", "stm32", "esp32"],

    # VLSI
    "vlsi":                    ["verilog", "vhdl", "system verilog", "rtl design", "asic"],

    # Mechanical
    "cad":                     ["autocad", "solidworks", "catia", "creo", "fusion 360"],
    "simulation":              ["ansys", "fea", "cfd", "matlab simulink"],
    "automation":              ["plc", "scada", "hmi", "industrial automation"],
    "lean manufacturing":      ["six sigma", "quality control", "kaizen"],

    # Civil
    "structural design":       ["staad pro", "etabs", "rcc design"],
    "gis":                     ["arcgis", "remote sensing", "civil 3d"],

    # Electrical
    "power electronics":       ["inverter", "rectifier", "motor drives", "battery management system"],
    "renewable energy":        ["solar energy", "wind energy", "electric vehicles"],
    "control systems":         ["pid controller", "matlab simulink", "labview"],

    # Robotics
    "robotics":                ["ros", "ros2", "slam", "path planning", "robotic arm", "drone"],

    # Testing
    "automation testing":      ["selenium", "cypress", "playwright", "jest", "pytest"],
    "api testing":             ["postman", "jmeter"],

    # Game Dev
    "game development":        ["unity", "unreal engine", "godot", "game design"],
    "3d modeling":             ["blender", "maya", "3ds max"],
    "vr":                      ["virtual reality", "oculus", "steamvr"],
    "ar":                      ["augmented reality", "arkit", "arcore"],

    # Blockchain
    "blockchain":              ["solidity", "ethereum", "smart contracts", "web3.js", "dapps"],

    # Data / Analytics
    "business intelligence":   ["power bi", "tableau", "data visualization"],
    "data pipeline":           ["apache spark", "kafka", "airflow", "etl"],
    "data warehouse":          ["snowflake", "bigquery", "redshift"],

    # Mobile
    "mobile development":      ["react native", "flutter", "android development", "ios development"],
    "android development":     ["kotlin", "jetpack compose", "android studio"],
    "ios development":         ["swift", "swiftui", "xcode"],

    # Modern AI/Data
    "rag":                     ["langchain", "llamaindex", "vector database", "chromadb", "pinecone"],
    "generative ai":           ["llm", "chatgpt", "copilot", "prompt engineering", "rag", "langchain"],

    # Product / Design
    "agile":                   ["scrum", "kanban", "jira", "sprint"],
    "ui/ux":                   ["figma", "adobe xd", "wireframing", "prototyping", "user research"],

     # Soft skills
    "collaboration": ["collaborated", "teamwork", "team", "worked with"],

     # Feature engineering
    "feature engineering": [
        "data preprocessing",
        "data cleaning",
        "feature extraction",
        "data transformation"
    ],

    # Teamwork
    "teamwork": [
        "team",
        "collaborated",
        "community",
        "worked with",
        "coordinated"
    ],

    # Analytical thinking
    "analytical thinking": [
        "data analysis",
        "problem solving",
        "analysis",
        "model training"
    ]
}


# ─────────────────────────────────────────────
# 🔹 EDUCATION KEYWORDS
# ─────────────────────────────────────────────

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "doctorate", "diploma",
    "b.tech", "m.tech", "b.sc", "m.sc", "b.e", "m.e",
    "bca", "mca", "bba", "mba", "b.com", "m.com",
    "university", "college", "institute", "school",
    "computer science", "information technology",
    "engineering", "mathematics", "physics", "chemistry",
    "electronics", "electrical", "mechanical", "civil",
    "degree", "graduation", "post graduation", "certification",
    "12th", "10th", "higher secondary", "secondary",
    "iit", "nit", "iiit", "bits", "vit", "lpu", "amity",
    "cgpa", "gpa", "percentage", "aggregate",
    "coursework", "specialization", "minor", "major",
    "data science", "artificial intelligence",
    "biotechnology", "biomedical", "chemical engineering",
    "aerospace", "automobile", "industrial engineering",
]


# ─────────────────────────────────────────────
# 🔹 EXPERIENCE KEYWORDS
# ─────────────────────────────────────────────

EXPERIENCE_KEYWORDS = [
    "experience", "years of experience", "worked at", "working at",
    "intern", "internship", "fresher", "junior", "senior",
    "lead", "manager", "director", "team lead",
    "full time", "part time", "freelance", "contract",
    "responsibilities", "achievements", "projects",
    "developed", "implemented", "designed", "managed",
    "built", "created", "maintained", "optimized",
    "deployed", "architected", "collaborated", "researched",
    "analyzed", "tested", "automated", "integrated",
    "mentored", "supervised", "coordinated", "delivered",
    "contributed", "spearheaded", "launched", "scaled",
    "improved", "reduced", "increased", "streamlined",
    "training", "workshop", "hackathon", "open source",
    "startup", "company", "organization", "firm",
    "software engineer", "data analyst", "web developer",
    "ml engineer", "devops engineer", "full stack developer",
]


# ─────────────────────────────────────────────
# 🔹 HELPER FUNCTION
# ─────────────────────────────────────────────

def _match(keyword: str, text: str) -> bool:
    """Word-boundary safe keyword match."""
    pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


# ─────────────────────────────────────────────
# 🔹 SKILL EXTRACTION
# ─────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """
    Extract skills from resume text using regex keyword matching.

    Args:
        text (str): Cleaned resume text.

    Returns:
        list[str]: Skills found in the text.
    """
    text_lower = text.lower()
    return [skill for skill in SKILLS_LIST if _match(skill, text_lower)]


# ─────────────────────────────────────────────
# 🔥 NORMALIZATION
# ─────────────────────────────────────────────

def normalize_skills(skills: list[str], text: str) -> list[str]:
    """
    Infer higher-level skills from related keywords found in text.
    Example: "tensorflow" found → "deep learning" automatically added.

    Args:
        skills (list[str]): Already extracted raw skills.
        text (str): Original resume text.

    Returns:
        list[str]: Expanded, deduplicated, sorted skill list.
    """
    normalized = set(skills)
    text_lower = text.lower()

    for main_skill, related_keywords in SKILL_MAP.items():
        for keyword in related_keywords:
            if _match(keyword, text_lower):
                normalized.add(main_skill)
                break

    return sorted(normalized)


# ─────────────────────────────────────────────
# 🔹 EDUCATION
# ─────────────────────────────────────────────

def extract_education(text: str) -> list[str]:
    """
    Find education-related keywords in resume text.

    Args:
        text (str): Cleaned resume text.

    Returns:
        list[str]: Education keywords found.
    """
    text_lower = text.lower()
    return list(set([kw for kw in EDUCATION_KEYWORDS if _match(kw, text_lower)]))


# ─────────────────────────────────────────────
# 🔹 EXPERIENCE
# ─────────────────────────────────────────────

def extract_experience(text: str) -> list[str]:
    """
    Find experience-related keywords in resume text.

    Args:
        text (str): Cleaned resume text.

    Returns:
        list[str]: Experience indicators found.
    """
    text_lower = text.lower()
    return list(set([kw for kw in EXPERIENCE_KEYWORDS if _match(kw, text_lower)]))


# ─────────────────────────────────────────────
# 🔹 MAIN PIPELINE
# ─────────────────────────────────────────────

def extract_all(text: str) -> dict:
    """
    Full extraction pipeline:
      1. Extract raw skills via regex
      2. Normalize using SKILL_MAP inference
      3. Extract education and experience

    Args:
        text (str): Cleaned resume text.

    Returns:
        dict: {
            "skills": [...],
            "education": [...],
            "experience": [...]
        }
    """
    raw_skills = extract_skills(text)
    final_skills = normalize_skills(raw_skills, text)

    return {
        "skills": final_skills,
        "education": extract_education(text),
        "experience": extract_experience(text),
    } 