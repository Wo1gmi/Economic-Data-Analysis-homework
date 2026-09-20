import re

CUTOFF_MONTH = "2025-02"

CUTOFF_SENSITIVITY = ["2024-11", "2025-02", "2025-06"]

PLACEBO_CUTOFF = "2024-02"
PLACEBO_SAMPLE_END = "2025-01"

EVENT_STUDY_BASE_MONTH = "2025-01"

ROLE_EXPOSURE_PERIOD_START = "2025-01"

JUN_J1_EXPERIENCE = {"noExperience", "between1And3"}
JUN_J2_EXPERIENCE = {"noExperience"}

JUN_J3_TITLE_PATTERN = re.compile(
    r"(?i)\bjunior\b|младш|стажёр|стажер|\bintern\b|\btrainee\b"
)

ORG_PLACEHOLDER_ROLE = "[организация]"
ORG_PLACEHOLDER_ROLE_TARGET = "Системный администратор"

ORG_PLACEHOLDER_ROLE_SUPPORT = "[организация] технической поддержки"
ORG_PLACEHOLDER_ROLE_SUPPORT_TARGET = "Специалист технической поддержки"

CODE_ROLES = {
    "Программист, разработчик",
    "Тестировщик",
    "DevOps-инженер",
    "Руководитель группы разработки",
}

NONCODE_ROLES = {
    "Специалист технической поддержки",
    ORG_PLACEHOLDER_ROLE_TARGET,
    "Сетевой инженер",
    "Системный инженер",
    "Специалист по информационной безопасности",
    "Руководитель проектов",
    "Дизайнер, художник",
    "Гейм-дизайнер",
    "Технический писатель",
    "Менеджер продукта",
}

ANALYTICS_ROLES = {
    "Аналитик",
    "Системный аналитик",
    "Бизнес-аналитик",
    "BI-аналитик, аналитик данных",
    "Дата-сайентист",
}

LEADERSHIP_ROLES = {
    "Технический директор (CTO)",
    "Директор по информационным технологиям (CIO)",
}

AI_MENTION_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"copilot|github\s*copilot|cursor\s*ai|"
    r"chatgpt|chat\s*gpt|gpt-?\d\w*|"
    r"llm|large\s+language\s+model|"
    r"нейросет\w*|промпт\w*|prompt\s*engineer\w*|"
    r"claude|anthropic|"
    r"gigachat|гигачат|yandexgpt|яндексgpt|"
    r"генеративн\w*\s+ии|генеративн\w*\s+искусственн\w*|"
    r"искусственн\w*\s+интеллект\w*|\bии\b|"
    r"vibe\s*coding|вайбкодинг|вайб-кодинг|"
    r"ai[- ]assist\w*"
    r")\b"
)

RANDOM_SEED = 20260219
RAW_N_ROWS = 85_086
