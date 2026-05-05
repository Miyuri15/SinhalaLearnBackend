# app/utils/sinhala_summary_prompt_builder.py
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _get_summary_rules(grade_level: str) -> dict:
    """
    Internal helper: maps grade_level enum to linguistic constraints
    """
    if grade_level == "6 - 8":
        return {
            "audience": "lower_secondary",
            "instruction": (
                "ඉතා සරල සිංහල භාවිතා කරන්න.\n"
                "කෙටි වාක්‍ය භාවිතා කරන්න (වාක්‍යයකට වචන 12-15 අතර).\n"
                "දැඩි අධ්‍යාපනික පද භාවිතා නොකරන්න.\n"
                "අමාරු වචන වෙනුවට සරල වචන භාවිතා කරන්න."
            ),
            "max_sentences": 7,
            "complexity": "basic",
        }

    if grade_level == "9 - 11":
        return {
            "audience": "upper_secondary",
            "instruction": (
                "සරල නමුත් සම්පූර්ණ සිංහල භාවිතා කරන්න.\n"
                "ප්‍රධාන අදහස් පැහැදිලි කරන්න.\n"
                "වාක්‍ය අතර සම්බන්ධතා පෙන්වන්න.\n"
                "විෂය කොටස් අතර පැහැදිලි සීමාවක් පවත්වා ගන්න."
            ),
            "max_sentences": 10,
            "complexity": "moderate",
        }

    if grade_level == "12 - 13":
        return {
            "audience": "advanced_secondary",
            "instruction": (
                "විස්තරාත්මක සිංහල භාවිතා කරන්න.\n"
                "අදහස් අතර සම්බන්ධතා පැහැදිලි කරන්න.\n"
                "ඓතිහාසික වැදගත්කම අවධාරණය කරන්න.\n"
                "සංකීර්ණ සංකල්ප සරලව පැහැදිලි කරන්න."
            ),
            "max_sentences": 13,
            "complexity": "detailed",
        }

    if grade_level == "university":
        return {
            "audience": "academic",
            "instruction": (
                "උසස් අධ්‍යයන මට්ටමේ සිංහල භාවිතා කරන්න.\n"
                "විශ්ලේෂණාත්මක වාක්‍ය සහ සංකල්ප භාවිතා කරන්න.\n"
                "කාරණා අතර සම්බන්ධතා සහ යටින් පවතින මූලධර්ම පැහැදිලි කරන්න.\n"
                "විවේචනාත්මක විශ්ලේෂණයක් ඉදිරිපත් කරන්න."
            ),
            "max_sentences": 16,
            "complexity": "analytical",
        }

    # safe fallback
    return {
        "audience": "default",
        "instruction": "සම්මත සිංහල සාරාංශයක් ලබා දෙන්න.",
        "max_sentences": 9,
        "complexity": "moderate",
    }


GRADE_LABEL_MAP = {
    "grade_6_8": "6 - 8",
    "grade_9_11": "9 - 11",
    "grade_12_13": "12 - 13",
    "university": "university",
}


def build_summary_prompt(
    context: str,
    grade_level: str,
    query: Optional[str] = None,
) -> str:
    """
    Builds a grade-adaptive, zero-hallucination Sinhala summary prompt
    """
    normalized_level = grade_level.value if hasattr(grade_level, "value") else grade_level
    key = GRADE_LABEL_MAP.get(normalized_level)
    rules = _get_summary_rules(key)
    
    query_part = f"ප්‍රශ්නය / ඉල්ලීම: {query}\n\n" if query else ""

    # complexity-based guidance with structure focus
    complexity_guidance = {
        "basic": """
📌 **සාරාංශ ව්‍යුහය:**
• අන්තර්ගතයේ ප්‍රධාන කොටස් 2-3 ක් වෙන් වෙන් වශයෙන් ඉදිරිපත් කරන්න
• එක් එක් කොටස සඳහා වාක්‍ය 2-3 බැගින් භාවිතා කරන්න
• කොටස් අතර පැහැදිලි බෙදීමක් ඇති කිරීමට අවකාශයක් තබන්න
• අනවශ්‍ය විස්තර ඉවත් කර ප්‍රධාන කරුණු පමණක් ඇතුළත් කරන්න
""",
        "moderate": """
📌 **සාරාංශ ව්‍යුහය:**
• අන්තර්ගතයේ ඇති ප්‍රධාන කොටස් වෙන් වෙන් වශයෙන් ඉදිරිපත් කරන්න
• එක් එක් කොටස සඳහා වාක්‍ය 3-4 බැගින් භාවිතා කරන්න
• කොටස් අතර සංක්‍රාන්තිය පැහැදිලි කරන්න
• ප්‍රධාන කරුණු සහ උප කරුණු යන දෙකම ඇතුළත් කරන්න
""",
        "detailed": """
📌 **සාරාංශ ව්‍යුහය:**
• අන්තර්ගතයේ සියලු ප්‍රධාන කොටස් වෙන් වෙන් වශයෙන් ඉදිරිපත් කරන්න
• එක් එක් කොටස සඳහා වාක්‍ය 4-5 බැගින් භාවිතා කරන්න
• එක් එක් කොටස තුළදී තේමාත්මක එකමුතුකමක් පවත්වා ගන්න
• වැදගත් උදාහරණ, දින, නම්, ස්ථාන ඇතුළත් කරන්න
• හේතු-ඵල සම්බන්ධතා පැහැදිලි කරන්න
""",
        "analytical": """
📌 **සාරාංශ ව්‍යුහය:**
• අන්තර්ගතයේ කොටස් අතර සම්බන්ධතා සහ වෙනස්කම් විශ්ලේෂණය කරන්න
• එක් එක් කොටස සඳහා වාක්‍ය 5-6 බැගින් භාවිතා කරන්න
• යටින් පවතින මූලධර්ම සහ රටා හඳුනාගන්න
• ඓතිහාසික/සංස්කෘතික/විද්‍යාත්මක වැදගත්කම විශ්ලේෂණය කරන්න
• විවේචනාත්මක දෘෂ්ටිකෝණයකින් විමසා බලන්න
• න්‍යායාත්මක සංකල්ප භාවිතා කරමින් විශ්ලේෂණය කරන්න
""",
    }
    
    complexity_text = complexity_guidance.get(rules.get("complexity", "moderate"), "")

    return f"""
ඔබට පහත **අධ්‍යයන අන්තර්ගතය** ලබා දී ඇත.
මෙම අන්තර්ගතය මත පමණක් පදනම්ව සාරාංශයක් සාදන්න.

🎓 ඉලක්ක පාඨකයා: {rules['audience']}
📏 උපරිම වාක්‍ය ගණන: {rules['max_sentences']}

🧠 භාෂා නියම:
{rules['instruction']}

{complexity_text}

🚫 **අනුගමනය කළ යුතු ආකෘතිය:**
• සාරාංශය පැරග්‍රාෆ් ආකාරයෙන් ලියන්න
• වාක්‍ය අංක, බුලට් පොයින්ට්, හයිෆන, තරු ලකුණු භාවිතා නොකරන්න
• ලැයිස්තුගත කිරීම් වලින් වළකින්න
• සියල්ල සම්පූර්ණ වාක්‍ය ලෙස ලියන්න

🔒 **නිරපේක්ෂ තහනම් කරුණු (ZERO HALLUCINATION):**
• ලබා දී ඇති අන්තර්ගතයේ නොමැති කිසිදු තොරතුරක් එකතු නොකරන්න
• අන්තර්ගතයේ නොමැති රටවල්, නගර, පුද්ගලයන්, සිදුවීම්, දිනයන් ඇතුළත් නොකරන්න
• නව උදාහරණ, නිගමන, අර්ථකථන සෑදීමෙන් වළකින්න
• ඔබගේ පෞද්ගලික දැනුමෙන් කිසිදු තොරතුරක් එකතු නොකරන්න
• අන්තර්ගතයේ ඇති තොරතුරු පමණක්, ඒවායේ ඇති ආකාරයටම භාවිතා කරන්න
• ලබා දී ඇති අන්තර්ගතය සාරාංශ කිරීමට හෝ ඉල්ලන ලද තොරතුරු ලබා දීමට ප්‍රමාණවත් නොවේ නම්, ඔබගේ පිළිතුර '[NOT_ANSWERABLE]' යන වචනයෙන් ආරම්භ කරන්න.

{query_part}📚 **අන්තර්ගතය:**
{context}

✍️ **සාරාංශය:**
"""