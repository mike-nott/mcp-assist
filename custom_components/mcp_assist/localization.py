"""Localization support for MCP Assist.

This module provides language-specific defaults for:
- System prompt language instructions
- Follow-up phrase detection patterns
- End conversation word detection patterns
"""

import logging

_LOGGER = logging.getLogger(__name__)

# Language metadata for generating system prompt instructions
LANGUAGE_METADATA = {
    # Tier 1 - Top 10 languages
    "de": {"english": "German", "native": "Deutsch"},
    "fr": {"english": "French", "native": "Français"},
    "es": {"english": "Spanish", "native": "Español"},
    "it": {"english": "Italian", "native": "Italiano"},
    "nl": {"english": "Dutch", "native": "Nederlands"},
    "pl": {"english": "Polish", "native": "Polski"},
    "pt": {"english": "Portuguese", "native": "Português"},
    "ru": {"english": "Russian", "native": "Русский"},
    "zh-hans": {"english": "Chinese Simplified", "native": "简体中文"},
    "ja": {"english": "Japanese", "native": "日本語"},

    # Tier 2 - Next 10 languages
    "ko": {"english": "Korean", "native": "한국어"},
    "sv": {"english": "Swedish", "native": "Svenska"},
    "nb": {"english": "Norwegian", "native": "Norsk Bokmål"},
    "da": {"english": "Danish", "native": "Dansk"},
    "fi": {"english": "Finnish", "native": "Suomi"},
    "cs": {"english": "Czech", "native": "Čeština"},
    "el": {"english": "Greek", "native": "Ελληνικά"},
    "tr": {"english": "Turkish", "native": "Türkçe"},
    "fil": {"english": "Filipino", "native": "Filipino"},
    "ar": {"english": "Arabic", "native": "العربية"},
    "hi": {"english": "Hindi", "native": "हिन्दी"},
}

# Follow-up phrases that indicate user wants to continue conversation
# These detect what the user actually says, so must be in their language
FOLLOW_UP_PHRASES = {
    "en": "anything else, what else, would you, do you, should i, can i, which, how can, what about, is there",

    # Tier 1
    "de": "noch etwas, was noch, würdest du, sollte ich, kann ich, welche, wie kann, was ist mit, gibt es",
    "fr": "autre chose, quoi d'autre, veux-tu, devrais-je, puis-je, lequel, comment puis, qu'en est-il, y a-t-il",
    "es": "algo más, qué más, harías, debería, puedo, cuál, cómo puedo, qué tal, hay",
    "it": "altro, cos'altro, vorresti, dovrei, posso, quale, come posso, che ne dici, c'è",
    "nl": "iets anders, wat nog meer, zou je, moet ik, kan ik, welke, hoe kan, hoe zit het met, is er",
    "pl": "coś jeszcze, co jeszcze, czy byś, czy powinienem, czy mogę, który, jak mogę, co z, czy jest",
    "pt": "mais alguma coisa, o que mais, você faria, devo, posso, qual, como posso, e quanto a, há",
    "ru": "что-нибудь ещё, что ещё, ты бы, должен ли я, могу ли я, какой, как я могу, а как насчёт, есть ли",
    "zh-hans": "还有什么, 还有其他, 你会, 我应该, 我可以, 哪个, 怎么能, 那个怎么样, 有没有",
    "ja": "他に何か, 他には, してくれる, すべき, できる, どれ, どうやって, どうですか, ありますか",

    # Tier 2
    "ko": "다른 것, 또 뭐, 할래, 해야 할까, 할 수 있어, 어떤, 어떻게, 어때, 있어",
    "sv": "något annat, vad mer, skulle du, borde jag, kan jag, vilken, hur kan, vad sägs om, finns det",
    "nb": "noe annet, hva mer, ville du, burde jeg, kan jeg, hvilken, hvordan kan, hva med, finnes det",
    "da": "noget andet, hvad mere, ville du, burde jeg, kan jeg, hvilken, hvordan kan, hvad med, findes der",
    "fi": "jotain muuta, mitä muuta, tekisitkö, pitäisikö minun, voinko, mikä, miten voin, entä, onko",
    "cs": "něco jiného, co ještě, udělal bys, měl bych, mohu, který, jak mohu, co třeba, je tam",
    "el": "κάτι άλλο, τι άλλο, θα ήθελες, πρέπει να, μπορώ να, ποιο, πώς μπορώ, τι λες για, υπάρχει",
    "tr": "başka bir şey, başka ne, yapar mısın, yapmalı mıyım, yapabilir miyim, hangi, nasıl yapabilirim, peki ya, var mı",
    "fil": "iba pa, ano pa, gagawin mo ba, dapat ba ako, maaari ba ako, alin, paano ako, paano naman, mayroon ba",
    "ar": "شيء آخر, ماذا أيضا, هل ستفعل, هل يجب أن, هل يمكنني, أي, كيف يمكنني, ماذا عن, هل يوجد",
    "hi": "कुछ और, और क्या, क्या आप, क्या मुझे करना चाहिए, क्या मैं कर सकता, कौन सा, मैं कैसे, क्या हो अगर, क्या है",
}

# End words that indicate user wants to stop conversation
# These detect what the user actually says, so must be in their language
END_WORDS = {
    "en": "stop, cancel, no, nope, thanks, thank you, bye, goodbye, done, never mind, nevermind, forget it, that's all, that's it",

    # Tier 1
    "de": "stopp, abbrechen, nein, danke, tschüss, auf wiedersehen, fertig, macht nichts, egal, vergiss es, das war's, das ist alles",
    "fr": "stop, arrête, annuler, non, merci, au revoir, salut, terminé, peu importe, laisse tomber, oublie ça, c'est tout",
    "es": "para, detener, cancelar, no, gracias, adiós, chao, hecho, no importa, olvídalo, es todo, eso es todo",
    "it": "stop, ferma, annulla, no, grazie, ciao, arrivederci, fatto, non importa, lascia perdere, dimenticalo, è tutto",
    "nl": "stop, annuleren, nee, bedankt, dank je, dag, doei, klaar, maakt niet uit, vergeet het, laat maar, dat is alles",
    "pl": "stop, anuluj, nie, dzięki, dziękuję, cześć, pa, gotowe, nieważne, zapomnij, to wszystko",
    "pt": "pare, parar, cancelar, não, obrigado, obrigada, tchau, adeus, pronto, não importa, esquece, é tudo, só isso",
    "ru": "стоп, отмена, нет, спасибо, пока, до свидания, готово, неважно, забудь, это всё, вот и всё",
    "zh-hans": "停止, 停, 取消, 不, 谢谢, 再见, 拜拜, 完成, 没关系, 算了, 忘了吧, 就这样, 好了",
    "ja": "ストップ, 止めて, キャンセル, いいえ, ありがとう, さようなら, バイバイ, 完了, 大丈夫, いいです, 忘れて, 以上, それだけ",

    # Tier 2
    "ko": "중지, 멈춰, 취소, 아니, 고마워, 감사, 안녕, 잘가, 완료, 괜찮아, 됐어, 잊어, 그게 다야, 끝",
    "sv": "stopp, stoppa, avbryt, nej, tack, hej då, adjö, klar, inget problem, glöm det, strunt samma, det var allt",
    "nb": "stopp, avbryt, nei, takk, ha det, adjø, ferdig, glem det, ikke noe, det var alt",
    "da": "stop, annuller, nej, tak, farvel, hej, færdig, glem det, ligemeget, det var det hele",
    "fi": "seis, pysäytä, peruuta, ei, kiitos, näkemiin, hei hei, valmis, ei haittaa, unohda, se oli kaikki",
    "cs": "stop, zastav, zrušit, ne, díky, děkuji, ahoj, nashledanou, hotovo, nevadí, zapomeň, to je vše",
    "el": "σταμάτα, στοπ, ακύρωση, όχι, ευχαριστώ, αντίο, γεια, τέλος, δεν πειράζει, άστο, ξέχνα το, αυτό είναι όλο",
    "tr": "dur, durdur, iptal, hayır, teşekkürler, güle güle, hoşça kal, tamam, bitti, önemli değil, boşver, unut, hepsi bu",
    "fil": "tigil, hinto, kanselahin, hindi, salamat, paalam, bye, tapos na, okay lang, kalimutan mo, yun lang, ayos na",
    "ar": "توقف, قف, إلغاء, لا, شكرا, مع السلامة, وداعا, انتهيت, لا بأس, انسى, هذا كل شيء, انتهى",
    "hi": "रुको, बंद करो, रद्द करो, नहीं, धन्यवाद, शुक्रिया, अलविदा, नमस्ते, हो गया, कोई बात नहीं, भूल जाओ, बस इतना ही",
}


def get_language_instruction(language_code: str) -> str:
    """Generate language-specific system prompt.

    Args:
        language_code: ISO 639-1 language code (e.g., "de", "fr-CA", "zh-Hans")

    Returns:
        Language-specific system prompt, or empty string for English (use default).

    Note:
        - System prompts stay in English for better LLM performance
        - Uses English language name in the main prompt for clarity
        - Uses native language name at the end as a stronger signal to the LLM
        - Handles regional variants by extracting base language code
    """
    if language_code == "en" or language_code.startswith("en-"):
        return ""  # English uses default prompt

    # Extract base language code (e.g., "fr-CA" -> "fr", "zh-Hans" -> "zh-hans")
    base_code = language_code.lower()
    if "-" in base_code and base_code not in LANGUAGE_METADATA:
        # Try extracting just the base part for variants like "pt-BR" -> "pt"
        base_code = base_code.split("-")[0]

    lang_info = LANGUAGE_METADATA.get(base_code)
    if not lang_info:
        _LOGGER.warning(
            "Language '%s' not found in metadata. Using English defaults. "
            "Consider adding this language to localization.py",
            language_code
        )
        return ""  # Fallback: use English default

    return (
        f"You are a helpful {lang_info['english']}-speaking Home Assistant voice assistant. "
        f"Respond naturally and conversationally to user requests in {lang_info['native']}."
    )


def get_follow_up_phrases(language_code: str) -> str:
    """Get follow-up phrases for language.

    Args:
        language_code: ISO 639-1 language code (e.g., "de", "fr-CA", "zh-Hans")

    Returns:
        Comma-separated string of follow-up phrases in the specified language.
        Falls back to English if language not found.
    """
    # Extract base language code
    base_code = language_code.lower()
    if "-" in base_code and base_code not in FOLLOW_UP_PHRASES:
        base_code = base_code.split("-")[0]

    phrases = FOLLOW_UP_PHRASES.get(base_code)
    if not phrases:
        _LOGGER.warning(
            "Follow-up phrases for language '%s' not found. Using English defaults. "
            "Consider adding this language to localization.py",
            language_code
        )
        return FOLLOW_UP_PHRASES["en"]

    return phrases


def get_end_words(language_code: str) -> str:
    """Get end conversation words for language.

    Args:
        language_code: ISO 639-1 language code (e.g., "de", "fr-CA", "zh-Hans")

    Returns:
        Comma-separated string of end words in the specified language.
        Falls back to English if language not found.
    """
    # Extract base language code
    base_code = language_code.lower()
    if "-" in base_code and base_code not in END_WORDS:
        base_code = base_code.split("-")[0]

    words = END_WORDS.get(base_code)
    if not words:
        _LOGGER.warning(
            "End words for language '%s' not found. Using English defaults. "
            "Consider adding this language to localization.py",
            language_code
        )
        return END_WORDS["en"]

    return words


def get_supported_languages() -> list[str]:
    """Get list of all supported language codes.

    Returns:
        List of ISO 639-1 language codes.
    """
    return sorted(LANGUAGE_METADATA.keys())
