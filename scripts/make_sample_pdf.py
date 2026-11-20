"""Generate the demo document samples/ichki-tartib-qoidalari.pdf.

The company and every rule in it are fictional, written for this demo and its evaluation set.
Run: python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT = Path(__file__).resolve().parent.parent / "samples" / "ichki-tartib-qoidalari.pdf"

TITLE = "Nurli Texnologiyalar MChJ - Xodimlar uchun ichki tartib qoidalari"
NOTE = "Eslatma: kompaniya va qoidalar xayoliy. Hujjat RAG tizimini sinash uchun tayyorlangan."

SECTIONS = [
    (
        "1-bo'lim. Umumiy qoidalar va ish vaqti",
        [
            "Ushbu qoidalar Nurli Texnologiyalar MChJ ning barcha xodimlariga, shu jumladan sinov muddatidagi "
            "xodimlarga ham tatbiq etiladi. Qoidalar 2026-yil 1-yanvardan kuchga kirgan.",
            "Ish haftasi besh kunlik: dushanbadan jumagacha. Ish kuni soat 09:00 da boshlanadi va 18:00 da "
            "tugaydi. Tushlik tanaffusi 13:00 dan 14:00 gacha bo'lib, ish vaqtiga kirmaydi. Shunday qilib, "
            "haftalik ish vaqti 40 soatni tashkil etadi.",
            "Ishga 15 daqiqadan ko'proq kechikadigan xodim bevosita rahbariga oldindan xabar berishi shart. "
            "Bir oy ichida uch martadan ortiq sababsiz kechikish intizomiy suhbatga asos bo'ladi.",
            "Ish vaqtidan tashqari ishlash faqat rahbarning yozma topshirig'i bilan amalga oshiriladi va "
            "ikki baravar miqdorda haq to'lanadi yoki xodimning tanloviga ko'ra qo'shimcha dam olish kuni bilan "
            "qoplanadi.",
        ],
    ),
    (
        "2-bo'lim. Ta'til va kasallik",
        [
            "Har bir xodim yiliga 21 ish kunidan iborat haq to'lanadigan mehnat ta'tiliga ega. Ta'tilni ikki "
            "qismga bo'lish mumkin, lekin qismlardan biri kamida 7 ish kunidan kam bo'lmasligi kerak.",
            "Ta'til uchun ariza HR tizimi orqali kamida 14 kalendar kun oldin topshiriladi. Ta'til jadvali "
            "har yili 15-dekabrgacha tasdiqlanadi.",
            "Xodim kasal bo'lib qolsa, ish kuni boshlanishidan oldin rahbariga xabar beradi. Kasallik varaqasi "
            "ishga qaytgandan keyin 3 ish kuni ichida HR bo'limiga topshiriladi.",
            "Oilaviy holatlar uchun qo'shimcha haq to'lanadigan dam olish kunlari: o'z to'yi uchun 3 kun, "
            "farzand tug'ilganda otaga 5 kun, yaqin qarindoshi vafot etganda 3 kun.",
        ],
    ),
    (
        "3-bo'lim. Masofaviy ish",
        [
            "Xodimlar haftasiga ko'pi bilan 2 kun masofadan (uydan) ishlashi mumkin. Masofaviy kunlar "
            "jamoa rahbari bilan kelishiladi va jamoa taqvimida belgilanadi.",
            "Sinov muddatidagi xodimlar, ya'ni ishga kirganidan keyingi dastlabki 3 oy davomida, masofadan "
            "ishlay olmaydi.",
            "Masofadan ishlaganda kompaniya tarmog'iga faqat korporativ VPN orqali ulaniladi. Ochiq jamoat "
            "Wi-Fi tarmoqlaridan VPN siz foydalanish taqiqlanadi.",
            "Masofaviy ish uchun internet xarajatlari kompensatsiyasi oyiga 150 000 so'm miqdorida ish haqi "
            "bilan birga to'lanadi.",
        ],
    ),
    (
        "4-bo'lim. Xizmat safari va xarajatlar",
        [
            "Xizmat safari rahbar va moliya bo'limi tomonidan oldindan tasdiqlanadi. Aviachiptalar faqat "
            "ekonom klassda xarid qilinadi.",
            "Sutkalik xarajatlar: O'zbekiston hududida kuniga 200 000 so'm, xorijiy davlatlarda kuniga "
            "60 AQSh dollari. Mehmonxona xarajati sutkalikka kirmaydi va alohida qoplanadi.",
            "Safardan qaytgach, xodim xarajatlar hisobotini barcha cheklar bilan birga 5 ish kuni ichida "
            "moliya bo'limiga topshiradi. Chek bo'lmagan xarajatlar qoplanmaydi.",
            "Ish bilan bog'liq taksi xarajatlari faqat soat 22:00 dan keyin yoki og'ir yuk tashilganda qoplanadi.",
        ],
    ),
    (
        "5-bo'lim. Axborot xavfsizligi",
        [
            "Barcha hisoblar uchun parol kamida 12 belgidan iborat bo'lishi, katta va kichik harflar, raqam va "
            "maxsus belgini o'z ichiga olishi kerak. Parol har 90 kunda almashtiriladi.",
            "Pochta, VPN va kod omboriga kirishda ikki bosqichli autentifikatsiya majburiy.",
            "Shaxsiy USB fleshkalar va tashqi disklarni ish kompyuterlariga ulash taqiqlanadi. Fayllar faqat "
            "korporativ bulutli xotira orqali almashiladi.",
            "Noutbuk yoki telefon yo'qolsa, xodim 1 soat ichida IT bo'limiga (ichki raqam 1100) xabar berishi "
            "shart, qurilma masofadan bloklanadi.",
        ],
    ),
    (
        "6-bo'lim. Ish haqi, o'qish va rivojlanish",
        [
            "Ish haqi oyiga ikki marta to'lanadi: har oyning 5 va 20-sanasida. To'lov kuni dam olish kuniga "
            "to'g'ri kelsa, ish haqi oldingi ish kunida to'lanadi.",
            "Har bir xodimning yillik o'qish byudjeti 4 000 000 so'm. Undan kurslar, konferensiyalar va "
            "sertifikatlash imtihonlari uchun foydalanish mumkin; ariza rahbar tomonidan tasdiqlanadi.",
            "Yangi xodimlarga 6 oylik mentorlik dasturi doirasida tajribali hamkasb biriktiriladi.",
            "Yillik baholash har yili dekabr oyida o'tkaziladi, natijalariga ko'ra lavozim va ish haqi "
            "qayta ko'rib chiqiladi.",
        ],
    ),
]


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=8)
    story = [Paragraph(TITLE, styles["Title"]), Paragraph(NOTE, styles["Italic"]), Spacer(1, 0.6 * cm)]
    for index, (heading, paragraphs) in enumerate(SECTIONS):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading2"]))
        story.extend(Paragraph(text, body) for text in paragraphs)
    SimpleDocTemplate(str(OUTPUT), pagesize=A4, title=TITLE, author="pdf-rag-assistant demo", invariant=1).build(story)
    return OUTPUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
