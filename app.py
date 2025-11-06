#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for the 'Ouderschapsplan' conversational assistant.
- Keeps your existing Gemini via Vertex AI flow
- Adds an easy-to-follow chat interface and progress sidebar
"""

import os
import json
import time
import re
from typing import List, Dict, Any
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

# Google AI (Vertex) SDK
from google import genai
from google.genai import types

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from jinja2 import Environment, BaseLoader, StrictUndefined, TemplateError

# ---------- ENV & CONFIG ----------
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION   = os.getenv("LOCATION", "europe-west4")
MODEL_ID   = os.getenv("MODEL_ID", "gemini-2.5-flash")

if not PROJECT_ID:
    raise ValueError("PROJECT_ID environment variable is required (.env)")

# Load question list from external JSON file
def load_questions():
    """Load questions from the questions.json file."""
    questions_path = os.path.join(os.path.dirname(__file__), "questions", "questions.json")
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found at: {questions_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in questions file: {e}")

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Ouderschapsplan Assistent", layout="wide")

# ---------- QUESTION LIST (same as your script) ----------
question_list_template: List[Dict[str, Any]] = load_questions()

tmpl = st.secrets["SYSTEM_INSTRUCTION"]

today_nl = time.strftime("%d-%m-%Y")

system_instruction = tmpl.replace("{{TODAY_NL}}", today_nl)

# ---------- SYSTEM INSTRUCTION ----------
# Raw string to avoid \s escape warnings inside regex-like text.

# system_instruction = st.secrets.get("SYSTEM_INSTRUCTION")
if not system_instruction and "SYSTEM_INSTRUCTION_B64" in st.secrets:
    system_instruction = base64.b64decode(st.secrets["SYSTEM_INSTRUCTION_B64"]).decode("utf-8")

if not system_instruction:
    st.error("Missing SYSTEM_INSTRUCTION in secrets.")
    st.stop()
# ---------- STREAMLIT STATE ----------
if "question_list" not in st.session_state:
    # Deep copy template
    st.session_state.question_list = json.loads(json.dumps(question_list_template, ensure_ascii=False))

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ---------- CLIENT (cached) ----------
@st.cache_resource(show_spinner=False)
def get_client():
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=types.HttpOptions(api_version="v1"),
    )
    return client

# ---------- HELPERS ----------

def _extract_balanced_json(text: str) -> str | None:
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def _normalize_quotes(s: str) -> str:
    return (s.replace('\u201c', '"').replace('\u201d', '"')
             .replace('\u2018', "'").replace('\u2019', "'"))

def coerce_json(raw_text: str) -> dict:
    if not raw_text:
        return {}
    s = raw_text.lstrip('\ufeff').strip()
    s = _normalize_quotes(s)
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try balanced extract
    core = _extract_balanced_json(s)
    if core:
        try:
            return json.loads(core)
        except Exception:
            # last resort: escape raw newlines inside strings
            core2 = core.replace('\r', '\\r').replace('\n', '\\n')
            return json.loads(core2)
    raise ValueError("Could not coerce model output to JSON")

jinja_env = Environment(
    loader=BaseLoader(),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined  # fail fast if a placeholder is missing
)

def render_summary_from_template(question: Dict[str, Any], details: Dict[str, Any]) -> str | None:
    """
    If question has a 'summary_template' and all needed fields are present in `details`,
    render a formal summary string. Returns None if it can't render.
    """
    tpl = question.get("summary_template")
    if not tpl:
        return None
    try:
        template = jinja_env.from_string(tpl)
        # normalize keys: allow both 'fields' names and derived names from required_details
        return template.render(**details).strip()
    except TemplateError:
        return None


# ---------- MODEL CALL ----------
def send_to_gemini(user_input: str, sender_id: str = "Sandra") -> Dict[str, Any]:
    client = get_client()

    contents = []
    for entry in st.session_state.history:
        role = "model" if entry["role"] == "assistant" else "user"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=entry["content"])]
            )
        )

    current_message = f"""[CURRENT MESSAGE TO ANALYZE] {sender_id}: {user_input}

[QUESTION_LIST]
{json.dumps(st.session_state.question_list, ensure_ascii=False, indent=2)}
"""

    contents.append(types.Content(role="user", parts=[types.Part(text=current_message)]))

    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.4,
        max_output_tokens=8192,  # Increased from 2048 to prevent truncation
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=cfg,
    )
    raw_text = resp.text or "{}"
    
    try:
        out = coerce_json(raw_text)

        # Try to render a formal summary if possible
        try:
            # find the current question being worked on (lowest pending OR one in updated_questions)
            current_q = None
            if out.get("updated_questions"):
                # when the model updates one, take that
                uq = out["updated_questions"][0]
                current_q = next((q for q in st.session_state.question_list if q["id"] == uq["id"]), uq)
            else:
                _, _, current_q = compute_progress()

            # fields from the model (optional)
            fields = out.get("fields") or {}

            if current_q:
                rendered = render_summary_from_template(current_q, fields)
                if rendered:
                    # ensure the question in session gets the summary
                    for q in st.session_state.question_list:
                        if q["id"] == current_q["id"]:
                            q["summary"] = rendered
                            break
                else:
                    # if not rendered but model provided a formal 'summary', use it
                    if out.get("summary"):
                        for q in st.session_state.question_list:
                            if q["id"] == current_q["id"]:
                                q["summary"] = out["summary"].strip()
                                break
        except Exception as e:
            print(f"[WARN] Summary rendering failed: {e}")

    except Exception as e:
        print(f"[ERROR] JSON coercion failed: {e}")
        print(f"[ERROR] Full raw text:\n{raw_text}")
        
        # Try to salvage the answer field
        m = re.search(r'"answer"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', raw_text)
        if m:
            answer_text = m.group(1).replace('\\"', '"')
            print(f"[SALVAGE] Extracted answer: {answer_text[:100]}...")
            out = {"answer": answer_text, "updated_questions": None}
        else:
            # If even that fails, provide a helpful error message
            raise ValueError(
                f"Could not parse model response. Length: {len(raw_text)} chars. "
                f"First 200 chars: {raw_text[:200]}... "
                f"Last 200 chars: ...{raw_text[-200:]}"
            )
    return out

# ---------- HELPERS ----------
def merge_updates(updated_questions: List[Dict[str, Any]]):
    if not updated_questions:
        return
    qmap = {q["id"]: q for q in st.session_state.question_list}
    for uq in updated_questions:
        if uq.get("id") in qmap:
            # Ensure critical fields are present when status is completed
            if uq.get("status") == "completed":
                if not uq.get("answer"):
                    print(f"[WARN] Question {uq.get('id')} marked completed but missing 'answer' field")
                if not uq.get("summary") and uq.get("summary_template"):
                    print(f"[WARN] Question {uq.get('id')} marked completed but missing 'summary' field")
            qmap[uq["id"]] = uq
    # Rebuild list preserving original order by position
    st.session_state.question_list = sorted(qmap.values(), key=lambda x: x["position"])

def compute_progress():
    total = len(st.session_state.question_list)
    done = sum(1 for q in st.session_state.question_list if q.get("status") == "completed")
    next_q = next((q for q in st.session_state.question_list if q.get("status") == "pending" and not q.get("answer")), None)
    return total, done, next_q

def export_payload():
    return {
        "question_list": st.session_state.question_list,
        "conversation_history": st.session_state.history,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def get_conversations_dir():
    """Get or create the conversations directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversations_dir = os.path.join(script_dir, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)
    return conversations_dir

# Korte, formele titels per vraag-id voor het formele PDF
TITLE_BY_ID = {
    "ouders_identiteit_0": "Gegevens van de ouders",
    "juridische_situatie_1a": "Burgerlijke staat",
    "gezag_1b": "Gezag",
    "relatie_tijdlijn_1c": "Tijdlijn relatie",
    "kinderen_identiteit_2": "Gegevens van de kinderen",
    "betrokkenheid_kinderen_3": "Betrokkenheid van de kinderen",
    "waarden_principes_4": "Gezamenlijke principes",
    "hoofdverblijf_5a": "Hoofdverblijfplaats",
    "kinderbijslag_budget_5b": "Kinderbijslag en kindgebonden budget",
    "zorgverdeling_week_6": "Weekritme zorgverdeling",
    "wissels_vervoer_7": "Overdrachten en vervoer",
    "contact_met_andere_ouder_8": "Contact met de andere ouder",
    "school_beleid_9": "School en onderwijs",
    "medisch_beleid_10": "Medische zorg",
    "identiteit_documenten_11a": "Identiteitsdocumenten",
    "reistoestemming_11b": "Reistoestemming buitenland",
    "verzekeringen_12": "Verzekeringen",
    "vakanties_13": "Schoolvakanties",
    "feestdagen_14": "Feestdagen",
    "verjaardagen_15": "Verjaardagen",
    "familie_contacten_16a": "Contact met familie",
    "overlijden_ouder_16b": "Bij overlijden ouder",
    "opvang_oppas_17": "Opvang en oppas",
    "sport_hobby_18": "Sport, muziek en hobby’s",
    "communicatie_19a": "Communicatie tussen ouders",
    "overleg_19b": "Formeel overleg",
    "financien_20": "Financiële afspraken",
    "jongmeerderjarig_21": "Afspraken 18–21 jaar / studiebijdrage",
    "spaarrekeningen_kind_22": "Spaarrekeningen kind",
    "verhuizen_23": "Verhuizen",
    "nieuwe_partner_24": "Introductie nieuwe partner",
    "evaluatie_geschillen_25": "Evaluatie en geschillenregeling",
    "ondertekening_26": "Ondertekening"
}

def generate_formal_pdf() -> BytesIO:
    """
    Generate a professional PDF document with completed questions only.
    Styled similar to official parenting plan documents.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )
    
    # Container for PDF elements
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Title style - elegant and professional
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=24
    )
    
    # Subtitle style
    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#5D6D7E'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica',
        leading=14
    )
    
    # Section heading style
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1A5490'),
        spaceAfter=10,
        spaceBefore=16,
        fontName='Helvetica-Bold',
        leading=16,
        borderWidth=0,
        borderPadding=0
    )
    
    # Question style
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=6,
        fontName='Helvetica-Bold',
        leading=14
    )
    
    # Korte clausule-titel i.p.v. de volledige vraag
    clause_title_style = ParagraphStyle(
        'ClauseTitle',
        parent=styles['Heading3'],
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor('#2C3E50'),
        spaceBefore=6,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )

    # Answer style
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=15,
        spaceAfter=18,
        leftIndent=15,
        fontName='Helvetica',
        textColor=colors.HexColor('#34495E'),
        alignment=TA_LEFT
    )
    
    # Info text style
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=10,
        fontName='Helvetica',
        leading=12
    )
    
    # Header section with title
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("OUDERSCHAPSPLAN", title_style))
    elements.append(Paragraph("Afspraken over de zorg en opvoeding van de kinderen", subtitle_style))
    
    # Horizontal line
    line_table = Table([['']], colWidths=[16*cm])
    line_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#BDC3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(line_table)
    
    # Date and metadata
    date_text = f"Datum opstelling: {time.strftime('%d %B %Y')}"
    elements.append(Paragraph(date_text, info_style))
    elements.append(Spacer(1, 0.8*cm))
    
    # Filter only completed questions and group by sections
    completed_questions = [q for q in st.session_state.question_list if q.get("status") == "completed"]
    
    if not completed_questions:
        elements.append(Paragraph("Nog geen vragen beantwoord.", answer_style))
    else:
        # Define sections for better organization
        section_mapping = {
            # 1) Partijen & juridische context
            1: "1. Partijen & juridische context",
            2: "1. Partijen & juridische context",
            3: "1. Partijen & juridische context",
            4: "1. Partijen & juridische context",

            # 2) Kinderen: gegevens & betrokkenheid
            5: "2. Kinderen: gegevens & betrokkenheid",
            6: "2. Kinderen: gegevens & betrokkenheid",

            # 3) Uitgangspunten & gezamenlijke principes
            7: "3. Uitgangspunten & gezamenlijke principes",

            # 4) Zorgregeling & contact
            8: "4. Zorgregeling & contact",
            9: "4. Zorgregeling & contact",
            10: "4. Zorgregeling & contact",
            11: "4. Zorgregeling & contact",
            12: "4. Zorgregeling & contact",

            # 5) School & huiswerk
            13: "5. School & huiswerk",

            # 6) Gezondheid, documenten & verzekeringen
            14: "6. Gezondheid, documenten & verzekeringen",
            15: "6. Gezondheid, documenten & verzekeringen",
            16: "6. Gezondheid, documenten & verzekeringen",
            17: "6. Gezondheid, documenten & verzekeringen",

            # 7) Vakanties, feestdagen & verjaardagen
            18: "7. Vakanties, feestdagen & verjaardagen",
            19: "7. Vakanties, feestdagen & verjaardagen",
            20: "7. Vakanties, feestdagen & verjaardagen",

            # 8) Familie & bijzondere omstandigheden
            21: "8. Familie & bijzondere omstandigheden",
            22: "8. Familie & bijzondere omstandigheden",

            # 9) Opvang, sport & hobby’s
            23: "9. Opvang, sport & hobby’s",
            24: "9. Opvang, sport & hobby’s",

            # 10) Communicatie & overleg
            25: "10. Communicatie & overleg",
            26: "10. Communicatie & overleg",

            # 11) Financiën
            27: "11. Financiën",
            28: "11. Financiën",
            29: "11. Financiën",

            # 12) Verhuizen & nieuwe partner
            30: "12. Verhuizen & nieuwe partner",
            31: "12. Verhuizen & nieuwe partner",

            # 13) Evaluatie & geschillen
            32: "13. Evaluatie & geschillen",

            # 14) Ondertekening
            33: "14. Ondertekening",
        }
        
        current_section = None
        
        # Add each completed question with its answer
        for question in completed_questions:
            position = question.get('position', 0)
            section_name = section_mapping.get(position, "")
            
            # Add section header if new section
            if section_name and section_name != current_section:
                current_section = section_name
                elements.append(Spacer(1, 0.3*cm))
                elements.append(Paragraph(section_name, section_style))
                # Add subtle line under section header
                section_line = Table([['']], colWidths=[16*cm])
                section_line.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#E8EBED')),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(section_line)
            
            # Title/onderwerp i.p.v. volledige vraag
            title_text = TITLE_BY_ID.get(question.get('id', ''), question.get('question', 'Onderwerp'))
            elements.append(Paragraph(title_text, clause_title_style))

            
            # Answer - use summary if available, otherwise answer
            answer_text = question.get('summary') or question.get('answer', 'Geen antwoord')
            # Clean up answer text for PDF
            answer_clean = answer_text.replace('\n', '<br/>')
            elements.append(Paragraph(answer_clean, answer_style))
    
    # Add page break before signatures
    elements.append(PageBreak())
    
    # Signature section - professional layout
    elements.append(Spacer(1, 1*cm))
    
    # Signature header
    sig_header_style = ParagraphStyle(
        'SigHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=10,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )
    elements.append(Paragraph("ONDERTEKENING", sig_header_style))
    
    # Horizontal line
    elements.append(line_table)
    elements.append(Spacer(1, 0.3*cm))
    
    # Declaration text
    declaration_style = ParagraphStyle(
        'Declaration',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=20,
        fontName='Helvetica',
        leading=14,
        alignment=TA_LEFT
    )
    
    elements.append(Paragraph(
        "Ondergetekenden verklaren dat zij kennis hebben genomen van de inhoud van dit ouderschapsplan "
        "en dat zij zich zullen houden aan de gemaakte afspraken. Dit plan is opgesteld in het belang "
        "van het kind/de kinderen en zal als uitgangspunt dienen voor de ouderlijke zorg en opvoeding.",
        declaration_style
    ))
    elements.append(Spacer(1, 0.8*cm))
    
    # Signature field style - more compact
    sig_field_style = ParagraphStyle(
        'SignatureField',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=6,
        fontName='Helvetica',
        leading=12
    )
    
    sig_label_style = ParagraphStyle(
        'SignatureLabel',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#1A5490'),
        spaceAfter=8,
        fontName='Helvetica-Bold',
        leading=12
    )
    
    # Parent 1 signature block
    elements.append(Paragraph("Ouder 1", sig_label_style))
    elements.append(Paragraph("Naam: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("Handtekening: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("Datum: _________________________________________________________________", sig_field_style))
    
    elements.append(Spacer(1, 0.8*cm))
    
    # Parent 2 signature block
    elements.append(Paragraph("Ouder 2", sig_label_style))
    elements.append(Paragraph("Naam: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("Handtekening: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("Datum: _________________________________________________________________", sig_field_style))
    
    elements.append(Spacer(1, 0.8*cm))
    
    # Mediator signature block
    elements.append(Paragraph("Mediator", sig_label_style))
    elements.append(Paragraph("Naam gekozen / te kiezen mediator: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("Handtekening: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("Datum: _________________________________________________________________", sig_field_style))
    
    # Footer note
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#95A5A6'),
        spaceAfter=0,
        fontName='Helvetica',
        leading=10,
        alignment=TA_CENTER
    )
    
    elements.append(Spacer(1, 1*cm))
    elements.append(Paragraph(
        "Dit ouderschapsplan is opgesteld met behulp van de BeNice.family",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def generate_discussion_pdf() -> BytesIO:
    """
    Generate an informal discussion document for mediation sessions.
    More conversational, focusing on parents' thoughts and opinions.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )
    
    # Container for PDF elements
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Title style - warm and approachable
    title_style = ParagraphStyle(
        'DiscussionTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#E67E22'),
        spaceAfter=8,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        leading=22
    )
    
    # Subtitle style
    subtitle_style = ParagraphStyle(
        'DiscussionSubTitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=25,
        alignment=TA_LEFT,
        fontName='Helvetica',
        leading=14
    )
    
    # Section heading style - friendly
    section_style = ParagraphStyle(
        'DiscussionSection',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#E67E22'),
        spaceAfter=8,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        leading=15
    )
    
    # Question style - conversational
    question_style = ParagraphStyle(
        'DiscussionQuestion',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=5,
        fontName='Helvetica-Bold',
        leading=13
    )
    
    # Answer style - natural and readable
    answer_style = ParagraphStyle(
        'DiscussionAnswer',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        spaceAfter=16,
        leftIndent=10,
        fontName='Helvetica',
        textColor=colors.HexColor('#2C3E50'),
        alignment=TA_LEFT
    )
    
    # Note style
    note_style = ParagraphStyle(
        'NoteStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#95A5A6'),
        spaceAfter=10,
        fontName='Helvetica-Oblique',
        leading=12
    )
    
    # Header section
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph("Gespreksdocument Ouderschapsplan", title_style))
    elements.append(Paragraph(
        "Een informeel overzicht van de afspraken en gedachten van beide ouders - voor gebruik tijdens mediation of overleg",
        subtitle_style
    ))
    
    # Date and metadata
    date_text = f"Gespreksnotitie van {time.strftime('%d %B %Y')}"
    elements.append(Paragraph(date_text, note_style))
    elements.append(Spacer(1, 0.5*cm))
    
    # Intro text
    intro_style = ParagraphStyle(
        'IntroStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#5D6D7E'),
        spaceAfter=20,
        fontName='Helvetica',
        leading=14,
        leftIndent=10,
        rightIndent=10
    )
    
    elements.append(Paragraph(
        "<i>Dit document bevat de gedachten en afspraken die Sandra gedeeld heeft tijdens het opstellen van haar ouderschapsplan in BeNice. "
        "Het is bedoeld als gespreksonderwerp en werkdocument - niet als juridisch bindend contract. "
        "De informatie hieronder weerspiegelt de meningen en wensen van Sandra in haar eigen woorden.</i>",
        intro_style
    ))
    elements.append(Spacer(1, 0.8*cm))
    
    # Filter only completed questions
    completed_questions = [q for q in st.session_state.question_list if q.get("status") == "completed"]
    
    if not completed_questions:
        elements.append(Paragraph("Nog geen gespreksonderwerpen besproken.", answer_style))
    else:
        # Define sections for discussion document
        section_mapping = {
            1: "👥 Over jullie en de kinderen",
            2: "👥 Over jullie en de kinderen",
            3: "💭 Wat vinden jullie belangrijk?",
            4: "🤝 Hoe gaan jullie met elkaar om?",
            5: "📅 De zorgregeling",
            6: "📅 De zorgregeling",
            7: "📅 De zorgregeling",
            8: "📅 De zorgregeling",
            9: "📅 De zorgregeling",
            10: "🎒 Praktische afspraken in het dagelijks leven",
            11: "🎒 Praktische afspraken in het dagelijks leven",
            12: "⚽ Sport, hobby's en activiteiten",
            13: "📚 School en ontwikkeling",
            14: "🏥 Gezondheid en medische zorg",
            15: "🏥 Gezondheid en medische zorg",
            16: "🏠 Opvoeding en huisregels",
            17: "👨‍👩‍👧 Opvang en hulp",
            18: "🚗 Als iemand wil verhuizen",
            19: "💰 Kosten en geld",
            20: "💰 Kosten en geld"
        }
        
        current_section = None
        
        # Add each completed question with its answer
        for question in completed_questions:
            position = question.get('position', 0)
            section_name = section_mapping.get(position, "")
            
            # Add section header if new section
            if section_name and section_name != current_section:
                current_section = section_name
                elements.append(Spacer(1, 0.3*cm))
                elements.append(Paragraph(section_name, section_style))
                # Subtle separator
                separator = Table([['']], colWidths=[16*cm])
                separator.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#F39C12')),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(separator)
            
            # Question text - more conversational
            question_text = f"• {question['question']}"
            elements.append(Paragraph(question_text, question_style))
            
            # Answer - use the raw answer (not summary) for more personal touch
            answer_text = question.get('answer', 'Geen antwoord gegeven')
            
            # Make the answer more conversational if it's too formal
            # Use answer instead of summary to keep the personal voice
            answer_clean = answer_text.replace('\n', '<br/>')
            
            # Add quotes if it's clearly from the parents' perspective
            if any(word in answer_text.lower() for word in ['jullie', 'we', 'wij', 'ons', 'onze']):
                answer_display = f'<i>"{answer_clean}"</i>'
            else:
                answer_display = answer_clean
            
            elements.append(Paragraph(answer_display, answer_style))
    
    # Footer - informal note
    elements.append(Spacer(1, 2*cm))
    
    footer_style = ParagraphStyle(
        'DiscussionFooter',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#95A5A6'),
        spaceAfter=0,
        fontName='Helvetica',
        leading=10,
        alignment=TA_CENTER
    )
    
    elements.append(Paragraph(
        "Gespreksdocument - Voor intern gebruik tijdens mediation en overleg<br/>"
        "Opgesteld met BeNice AI-assistent",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("📋 Voortgang")
    total, done, next_q = compute_progress()
    st.progress(done / total if total else 0.0, text=f"{done}/{total} voltooid")

    # Compact status table
    simple_rows = [
        {
            "Pos": q["position"],
            "ID": q["id"],
            "Status": q.get("status", ""),
            "Vraag": q["question"][:50] + ("…" if len(q["question"]) > 50 else "")
        }
        for q in st.session_state.question_list
    ]
    st.caption("Overzicht vragen")
    st.dataframe(simple_rows, hide_index=True, use_container_width=True)

    # Export & Reset
    st.divider()
    dl = json.dumps(export_payload(), ensure_ascii=False, indent=2)
    
    # Save to conversations directory and offer download
    conversations_dir = get_conversations_dir()
    timestamp = int(time.time())
    file_path = os.path.join(conversations_dir, f"conversation_{timestamp}.json")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download JSON",
            data=dl.encode("utf-8"),
            file_name=f"conversation_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        if st.button("💾 Opslaan", use_container_width=True):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(dl)
            st.success(f"Opgeslagen in conversations/")
    
    st.caption(f"Bestanden worden opgeslagen in: `{os.path.basename(conversations_dir)}/`")
    
    # PDF Export
    st.divider()
    st.subheader("📄 PDF Export")
    completed_count = sum(1 for q in st.session_state.question_list if q.get("status") == "completed")
    st.caption(f"{completed_count} van {total} vragen beantwoord")
    
    if completed_count > 0:
        st.markdown("**Formeel Tekenbaar Document**")
        st.caption("Juridisch en professioneel - klaar voor ondertekening")
        try:
            formal_pdf = generate_formal_pdf()
            st.download_button(
                "� Download Formeel Ouderschapsplan",
                data=formal_pdf,
                file_name=f"ouderschapsplan_formeel_{timestamp}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="formal_pdf"
            )
        except Exception as e:
            st.error(f"Fout bij genereren formeel PDF: {e}")
        
        st.markdown("---")
        
        st.markdown("**Gespreksdocument voor Mediation**")
        st.caption("Informeel en toegankelijk - voor overleg en bespreking")
        try:
            discussion_pdf = generate_discussion_pdf()
            st.download_button(
                "💬 Download Gespreksdocument",
                data=discussion_pdf,
                file_name=f"gespreksdocument_{timestamp}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="discussion_pdf"
            )
        except Exception as e:
            st.error(f"Fout bij genereren gespreksdocument: {e}")
    else:
        st.info("Beantwoord eerst minimaal één vraag om een PDF te kunnen genereren.")
    if st.button("🔁 Reset gesprek", use_container_width=True):
        st.session_state.history = []
        st.session_state.question_list = json.loads(json.dumps(question_list_template, ensure_ascii=False))
        st.session_state.initialized = False
        st.rerun()

# ---------- MAIN AREA ----------
st.title("👪 Ouderschapsplan Assistent")
st.caption("Begrijpelijk overzicht van het gesprek, met voortgang per vraag.")

# Auto-initialize with assistant’s first prompt
if not st.session_state.initialized:
    with st.spinner("Assistent start op…"):
        initial = send_to_gemini("Hallo, ik ben klaar om te beginnen met de vragenlijst.")
    st.session_state.history.append({"role": "user", "content": "Hallo, ik ben klaar om te beginnen met de vragenlijst."})
    st.session_state.history.append({"role": "assistant", "content": initial.get("answer", "")})
    merge_updates(initial.get("updated_questions") or initial.get("question_list"))
    st.session_state.initialized = True

# Render chat history
for msg in st.session_state.history:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.markdown(msg["content"])

# Hint about what’s next
_, _, next_q = compute_progress()
if next_q:
    st.info(f"**Volgende vraag** ({next_q['position']}): {next_q['question']}")

# Chat input
user_text = st.chat_input("Typ je antwoord of vraag…")
if user_text:
    # Append user message
    st.session_state.history.append({"role": "user", "content": user_text})

    # Remember which question was being worked on before this interaction
    _, _, previous_pending_q = compute_progress()
    previous_pending_id = previous_pending_q["id"] if previous_pending_q else None

    # Call model
    with st.chat_message("assistant"):
        with st.spinner("Denken…"):
            try:
                out = send_to_gemini(user_text)
            except Exception as e:
                st.error(f"Fout bij model: {e}")
                st.stop()
            answer = out.get("answer", "")
            st.markdown(answer)

    # Save assistant message and merge question updates
    st.session_state.history.append({"role": "assistant", "content": answer})
    merge_updates(out.get("updated_questions") or out.get("question_list"))

    # VALIDATION: Ensure question progression is tracked properly
    # If the model moved to a new question, ensure the previous one is marked completed
    _, _, new_pending_q = compute_progress()
    new_pending_id = new_pending_q["id"] if new_pending_q else None
    
    if previous_pending_id and new_pending_id and previous_pending_id != new_pending_id:
        # The model moved to a different question - ensure the previous one is completed
        for q in st.session_state.question_list:
            if q["id"] == previous_pending_id and q.get("status") != "completed":
                # Model forgot to mark it completed - fix it now
                q["status"] = "completed"
                # If answer/summary are still missing, try to extract from conversation
                if not q.get("answer"):
                    # Use the user's last input as the answer
                    q["answer"] = user_text
                if not q.get("summary") and out.get("summary"):
                    q["summary"] = out.get("summary")
                print(f"[AUTO-FIX] Question {previous_pending_id} was answered but not marked completed. Fixed automatically.")
                break

    # Rerun so sidebar/progress refresh immediately
    st.rerun()
