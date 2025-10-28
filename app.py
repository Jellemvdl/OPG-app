#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for the 'Ouderschapsplan' conversational assistant.
- Vertex AI (Gemini) via google-genai (vertex mode)
- Chat interface + progress sidebar
- Streamlit Cloud-only auth: secrets -> /tmp/gcp_sa.json (no .env)
"""

import os
import json
import time
from typing import List, Dict, Any
from io import BytesIO

import streamlit as st

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Ouderschapsplan Assistent", layout="wide")

# ---------- GCP AUTH (Streamlit Secrets only) ----------
# We force creds to come from Streamlit secrets and write to /tmp.
# This avoids relying on any local .env or repo files.
def _bootstrap_gcp_from_secrets() -> Dict[str, str]:
    missing = []
    for k in ["PROJECT_ID", "LOCATION", "MODEL_ID", "GCP_SA_KEY_JSON"]:
        if k not in st.secrets:
            missing.append(k)
    if missing:
        st.error(f"Missing Streamlit secrets: {', '.join(missing)}")
        st.stop()

    project_id = st.secrets["PROJECT_ID"]
    location   = st.secrets["LOCATION"]
    model_id   = st.secrets["MODEL_ID"]
    sa_json    = st.secrets["GCP_SA_KEY_JSON"]

    # Write JSON to /tmp and set GOOGLE_APPLICATION_CREDENTIALS
    sa_path = "/tmp/gcp_sa.json"
    try:
        # Validate JSON is well-formed
        sa_obj = json.loads(sa_json)
    except Exception as e:
        st.error(f"GCP_SA_KEY_JSON is not valid JSON: {e}")
        st.stop()

    with open(sa_path, "w") as f:
        json.dump(sa_obj, f)

    # Force the environment variable to point to the temp file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    # Also clear any old/incorrect values that might be set by the environment
    # (prevents the library from trying to use a non-existent file)
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        if var in os.environ and not os.environ[var]:
            del os.environ[var]

    return {
        "PROJECT_ID": project_id,
        "LOCATION": location,
        "MODEL_ID": model_id,
        "SA_PATH": sa_path,
    }

AUTH = _bootstrap_gcp_from_secrets()

# ---------- LOAD QUESTIONS ----------
def load_questions() -> List[Dict[str, Any]]:
    """Load questions from the questions/questions.json file (bundled in repo)."""
    questions_path = os.path.join(os.path.dirname(__file__), "questions", "questions.json")
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found at: {questions_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in questions file: {e}")

# ---------- QUESTION LIST ----------
question_list_template: List[Dict[str, Any]] = load_questions()

# ---------- SYSTEM INSTRUCTION ----------
# Raw string to avoid \s escape warnings inside regex-like text.

system_instruction = st.secrets.get("SYSTEM_INSTRUCTION")
if not system_instruction and "SYSTEM_INSTRUCTION_B64" in st.secrets:
    system_instruction = base64.b64decode(st.secrets["SYSTEM_INSTRUCTION_B64"]).decode("utf-8")

if not system_instruction:
    st.error("Missing SYSTEM_INSTRUCTION in secrets.")
    st.stop()
# ---------- STREAMLIT STATE ----------
if "question_list" not in st.session_state:
    st.session_state.question_list = json.loads(json.dumps(question_list_template, ensure_ascii=False))

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ---------- CLIENT (cached) ----------
@st.cache_resource(show_spinner=False)
def get_client():
    # Ensure GOOGLE_APPLICATION_CREDENTIALS still points to our /tmp path
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") != AUTH["SA_PATH"]:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = AUTH["SA_PATH"]

    # Import here to avoid premature initialization before creds exist
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=AUTH["PROJECT_ID"],
        location=AUTH["LOCATION"],
        http_options=types.HttpOptions(api_version="v1"),
    )
    return client

# ---------- MODEL CALL ----------
def send_to_gemini(user_input: str, sender_id: str = "Sandra") -> Dict[str, Any]:
    from google.genai import types  # local import keeps module load inside app

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
        max_output_tokens=2048,
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model=AUTH["MODEL_ID"],
        contents=contents,
        config=cfg,
    )
    raw_text = resp.text or "{}"

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: try to salvage the "answer" field
        import re
        m = re.search(r'"answer"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', raw_text)
        if m:
            return {"answer": m.group(1).replace('\\"', '"'), "updated_questions": None}
        raise

# ---------- HELPERS ----------
def merge_updates(updated_questions: List[Dict[str, Any]]):
    if not updated_questions:
        return
    qmap = {q["id"]: q for q in st.session_state.question_list}
    for uq in updated_questions:
        if uq.get("id") in qmap:
            qmap[uq["id"]] = uq
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
    """Get or create the conversations directory (ephemeral in Streamlit Cloud)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversations_dir = os.path.join(script_dir, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)
    return conversations_dir

# ---------- PDF GENERATORS ----------
def generate_formal_pdf() -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )

    elements = []
    styles = getSampleStyleSheet()

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
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=6,
        fontName='Helvetica-Bold',
        leading=14
    )
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
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=10,
        fontName='Helvetica',
        leading=12
    )

    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("OUDERSCHAPSPLAN", title_style))
    elements.append(Paragraph("Afspraken over de zorg en opvoeding van de kinderen", subtitle_style))

    line_table = Table([['']], colWidths=[16*cm])
    line_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#BDC3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(line_table)

    date_text = f"Datum opstelling: {time.strftime('%d %B %Y')}"
    elements.append(Paragraph(date_text, info_style))
    elements.append(Spacer(1, 0.8*cm))

    completed_questions = [q for q in st.session_state.question_list if q.get("status") == "completed"]

    if not completed_questions:
        elements.append(Paragraph("Nog geen vragen beantwoord.", answer_style))
    else:
        section_mapping = {
            1: "1. Gegevens ouders en kinderen",
            2: "1. Gegevens ouders en kinderen",
            3: "2. Uitgangspunten",
            4: "3. Principes en afspraken",
            5: "4. Zorgregeling",
            6: "4. Zorgregeling",
            7: "4. Zorgregeling",
            8: "4. Zorgregeling",
            9: "4. Zorgregeling",
            10: "5. Praktische afspraken",
            11: "5. Praktische afspraken",
            12: "6. Activiteiten en ontwikkeling",
            13: "6. Activiteiten en ontwikkeling",
            14: "7. Gezondheid en zorg",
            15: "7. Gezondheid en zorg",
            16: "8. Opvoeding",
            17: "9. Opvang en kinderopvang",
            18: "10. Verhuizen",
            19: "11. Financiële afspraken",
            20: "11. Financiële afspraken"
        }

        current_section = None
        for question in completed_questions:
            position = question.get('position', 0)
            section_name = section_mapping.get(position, "")

            if section_name and section_name != current_section:
                current_section = section_name
                elements.append(Spacer(1, 0.3*cm))
                elements.append(Paragraph(section_name, section_style))
                section_line = Table([['']], colWidths=[16*cm])
                section_line.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#E8EBED')),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(section_line)

            question_text = question['question']
            elements.append(Paragraph(question_text, question_style))

            answer_text = question.get('summary') or question.get('answer', 'Geen antwoord')
            answer_clean = answer_text.replace('\n', '<br/>')
            elements.append(Paragraph(answer_clean, answer_style))

    elements.append(PageBreak())

    elements.append(Spacer(1, 1*cm))
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

    elements.append(line_table)
    elements.append(Spacer(1, 0.3*cm))

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
    elements.append(Spacer(1, 1.5*cm))

    sig_field_style = ParagraphStyle(
        'SignatureField',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=8,
        fontName='Helvetica',
        leading=14
    )
    sig_label_style = ParagraphStyle(
        'SignatureLabel',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1A5490'),
        spaceAfter=12,
        fontName='Helvetica-Bold',
        leading=14
    )

    elements.append(Paragraph("Ouder 1", sig_label_style))
    elements.append(Paragraph("Naam: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.8*cm))
    elements.append(Paragraph("Handtekening: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.8*cm))
    elements.append(Paragraph("Datum: _________________________________________________________________", sig_field_style))

    elements.append(Spacer(1, 1.5*cm))

    elements.append(Paragraph("Ouder 2", sig_label_style))
    elements.append(Paragraph("Naam: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.8*cm))
    elements.append(Paragraph("Handtekening: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.8*cm))
    elements.append(Paragraph("Datum: _________________________________________________________________", sig_field_style))
    elements.append(Spacer(1, 0.8*cm))

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
        "Dit ouderschapsplan is opgesteld met behulp van de BeNice AI-assistent",
        footer_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def generate_discussion_pdf() -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )

    elements = []
    styles = getSampleStyleSheet()

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
    question_style = ParagraphStyle(
        'DiscussionQuestion',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=5,
        fontName='Helvetica-Bold',
        leading=13
    )
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
    note_style = ParagraphStyle(
        'NoteStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#95A5A6'),
        spaceAfter=10,
        fontName='Helvetica-Oblique',
        leading=12
    )

    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph("Gespreksdocument Ouderschapsplan", title_style))
    elements.append(Paragraph(
        "Een informeel overzicht van de afspraken en gedachten van beide ouders - voor gebruik tijdens mediation of overleg",
        subtitle_style
    ))

    date_text = f"Gespreksnotitie van {time.strftime('%d %B %Y')}"
    elements.append(Paragraph(date_text, note_style))
    elements.append(Spacer(1, 0.5*cm))

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

    completed_questions = [q for q in st.session_state.question_list if q.get("status") == "completed"]

    if not completed_questions:
        elements.append(Paragraph("Nog geen gespreksonderwerpen besproken.", answer_style))
    else:
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
        for question in completed_questions:
            position = question.get('position', 0)
            section_name = section_mapping.get(position, "")

            if section_name and section_name != current_section:
                current_section = section_name
                elements.append(Spacer(1, 0.3*cm))
                elements.append(Paragraph(section_name, section_style))
                separator = Table([['']], colWidths=[16*cm])
                separator.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#F39C12')),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(separator)

            question_text = f"• {question['question']}"
            elements.append(Paragraph(question_text, question_style))

            answer_text = question.get('answer', 'Geen antwoord gegeven')
            answer_clean = answer_text.replace('\n', '<br/>')
            if any(word in answer_text.lower() for word in ['jullie', 'we', 'wij', 'ons', 'onze']):
                answer_display = f'<i>"{answer_clean}"</i>'
            else:
                answer_display = answer_clean
            elements.append(Paragraph(answer_display, answer_style))

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
    # Replace deprecated use_container_width with width='stretch'
    st.dataframe(simple_rows, hide_index=True, width='stretch')

    # Export & Reset
    st.divider()
    dl = json.dumps(export_payload(), ensure_ascii=False, indent=2)

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
            width='stretch',
        )
    with col2:
        if st.button("💾 Opslaan", width='stretch'):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(dl)
            st.success("Opgeslagen in conversations/")

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
                "📝 Download Formeel Ouderschapsplan",
                data=formal_pdf,
                file_name=f"ouderschapsplan_formeel_{timestamp}.pdf",
                mime="application/pdf",
                width='stretch',
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
                width='stretch',
                key="discussion_pdf"
            )
        except Exception as e:
            st.error(f"Fout bij genereren gespreksdocument: {e}")
    else:
        st.info("Beantwoord eerst minimaal één vraag om een PDF te kunnen genereren.")

    if st.button("🔁 Reset gesprek", width='stretch'):
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

# Chat input
user_text = st.chat_input("Typ je antwoord of vraag…")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})

    with st.chat_message("assistant"):
        with st.spinner("Denken…"):
            try:
                out = send_to_gemini(user_text)
            except Exception as e:
                st.error(f"Fout bij model: {e}")
                st.stop()
            answer = out.get("answer", "")
            st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
    merge_updates(out.get("updated_questions") or out.get("question_list"))
    st.rerun()
