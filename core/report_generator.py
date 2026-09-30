import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from core.image_utils import get_image_hash

def generate_pdf_report(
    img_a_pil, 
    img_b_pil_or_video_crops, 
    crop_a_pil, 
    crop_b_pil, 
    metrics, 
    procedure_id="CONFRONTO-BIOMÉTRICO", 
    expert_name="Analista Biométrico", 
    notes="",
    video_persons=None,
    matched_pairs=None
):
    """
    Gera o relatório PDF com parecer pericial e a seção final 'RECURSOS TÉCNICOS UTILIZADOS'.
    Suporta confrontos 1x1, vídeos de CFTV e fotos estáticas multi-faciais (N x M).
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter,
        rightMargin=36, 
        leftMargin=36, 
        topMargin=36, 
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=1,
        textColor=colors.HexColor('#1a252f')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#4A5568')
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1a252f'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2d3748')
    )

    alert_style = ParagraphStyle(
        'AlertBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        alignment=1,
        textColor=colors.HexColor('#dc3545')
    )

    story = []

    # Cabeçalho Principal
    if video_persons is not None:
        report_title = "RELATÓRIO DE CONFRONTO FACIAL BIOMÉTRICO (VÍDEO DE CFTV)"
    elif matched_pairs is not None:
        report_title = "RELATÓRIO DE CONFRONTO FACIAL MULTI-AMBO (FOTOS ESTÁTICAS)"
    else:
        report_title = "RELATÓRIO DE CONFRONTO FACIAL BIOMÉTRICO"

    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a252f'), spaceAfter=10))

    # Metadados
    now_str = datetime.datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
    hash_a = get_image_hash(img_a_pil)[:16] + "..."

    meta_data = [
        [Paragraph("<b>Identificador da Comparação:</b>", body_style), Paragraph(procedure_id, body_style)],
        [Paragraph("<b>Data/Hora do Processamento:</b>", body_style), Paragraph(now_str, body_style)],
        [Paragraph("<b>Hash SHA-256 (Foto A - Referência):</b>", body_style), Paragraph(hash_a, body_style)],
    ]
    
    if video_persons is not None:
        meta_data.append([Paragraph("<b>Origem da Mídia B:</b>", body_style), Paragraph(f"Vídeo de Câmera (CFTV) - <b>{len(video_persons)} indivíduos analisados</b>", body_style)])
    elif matched_pairs is not None:
        meta_data.append([Paragraph("<b>Tipo de Análise:</b>", body_style), Paragraph(f"Varredura Multi-Facial Matricial (N x M) - <b>{len(matched_pairs)} par(es) compatível(is)</b>", body_style)])

    t_meta = Table(meta_data, colWidths=[180, 360])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    def pil_to_rl_image(pil_img, max_w=110, max_h=110):
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return RLImage(buf, width=max_w, height=max_h)

    # 1. REGISTRO FOTOGRÁFICO
    story.append(Paragraph("1. REGISTRO FOTOGRÁFICO E RECORTE BIOMÉTRICO", section_style))

    if video_persons is not None:
        compatible_persons = [p for p in video_persons if (p['metrics']['status_code'] in ['MATCH', 'INCONCLUSIVE'] or p['metrics']['cosine_sim'] >= 0.45)]
        
        img_a_rl = pil_to_rl_image(crop_a_pil if crop_a_pil else img_a_pil, 120, 120)
        ref_table = Table([
            [Paragraph("<b>FOTO A (Referência Oficial / Banco de Dados)</b>", subtitle_style)],
            [img_a_rl]
        ], colWidths=[540])
        ref_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(ref_table)
        story.append(Spacer(1, 10))
        
        if len(compatible_persons) > 0:
            story.append(Paragraph("<b>ROSTOS COMPATÍVEIS IDENTIFICADOS NO VÍDEO:</b>", subtitle_style))
            story.append(Spacer(1, 4))
            
            for p in compatible_persons:
                p_crop_rl = pil_to_rl_image(p['crop_hud'], 95, 95)
                p_status = f"<font color='{p['metrics']['color']}'><b>{p['metrics']['classification']}</b></font>"
                p_info = Paragraph(
                    f"<b>{p['person_id']}</b> (Capturado em {p['timestamp_str']})<br/>"
                    f"Similaridade de Cosseno: <b>{p['metrics']['cosine_sim']:.4f}</b> | Distância L2: <b>{p['metrics']['euclidean_dist']:.4f}</b><br/>"
                    f"Grau de Certeza Estimado: <b><font color='{p['metrics']['color']}'>{p['metrics']['certainty_pct']:.1f}%</font></b><br/>"
                    f"Resultado: {p_status}<br/>"
                    f"<font size=8 color='#4a5568'>{p['metrics']['reason']}</font>",
                    body_style
                )
                
                p_table = Table([[p_crop_rl, p_info]], colWidths=[110, 430])
                p_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(p_table)
                story.append(Spacer(1, 6))
        else:
            no_match_box = Table([[Paragraph("⚠️ NENHUM ROSTO COMPATÍVEL ENCONTRADO NO VÍDEO.", alert_style)]], colWidths=[540])
            no_match_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff5f5')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#feb2b2')),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(no_match_box)
            story.append(Spacer(1, 10))

    elif matched_pairs is not None:
        if len(matched_pairs) > 0:
            for idx, pair in enumerate(matched_pairs, 1):
                fa, fb = pair['face_a'], pair['face_b']
                m = pair['metrics']
                
                img_fa_rl = pil_to_rl_image(fa['crop_hud'], 105, 105)
                img_fb_rl = pil_to_rl_image(fb['crop_hud'], 105, 105)
                
                p_status = f"<font color='{m['color']}'><b>{m['classification']}</b></font>"
                p_info = Paragraph(
                    f"<b>PAR #{idx}: {fa['label']} (Foto A) ↔ {fb['label']} (Foto B)</b><br/>"
                    f"Similaridade de Cosseno: <b>{m['cosine_sim']:.4f}</b> | Distância L2: <b>{m['euclidean_dist']:.4f}</b><br/>"
                    f"Grau de Certeza Estimado: <b><font color='{m['color']}'>{m['certainty_pct']:.1f}%</font></b><br/>"
                    f"Resultado: {p_status}<br/>"
                    f"<font size=8 color='#4a5568'>{m['reason']}</font>",
                    body_style
                )
                
                pair_table = Table([[img_fa_rl, img_fb_rl, p_info]], colWidths=[115, 115, 310])
                pair_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(pair_table)
                story.append(Spacer(1, 6))
        else:
            no_match_box = Table([[Paragraph("⚠️ NENHUM ROSTO COMPATÍVEL ENCONTRADO ENTRE AS DUAS FOTOS.", alert_style)]], colWidths=[540])
            no_match_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff5f5')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#feb2b2')),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(no_match_box)
            story.append(Spacer(1, 10))

    else:
        img_a_rl = pil_to_rl_image(crop_a_pil if crop_a_pil else img_a_pil, 140, 140)
        img_b_rl = pil_to_rl_image(crop_b_pil if crop_b_pil else img_b_pil_or_video_crops, 140, 140)

        img_data = [
            [Paragraph("<b>FOTO A (Referência Oficial)</b>", subtitle_style), Paragraph("<b>FOTO B (Questionada / CFTV)</b>", subtitle_style)],
            [img_a_rl, img_b_rl]
        ]

        t_img = Table(img_data, colWidths=[270, 270])
        t_img.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
        ]))
        story.append(t_img)
        
    story.append(Spacer(1, 10))

    # 2. ANÁLISE BIOMÉTRICA E MÉTRICAS
    story.append(Paragraph("2. ANÁLISE BIOMÉTRICA E MÉTRICAS DE CONVERGÊNCIA", section_style))

    if matched_pairs is not None and len(matched_pairs) > 0:
        for idx, pair in enumerate(matched_pairs, 1):
            m = pair['metrics']
            status_text = f"<b><font size=11 color='{m['color']}'>{m['classification']}</font></b>"

            metrics_table_data = [
                [Paragraph("<b>Par Biométrico Analisado:</b>", body_style), Paragraph(f"<b>PAR #{idx}: {pair['face_a']['label']} (Foto A) ↔ {pair['face_b']['label']} (Foto B)</b>", body_style)],
                [Paragraph("<b>Algoritmo Extrator:</b>", body_style), Paragraph("Deep FaceNet (InceptionResnetV1 512d - VGGFace2 com Multi-crop)", body_style)],
                [Paragraph("<b>Similaridade de Cosseno:</b>", body_style), Paragraph(f"<b>{m['cosine_sim']:.4f}</b>", body_style)],
                [Paragraph("<b>Distância Euclidiana (L2):</b>", body_style), Paragraph(f"<b>{m['euclidean_dist']:.4f}</b>", body_style)],
                [Paragraph("<b>Grau de Certeza Estimado:</b>", body_style), Paragraph(f"<b><font size=10 color='{m['color']}'>{m['certainty_pct']:.1f}%</font></b>", body_style)],
                [Paragraph("<b>Resultado da Análise:</b>", body_style), Paragraph(status_text, body_style)],
                [Paragraph("<b>Fundamentação:</b>", body_style), Paragraph(m['reason'], body_style)],
            ]

            t_metrics = Table(metrics_table_data, colWidths=[180, 360])
            t_metrics.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t_metrics)
            story.append(Spacer(1, 8))

    elif video_persons is not None and len(video_persons) > 0:
        compatible_persons_list = [p for p in video_persons if (p['metrics']['status_code'] in ['MATCH', 'INCONCLUSIVE'] or p['metrics']['cosine_sim'] >= 0.45)]
        target_list = compatible_persons_list if len(compatible_persons_list) > 0 else video_persons
        
        for p in target_list:
            m = p['metrics']
            status_text = f"<b><font size=11 color='{m['color']}'>{m['classification']}</font></b>"

            metrics_table_data = [
                [Paragraph("<b>Indivíduo Analisado:</b>", body_style), Paragraph(f"<b>{p['person_id']}</b> ({p['timestamp_str']})", body_style)],
                [Paragraph("<b>Algoritmo Extrator:</b>", body_style), Paragraph("Deep FaceNet (InceptionResnetV1 512d - VGGFace2 com Multi-crop)", body_style)],
                [Paragraph("<b>Similaridade de Cosseno:</b>", body_style), Paragraph(f"<b>{m['cosine_sim']:.4f}</b>", body_style)],
                [Paragraph("<b>Distância Euclidiana (L2):</b>", body_style), Paragraph(f"<b>{m['euclidean_dist']:.4f}</b>", body_style)],
                [Paragraph("<b>Grau de Certeza Estimado:</b>", body_style), Paragraph(f"<b><font size=10 color='{m['color']}'>{m['certainty_pct']:.1f}%</font></b>", body_style)],
                [Paragraph("<b>Resultado da Análise:</b>", body_style), Paragraph(status_text, body_style)],
                [Paragraph("<b>Fundamentação:</b>", body_style), Paragraph(m['reason'], body_style)],
            ]

            t_metrics = Table(metrics_table_data, colWidths=[180, 360])
            t_metrics.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t_metrics)
            story.append(Spacer(1, 8))

    else:
        res_color = colors.HexColor(metrics['color'])
        status_text = f"<b><font size=11 color='{metrics['color']}'>{metrics['classification']}</font></b>"

        metrics_table_data = [
            [Paragraph("<b>Algoritmo Extrator:</b>", body_style), Paragraph("Deep FaceNet (InceptionResnetV1 512d - VGGFace2 com Multi-crop)", body_style)],
            [Paragraph("<b>Similaridade de Cosseno:</b>", body_style), Paragraph(f"<b>{metrics['cosine_sim']:.4f}</b>", body_style)],
            [Paragraph("<b>Distância Euclidiana (L2):</b>", body_style), Paragraph(f"<b>{metrics['euclidean_dist']:.4f}</b>", body_style)],
            [Paragraph("<b>Grau de Certeza Estimado:</b>", body_style), Paragraph(f"<b><font size=10 color='{metrics['color']}'>{metrics['certainty_pct']:.1f}%</font></b>", body_style)],
            [Paragraph("<b>Resultado da Análise:</b>", body_style), Paragraph(status_text, body_style)],
            [Paragraph("<b>Fundamentação:</b>", body_style), Paragraph(metrics['reason'], body_style)],
        ]

        t_metrics = Table(metrics_table_data, colWidths=[180, 360])
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 10))

    # 3. PARECER TÉCNICO E RECURSOS TÉCNICOS UTILIZADOS
    story.append(Paragraph("3. PARECER TÉCNICO PERICIAL / OBSERVAÇÕES", section_style))
    notes_content = notes if notes.strip() else "Nenhuma observação complementar inserida."
    
    tech_resources_pdf = (
        "<b>RECURSOS TÉCNICOS UTILIZADOS:</b><br/>"
        "• <b>Extrator Biométrico:</b> Rede Neural Profunda FaceNet (InceptionResnetV1) de 512 dimensões pré-treinada na base VGGFace2 com vetorização L2 e média multi-crop Flip-Invariance.<br/>"
        "• <b>Detector e Alinhador Anatômico:</b> MTCNN (Multi-task Cascaded Convolutional Networks) com localização de 5 pontos biométricos chave (olhos, nariz e cantos da boca).<br/>"
        "• <b>Métricas de Convergência:</b> Produto Escalar Normalizado de Cosseno (Espaço Vetorial 512D) e Distância Euclidiana L2 com calibração sigmoidal pericial.<br/>"
        "• <b>Filtros de Qualidade e Integridade:</b> Avaliação de nitidez facial por variância do operador Laplaciano (Laplacian Blur Score) e alinhamento geométrico pupilar."
    )
    
    full_notes_pdf = f"{notes_content}<br/><br/>{tech_resources_pdf}"
    
    t_notes = Table([[Paragraph(full_notes_pdf, body_style)]], colWidths=[540])
    t_notes.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_notes)

    doc.build(story)
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
