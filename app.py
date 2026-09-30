import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

from core.face_engine import (
    process_face_image, 
    get_face_embedding, 
    compare_embeddings, 
    process_video_file
)
from core.image_utils import adjust_image, blend_images
from core.report_generator import generate_pdf_report

# Configuração da Página do Streamlit
st.set_page_config(
    page_title="Comparador Facial Biométrico",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

def reset_comparison():
    st.session_state['uploader_key'] += 1
    st.rerun()

# Estilização CSS Minimalista e Responsiva (Desktop + Celular / Mobile)
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    header[data-testid="stHeader"] {
        background: transparent;
    }

    .clean-section-title {
        font-size: 14px;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 24px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .image-card-header {
        background-color: #182234;
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 8px 8px 0 0;
        border: 1px solid #1e293b;
        border-bottom: none;
        text-align: center;
    }

    .minimal-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #111827;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1f2937;
        margin-top: 8px;
    }
    .minimal-table tr {
        border-bottom: 1px solid #1f2937;
    }
    .minimal-table tr:last-child {
        border-bottom: none;
    }
    .minimal-table td {
        padding: 12px 18px;
        font-size: 14px;
        color: #cbd5e1;
    }
    .minimal-table td.col-label {
        font-weight: 600;
        width: 260px;
        color: #64748b;
        background-color: #141c2e;
    }

    .minimal-parecer {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 14px;
        line-height: 1.6;
        color: #cbd5e1;
        margin-top: 8px;
    }

    .person-card {
        background-color: #111827;
        border-radius: 8px;
        border: 1px solid #1f2937;
        padding: 8px;
        text-align: center;
    }

    /* 📱 ADAPTAÇÃO RESPONSIVA PARA DISPOSITIVOS MÓVEIS (CELULARES E TABLETS) */
    @media (max-width: 768px) {
        .minimal-table td {
            padding: 8px 10px !important;
            font-size: 12px !important;
        }
        .minimal-table td.col-label {
            width: 130px !important;
            font-size: 11px !important;
        }
        .clean-section-title {
            font-size: 12px !important;
            margin-top: 18px !important;
        }
        .minimal-parecer {
            padding: 12px 14px !important;
            font-size: 13px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v3.4.0"

# Top Bar com Botão de Reset e Badge de Versão
col_head1, col_head2 = st.columns([3, 1])

with col_head1:
    st.markdown(f"""
    <div style="padding-top: 4px;">
        <div style="font-size: 20px; font-weight: 700; color: #f8fafc; letter-spacing: -0.5px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            🛡️ Comparador Facial Biométrico
            <span style="font-size: 11px; background-color: #0f172a; color: #38bdf8; border: 1px solid #0284c7; padding: 2px 9px; border-radius: 12px; font-weight: 700; letter-spacing: 0.5px;">{APP_VERSION}</span>
        </div>
        <div style="font-size: 12px; color: #64748b; margin-top: 2px;">
            Confronto biométrico automatizado por IA com varredura de vídeos de CFTV e fotos
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_head2:
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Nova Comparação", use_container_width=True):
        reset_comparison()

st.markdown("---")

# Sidebar Opcional para Ajustes Avançados
with st.sidebar:
    st.title("🎛️ Configurações")
    st.caption(f"Versão do Sistema: **{APP_VERSION}**")
    threshold_mode = st.selectbox(
        "Limiar de Rigor:",
        options=['rigoroso', 'padrao', 'permissivo'],
        format_func=lambda x: {
            'rigoroso': "🔒 Rigoroso (80%+)",
            'padrao':   "⚖️ Padrão (70%+)",
            'permissivo': "🔍 Permissivo (60%+)"
        }[x],
        index=1
    )
    sample_rate = st.slider("Amostragem de Frames em Vídeo:", 1, 10, 2, help="Processa 1 frame a cada N frames do vídeo.")
    min_blur = st.slider("Filtro Anti-Desfoque (Nitidez Mínima):", 1, 50, 8, help="Filtra e ignora rostos desfocados no vídeo.")

# Seção de Upload com Ícones de Mídia Enviada
key_suffix = st.session_state['uploader_key']
col_u1, col_u2 = st.columns(2)

with col_u1:
    st.markdown("<div style='font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 6px;'>FOTO A (Referência Oficial / Suspeito)</div>", unsafe_allow_html=True)
    uploaded_a = st.file_uploader("Upload da Foto A", type=['jpg', 'jpeg', 'png', 'webp', 'bmp'], key=f"img_a_{key_suffix}", label_visibility="collapsed")
    
    if uploaded_a is not None:
        try:
            img_preview_a = Image.open(uploaded_a)
            size_mb = uploaded_a.size / (1024 * 1024)
            col_icon_a1, col_icon_a2 = st.columns([1, 4])
            with col_icon_a1:
                st.image(img_preview_a, width=54)
            with col_icon_a2:
                st.markdown(f"<div style='font-size: 11px; color: #10b981; font-weight: 600; margin-top: 8px;'>📷 Foto A Carregada</div><div style='font-size: 11px; color: #64748b;'>{uploaded_a.name} ({size_mb:.1f} MB)</div>", unsafe_allow_html=True)
        except Exception:
            pass

with col_u2:
    st.markdown("<div style='font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 6px;'>MÍDIA B (Questionada: Foto ou Vídeo de CFTV)</div>", unsafe_allow_html=True)
    uploaded_b = st.file_uploader("Upload da Mídia B (Foto ou Vídeo)", type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'mp4', 'avi', 'mov', 'mkv'], key=f"media_b_{key_suffix}", label_visibility="collapsed")
    
    if uploaded_b is not None:
        size_mb_b = uploaded_b.size / (1024 * 1024)
        is_vid_b = uploaded_b.name.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        if is_vid_b:
            st.markdown(f"<div style='font-size: 11px; color: #38bdf8; font-weight: 600; margin-top: 8px;'>🎥 Vídeo B de CFTV Carregado</div><div style='font-size: 11px; color: #64748b;'>{uploaded_b.name} ({size_mb_b:.1f} MB)</div>", unsafe_allow_html=True)
        else:
            try:
                img_preview_b = Image.open(uploaded_b)
                col_icon_b1, col_icon_b2 = st.columns([1, 4])
                with col_icon_b1:
                    st.image(img_preview_b, width=54)
                with col_icon_b2:
                    st.markdown(f"<div style='font-size: 11px; color: #10b981; font-weight: 600; margin-top: 8px;'>📷 Foto B Carregada</div><div style='font-size: 11px; color: #64748b;'>{uploaded_b.name} ({size_mb_b:.1f} MB)</div>", unsafe_allow_html=True)
            except Exception:
                pass

if uploaded_a and uploaded_b:
    img_a = Image.open(uploaded_a)
    
    with st.spinner("Analisando feições da Foto de Referência..."):
        tensor_a, crop_a_hud, _, box_a, lmk_a, err_a = process_face_image(img_a, label="REFERÊNCIA_A", is_match=True)
        
    if err_a:
        st.error(f"Erro na Foto A: {err_a}")
    else:
        emb_a = get_face_embedding(tensor_a)
        is_video = uploaded_b.name.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        if is_video:
            # -------------------------------------------------------------
            # PROCESSAMENTO DE VÍDEO DE CFTV COM BARRA DE PROGRESSO EM %
            # -------------------------------------------------------------
            st.markdown(f"<div style='font-size: 13px; color: #38bdf8; margin-top: 10px;'>🎥 Analisando vídeo <b>{uploaded_b.name}</b>...</div>", unsafe_allow_html=True)
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(pct):
                pct_int = int(pct * 100)
                progress_bar.progress(pct)
                status_text.markdown(f"<span style='font-size: 12px; color: #94a3b8;'>Varredura de vídeo: <b>{pct_int}%</b> concluído</span>", unsafe_allow_html=True)

            video_bytes = uploaded_b.read()
            video_persons, vid_err = process_video_file(
                video_bytes, 
                emb_a, 
                sample_fps_step=sample_rate, 
                threshold_mode=threshold_mode, 
                min_blur_score=min_blur,
                progress_callback=update_progress
            )
            
            status_text.empty()
            progress_bar.empty()
                
            if vid_err:
                st.error(f"Erro no Vídeo: {vid_err}")
            elif len(video_persons) == 0:
                st.warning("Nenhum rosto com qualidade suficiente foi detectado no vídeo.")
            else:
                # Filtrar rostos com status COMPATÍVEL ou INCONCLUSIVO significativo (Cosseno >= 0.45 / Certeza >= 50%)
                compatible_persons = [p for p in video_persons if (p['metrics']['status_code'] in ['MATCH', 'INCONCLUSIVE'] or p['metrics']['cosine_sim'] >= 0.45)]
                
                if len(compatible_persons) == 0:
                    st.warning("⚠️ NENHUM ROSTO COMPATÍVEL ENCONTRADO NO VÍDEO.")
                    
                    # Parecer para quando não há nenhum rosto compatível no vídeo
                    parecer_text = (
                        f"Confronto facial realizado entre a Foto A (Referência Oficial) e o vídeo de CFTV ({uploaded_b.name}). "
                        f"Foram analisados {len(video_persons)} indivíduo(s) qualificado(s) no vídeo. "
                        f"NENHUM ROSTO COMPATÍVEL COM A FOTO DE REFERÊNCIA FOI ENCONTRADO."
                    )
                    
                    st.markdown('<div class="clean-section-title">3. PARECER TÉCNICO PERICIAL / OBSERVAÇÕES</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="minimal-parecer">
                        {parecer_text}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
                    with col_pdf2:
                        pdf_bytes = generate_pdf_report(
                            img_a, None, crop_a_hud, None, 
                            {'color': '#dc3545', 'classification': 'NENHUM COMPATÍVEL', 'cosine_sim': 0.0, 'euclidean_dist': 0.0, 'certainty_pct': 0.0, 'reason': 'Nenhum rosto compatível no vídeo.'},
                            procedure_id="CONFRONTO-CFTV-VIDEO", 
                            expert_name="Analista Biométrico", 
                            notes=parecer_text,
                            video_persons=[]
                        )
                        st.download_button(
                            label="📄 Baixar Relatório Completo do Vídeo (.PDF)",
                            data=pdf_bytes,
                            file_name=f"Relatorio_Confronto_Video_{uploaded_b.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                else:
                    st.success(f"🔍 **{len(compatible_persons)} rosto(s) compatível(is)** encontrado(s) no vídeo!")
                    
                    # 1. REGISTRO FOTOGRÁFICO (Exibir APENAS rostos COMPATÍVEIS)
                    st.markdown('<div class="clean-section-title">1. REGISTRO FOTOGRÁFICO E RECORTE BIOMÉTRICO (VÍDEO CFTV)</div>', unsafe_allow_html=True)
                    
                    col_va, col_vb = st.columns([1, 2.5])
                    with col_va:
                        st.markdown('<div class="image-card-header">FOTO A (Referência Oficial)</div>', unsafe_allow_html=True)
                        st.image(crop_a_hud, use_container_width=True)
                        
                    with col_vb:
                        st.markdown('<div class="image-card-header">ROSTO(S) COMPATÍVEL(IS) DETECTADO(S) NO VÍDEO</div>', unsafe_allow_html=True)
                        
                        cols_p = st.columns(min(3, len(compatible_persons)))
                        for idx, p in enumerate(compatible_persons):
                            c_target = cols_p[idx % min(3, len(compatible_persons))]
                            with c_target:
                                st.markdown(f"""
                                <div class="person-card">
                                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8;">{p['person_id']} ({p['timestamp_str']})</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.image(p['crop_hud'], use_container_width=True)
                                color_h = p['metrics']['color']
                                st.markdown(f"""
                                <div style="text-align: center; margin-top: 4px;">
                                    <span style="font-size: 13px; font-weight: 800; color: {color_h};">{p['metrics']['certainty_pct']:.1f}%</span><br/>
                                    <span style="font-size: 10px; font-weight: 600; color: {color_h};">{p['metrics']['classification']}</span>
                                </div>
                                """, unsafe_allow_html=True)

                    selected_person = compatible_persons[0] # Maior match compatível
                    res = selected_person['metrics']
                    
                    # 2. ANÁLISE BIOMÉTRICA E MÉTRICAS
                    st.markdown('<div class="clean-section-title">2. ANÁLISE BIOMÉTRICA E MÉTRICAS DE CONVERGÊNCIA</div>', unsafe_allow_html=True)
                    
                    color_hex = res['color']
                    certainty_str = f"{res['certainty_pct']:.1f}%"
                    status_html = f"<b style='color: {color_hex}; font-size: 14px;'>{res['classification']}</b>"
                    certainty_html = f"<b style='color: {color_hex}; font-size: 14px;'>{certainty_str}</b>"

                    table_html = f"""
                    <table class="minimal-table">
                        <tr>
                            <td class="col-label">Indivíduo Analisado:</td>
                            <td><b>{selected_person['person_id']}</b> ({selected_person['timestamp_str']})</td>
                        </tr>
                        <tr>
                            <td class="col-label">Algoritmo Extrator:</td>
                            <td>Deep FaceNet (InceptionResnetV1 512d - VGGFace2)</td>
                        </tr>
                        <tr>
                            <td class="col-label">Similaridade de Cosseno:</td>
                            <td><b>{res['cosine_sim']:.4f}</b></td>
                        </tr>
                        <tr>
                            <td class="col-label">Distância Euclidiana (L2):</td>
                            <td><b>{res['euclidean_dist']:.4f}</b></td>
                        </tr>
                        <tr>
                            <td class="col-label">Grau de Certeza Estima:</td>
                            <td>{certainty_html}</td>
                        </tr>
                        <tr>
                            <td class="col-label">Resultado da Análise:</td>
                            <td>{status_html}</td>
                        </tr>
                        <tr>
                            <td class="col-label">Fundamentação:</td>
                            <td>{res['reason']}</td>
                        </tr>
                    </table>
                    """
                    st.markdown(table_html, unsafe_allow_html=True)

                    # 3. PARECER TÉCNICO
                    st.markdown('<div class="clean-section-title">3. PARECER TÉCNICO PERICIAL / OBSERVAÇÕES</div>', unsafe_allow_html=True)
                    
                    parecer_text = (
                        f"Confronto facial realizado entre a Foto A (Referência Oficial) e o vídeo de CFTV ({uploaded_b.name}). "
                        f"Foram identificados {len(compatible_persons)} rosto(s) compatível(is) no vídeo. "
                        f"O maior grau de convergência foi no {selected_person['person_id']} ({selected_person['timestamp_str']}) "
                        f"com similaridade de cosseno de {res['cosine_sim']:.4f} ({res['certainty_pct']:.1f}% de certeza). "
                        f"Resultado conclusivo para {res['classification']}."
                    )

                    tech_resources_html = """
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #1f2937; font-size: 13px; color: #94a3b8;">
                        <b style="color: #e2e8f0;">🛠️ RECURSOS TÉCNICOS UTILIZADOS:</b><br/>
                        • <b>Extrator Biométrico:</b> Deep Learning FaceNet (InceptionResnetV1) de 512 dimensões pré-treinado em VGGFace2 com vetorização L2 e amostragem multi-crop Flip-Invariance.<br/>
                        • <b>Detector Anatômico:</b> MTCNN (Multi-task Cascaded Convolutional Networks) com localização de 5 pontos biométricos (olhos, nariz e cantos da boca).<br/>
                        • <b>Métricas de Convergência:</b> Similaridade de Cosseno no hiperespaço 512D e Distância Euclidiana L2 com modelo de calibração de certeza.<br/>
                        • <b>Filtros de Qualidade:</b> Avaliação de nitidez facial por variância do operador Laplaciano (Laplacian Blur Score) e validação de geometria anatômica.
                    </div>
                    """

                    st.markdown(f"""
                    <div class="minimal-parecer">
                        {parecer_text}
                        {tech_resources_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # BOTÃO DE PDF (INCLUI APENAS OS ROSTOS COMPATÍVEIS NO PDF)
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
                    with col_pdf2:
                        pdf_bytes = generate_pdf_report(
                            img_a, None, crop_a_hud, None, res, 
                            procedure_id="CONFRONTO-CFTV-VIDEO", 
                            expert_name="Analista Biométrico", 
                            notes=parecer_text,
                            video_persons=compatible_persons
                        )
                        st.download_button(
                            label="📄 Baixar Relatório Completo do Vídeo (.PDF)",
                            data=pdf_bytes,
                            file_name=f"Relatorio_Confronto_Video_{uploaded_b.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

        else:
            # -------------------------------------------------------------
            # PROCESSAMENTO DE IMAGEM ESTÁTICA 1 X 1
            # -------------------------------------------------------------
            img_b = Image.open(uploaded_b)
            with st.spinner("Analisando feições faciais da Foto B..."):
                tensor_b, crop_b_hud, _, box_b, lmk_b, err_b = process_face_image(img_b, label="QUESTIONADO_B", is_match=True)
                
            if err_b:
                st.error(f"Foto B: {err_b}")
            else:
                emb_b = get_face_embedding(tensor_b)
                res = compare_embeddings(emb_a, emb_b, threshold_mode=threshold_mode)
                
                is_match = (res['status_code'] == 'MATCH')
                
                _, crop_a_hud, _, _, _, _ = process_face_image(img_a, label="FOTO_A", is_match=is_match)
                _, crop_b_hud, _, _, _, _ = process_face_image(img_b, label="FOTO_B", is_match=is_match)

                # 1. REGISTRO FOTOGRÁFICO
                st.markdown('<div class="clean-section-title">1. REGISTRO FOTOGRÁFICO E RECORTE BIOMÉTRICO</div>', unsafe_allow_html=True)
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown('<div class="image-card-header">FOTO A (Referência / Suspeito)</div>', unsafe_allow_html=True)
                    st.image(crop_a_hud, use_container_width=True)
                    
                with col_r2:
                    st.markdown('<div class="image-card-header">FOTO B (Questionada / CFTV)</div>', unsafe_allow_html=True)
                    st.image(crop_b_hud, use_container_width=True)

                # 2. ANÁLISE BIOMÉTRICA
                st.markdown('<div class="clean-section-title">2. ANÁLISE BIOMÉTRICA E MÉTRICAS DE CONVERGÊNCIA</div>', unsafe_allow_html=True)
                
                color_hex = res['color']
                certainty_str = f"{res['certainty_pct']:.1f}%"
                status_html = f"<b style='color: {color_hex}; font-size: 15px;'>{res['classification']}</b>"
                certainty_html = f"<b style='color: {color_hex}; font-size: 15px;'>{certainty_str}</b>"

                table_html = f"""
                <table class="minimal-table">
                    <tr>
                        <td class="col-label">Algoritmo Extrator:</td>
                        <td>Deep FaceNet (InceptionResnetV1 512d - VGGFace2)</td>
                    </tr>
                    <tr>
                        <td class="col-label">Similaridade de Cosseno:</td>
                        <td><b>{res['cosine_sim']:.4f}</b></td>
                    </tr>
                    <tr>
                        <td class="col-label">Distância Euclidiana (L2):</td>
                        <td><b>{res['euclidean_dist']:.4f}</b></td>
                    </tr>
                    <tr>
                        <td class="col-label">Grau de Certeza Estima:</td>
                        <td>{certainty_html}</td>
                    </tr>
                    <tr>
                        <td class="col-label">Resultado da Análise:</td>
                        <td>{status_html}</td>
                    </tr>
                    <tr>
                        <td class="col-label">Fundamentação:</td>
                        <td>{res['reason']}</td>
                    </tr>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)

                # 3. PARECER TÉCNICO
                st.markdown('<div class="clean-section-title">3. PARECER TÉCNICO PERICIAL / OBSERVAÇÕES</div>', unsafe_allow_html=True)
                
                parecer_text = (
                    f"Confronto facial realizado via algoritmo Deep FaceNet (InceptionResnetV1). "
                    f"Constatada similaridade de cosseno de {res['cosine_sim']:.4f} com grau de certeza estimado em {res['certainty_pct']:.1f}%. "
                    f"Resultado conclusivo para {res['classification']}."
                )

                tech_resources_html = """
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #1f2937; font-size: 13px; color: #94a3b8;">
                    <b style="color: #e2e8f0;">🛠️ RECURSOS TÉCNICOS UTILIZADOS:</b><br/>
                    • <b>Extrator Biométrico:</b> Deep Learning FaceNet (InceptionResnetV1) de 512 dimensões pré-treinado em VGGFace2 com vetorização L2 e amostragem multi-crop Flip-Invariance.<br/>
                    • <b>Detector Anatômico:</b> MTCNN (Multi-task Cascaded Convolutional Networks) com localização de 5 pontos biométricos (olhos, nariz e cantos da boca).<br/>
                    • <b>Métricas de Convergência:</b> Similaridade de Cosseno no hiperespaço 512D e Distância Euclidiana L2 com modelo de calibração de certeza.<br/>
                    • <b>Filtros de Qualidade:</b> Avaliação de nitidez facial por variância do operador Laplaciano (Laplacian Blur Score) e validação de geometria anatômica.
                </div>
                """

                st.markdown(f"""
                <div class="minimal-parecer">
                    {parecer_text}
                    {tech_resources_html}
                </div>
                """, unsafe_allow_html=True)

                # BOTÃO DE PDF (FOTO ESTÁTICA: SEMPRE GERA O RELATÓRIO INDEPENDENTE DO RESULTADO)
                st.markdown("<br>", unsafe_allow_html=True)
                col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
                with col_pdf2:
                    pdf_bytes = generate_pdf_report(
                        img_a, img_b, crop_a_hud, crop_b_hud, res, 
                        procedure_id="CONFRONTO-BIOMÉTRICO", 
                        expert_name="Analista Biométrico", 
                        notes=parecer_text
                    )
                    st.download_button(
                        label="📄 Baixar Relatório da Comparação (.PDF)",
                        data=pdf_bytes,
                        file_name=f"Relatorio_Confronto_Facial_{int(res['certainty_pct'])}pct.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

else:
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; color: #64748b; font-size: 14px; background: #111827; border-radius: 8px; border: 1px dashed #1f2937; margin-top: 10px;">
        💡 Faça o upload da <b>Foto A</b> (Referência) e da <b>Mídia B</b> (Foto ou Vídeo de CFTV) acima para iniciar.
    </div>
    """, unsafe_allow_html=True)
