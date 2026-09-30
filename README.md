# 🛡️ Comparador Facial Pericial (Uso Policial)

Aplicação web desktop de alta precisão para comparação biométrica facial e emissão de laudo pericial, desenvolvida em Python com **Streamlit**, **PyTorch**, **FaceNet (InceptionResnetV1)** e **MTCNN**.

---

## 🚀 Como Executar o Aplicativo

### 1. Abrir o Terminal na pasta do projeto
No prompt de comando (CMD ou PowerShell), navegue até a pasta do projeto:
```bash
cd "c:\Users\luiss\OneDrive\Área de Trabalho\Comparador Facial"
```

### 2. Iniciar o Streamlit
Execute o comando abaixo:
```bash
streamlit run app.py
```
O navegador será aberto automaticamente no endereço `http://localhost:8501`.

---

## 🛠️ Principais Recursos

1. **📊 Confronto Biométrico & Grau de Certeza (%):**
   - Extração de embeddings faciais de 512 dimensões com Deep FaceNet (treinado em VGGFace2).
   - Detecção automática de rostos e 5 landmarks anatômicos (olhos, nariz, cantos da boca) com MTCNN.
   - Métricas periciais: **Similaridade de Cosseno** e **Distância Euclidiana (L2)**.
   - Níveis de rigor selecionáveis na barra lateral (*Rigoroso 80%+*, *Padrão 70%+*, *Permissivo 60%+*).

2. **🔬 Análise Forense Avançada:**
   - **Filtros de Tratamento de Imagem:** Brilho, contraste, nitidez, conversão para escala de cinza e **Equalização CLAHE** (ideal para imagens escuras ou de câmeras de segurança/CFTV).
   - **Inspeção de Sobreposição Anatômica (Cross-fade / Transparência):** Slider interativo para intercalar entre Foto A e Foto B recortadas e alinhadas para verificação visual da anatomia facial.

3. **📄 Emissão de Laudo Pericial Oficial (PDF):**
   - Geração de documento oficial em formato PDF com hashes SHA-256 das fotos (cadeia de custódia), data/hora, metadados, recortes biométricos, tabela de métricas e campo para parecer técnico do perito criminal / analista policial.
