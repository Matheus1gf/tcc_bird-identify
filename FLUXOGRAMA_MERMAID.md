# 📊 FLUXOGRAMA EM FORMATO MERMAID

## Versão para uso em apresentações e documentação

### Fluxograma Completo

```mermaid
flowchart TD
    A[Entrada de Imagem] --> B[Pré-processamento]
    B --> C[YOLO Detecta Objetos]
    C --> D{YOLO Detectou?}
    D -->|Sim| E[Keras Classifica Espécie]
    D -->|Não| F[Keras tem Intuição?]
    F -->|Não| G[Resultado: Não Identificado]
    F -->|Sim| H[IntuitionEngine Analisa]
    H --> I[Grad-CAM Gera Anotação]
    I --> J[Validação Híbrida]
    J --> K{Decisão Automatizada}
    K -->|API: Sim + Grad-CAM Forte| L[AUTO-APROVAÇÃO]
    K -->|API: Não| M[AUTO-REJEIÇÃO]
    K -->|API: Sim mas Grad-CAM Fraco| N[REVISÃO HUMANA]
    L --> O[Adiciona ao Dataset]
    N --> P{Decisão Humana}
    P -->|Aprovado| O
    P -->|Rejeitado| Q[Fim do Processo]
    M --> Q
    O --> R[Atualiza Contadores]
    R --> S{Threshold Atingido?}
    S -->|Não| T[Aguarda Mais Dados]
    S -->|Sim| U[Re-treinamento Automático]
    U --> V[Prepara Dataset]
    V --> W[Treina YOLO]
    W --> X[Treina Keras]
    X --> Y[Avalia Performance]
    Y --> Z{Melhoria > Threshold?}
    Z -->|Sim| AA[Substitui Modelos]
    Z -->|Não| AB[Mantém Modelos Anteriores]
    AA --> AC[Atualiza Métricas]
    AB --> AC
    AC --> AD[Próxima Imagem]
    E --> AE[Resultado Final]
    T --> AD
    AD --> A
    AE --> Q
    G --> Q
```

### Fluxograma Simplificado (Para Apresentação)

```mermaid
flowchart LR
    A[Imagem] --> B[YOLO]
    B --> C{YOLO OK?}
    C -->|Sim| D[Keras Classifica]
    C -->|Não| E[Intuição?]
    E -->|Sim| F[Grad-CAM]
    E -->|Não| G[Fim]
    F --> H[Validação]
    H --> I{Aprovado?}
    I -->|Sim| J[Adiciona Dataset]
    I -->|Não| G
    J --> K{Threshold?}
    K -->|Sim| L[Re-treina]
    K -->|Não| M[Aguarda]
    L --> N[Melhoria]
    D --> O[Resultado]
```

### Fluxograma do Sistema Santo Graal

```mermaid
flowchart TD
    A[Nova Imagem] --> B[YOLO Falha]
    B --> C[Keras tem Intuição]
    C --> D[Grad-CAM Gera Anotação]
    D --> E[API Valida: É pássaro?]
    E --> F{Decisão}
    F -->|API: Sim + Grad-CAM Forte| G[AUTO-APROVAÇÃO]
    F -->|API: Não| H[AUTO-REJEIÇÃO]
    F -->|API: Sim mas Grad-CAM Fraco| I[REVISÃO HUMANA]
    G --> J[Adiciona ao Dataset]
    I --> K{Humano Aprova?}
    K -->|Sim| J
    K -->|Não| L[Rejeitado]
    J --> M{15+ Anotações?}
    M -->|Sim| N[Re-treinamento]
    M -->|Não| O[Aguarda]
    N --> P[Modelos Melhorados]
    P --> Q[Próxima Imagem]
    O --> Q
    H --> Q
    L --> Q
    Q --> A
```

### Fluxograma do Ciclo de Aprendizado Contínuo

```mermaid
flowchart TD
    A[Detecção de Intuição] --> B[Geração de Anotações]
    B --> C[Validação Híbrida]
    C --> D[Execução de Decisões]
    D --> E[Armazenamento]
    E --> F{Threshold?}
    F -->|Não| G[Aguarda]
    F -->|Sim| H[Re-treinamento]
    H --> I[Avaliação]
    I --> J{Melhoria?}
    J -->|Sim| K[Atualiza Modelos]
    J -->|Não| L[Mantém Modelos]
    K --> M[Próximo Ciclo]
    L --> M
    G --> M
    M --> A
```

## 📝 Como Usar

### Para Apresentação PowerPoint:
1. Copie o código Mermaid
2. Use ferramentas como:
   - [Mermaid Live Editor](https://mermaid.live/)
   - [Draw.io](https://app.diagrams.net/) (suporta Mermaid)
   - Extensões do VS Code para Mermaid
3. Exporte como imagem PNG/SVG
4. Insira na apresentação

### Para Documentação:
- O formato Mermaid é suportado nativamente em:
  - GitHub
  - GitLab
  - Notion
  - Muitos editores Markdown modernos

### Alternativa: Diagrama em Texto
- Use o arquivo `FLUXOGRAMA_PROCESSO.md` para versão em texto ASCII
- Pode ser convertido para diagrama visual usando ferramentas online

---

**Versão:** 1.0  
**Formato:** Mermaid Diagram  
**Compatibilidade:** GitHub, GitLab, Notion, VS Code, e muitos outros

